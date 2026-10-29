#!/bin/sh
# bench.sh - bench-standard: benchmark an LLM server the way everyone in the
# group does, with BetterBench 0.6.1+. Every result records its setup (GPU,
# PCIe, engine, launch flags, ...), so results stay comparable.
#
# A quick check (a few minutes), then the same with --full for the real run:
#   ./bench.sh --endpoint http://localhost:8080/v1 --model MODEL_ID \
#     --tp 1 --pp 1 --quant Q4_K_M --kv f16 --spec none --thinking default \
#     --submitter YOUR_NAME --server-cmd-file server-cmd.txt
#
# You declare (these can't be detected):
#   --tp N / --pp N      tensor / pipeline parallel size
#   --quant Q            weight quantization: Q4_K_M, UD-Q4_K_M, w4a16, bf16
#   --kv KV              KV cache type: f16, bf16, q8_0
#   --spec S             speculative decoding: none, or METHOD:N with METHOD
#                        mtp, eagle, eagle3, ngram or draft (e.g. mtp:5)
#   --thinking T         thinking mode set on the server: off, on, low, medium,
#                        high or default
#   --submitter NAME     you
#   --server-cmd-file F  the command that started the server (not needed
#                        behind llama-swap, which reports it)
#
# Options:
#   --full               the run that counts (default: a quick check)
#   --verbose            show all of BetterBench's output
#   --dry-run            detect everything, print what would run, stop
#   --mem-bw GBPS        the card's spec memory bandwidth (measured anyway)
#   --power-limit W      when it can't be read from the GPU, or 'stock'
#   --pcie "Gen4 x16"    inside a VM, where the link can't be read
#   --special-notes TXT  anything unusual: custom kernels, undervolt, ...
#   --api-key KEY        or $BETTERBENCH_API_KEY; never stored
#   --no-concurrency     skip the concurrency sweep
#   --results-dir DIR    default: results/ next to bench.sh
#   --env-file FILE      when BetterBench runs on another machine than the GPU:
#                        the output of 'bench.sh collect-env' run on the GPU box
#
# Other commands:
#   bench.sh collect-env --endpoint URL --model ID   print the detected setup
#   bench.sh power [SECONDS]   sample GPU power, energy and clock until killed
#   bench.sh --version
#
# Values can also come from the environment: BENCH_TP, BENCH_PP, BENCH_QUANT,
# BENCH_KV, BENCH_SPEC, BENCH_THINKING, BENCH_SUBMITTER, BENCH_POWER_LIMIT,
# BENCH_PCIE, BENCH_MEM_BW, BENCH_SPECIAL_NOTES, BENCH_RESULTS_DIR. Machine-
# local defaults like these go in bench.local.env next to bench.sh.
# Send the whole results/<run>/ folder. Only --full runs count.
# The standard models and their hashes: README.md. For agents: AGENTS.md.

set -eu
umask 002
# here: the helper files. home: where results and bench.local.env go - next to
# the bench.sh people run, which unpacks the helpers to a temp dir.
here=$(cd "$(dirname "$0")" && pwd)
home=${BENCH_HOME:-$here}
. "$here/standard.env"
# Machine-local defaults (e.g. BENCH_RESULTS_DIR), not part of the standard.
[ -f "$home/bench.local.env" ] && . "$home/bench.local.env"

die() { echo "bench.sh: $*" >&2; exit 1; }
usage() { sed -n '2,/^$/p' "$0" | sed -e 's/^# \{0,1\}//' -e '$d'; exit "${1:-0}"; }

case "${1:-}" in
  collect-env) shift; exec sh "$here/collect-env.sh" "$@" ;;
  power)       shift; exec sh "$here/gpu-power.sh" "$@" ;;
  --version)   echo "bench-standard $STANDARD_VERSION (BetterBench $BETTERBENCH_MIN_VERSION or newer)"; exit 0 ;;
esac

endpoint="" model="" server_cmd_file="" env_file="" quick=1 no_conc=0 dry=0 verbose=0
tp=${BENCH_TP:-} pp=${BENCH_PP:-} quant=${BENCH_QUANT:-} kv=${BENCH_KV:-}
spec=${BENCH_SPEC:-} thinking=${BENCH_THINKING:-} power_limit=${BENCH_POWER_LIMIT:-}
submitter=${BENCH_SUBMITTER:-} pcie=${BENCH_PCIE:-} special=${BENCH_SPECIAL_NOTES:-}
mem_bw=${BENCH_MEM_BW:-}
results_dir=${BENCH_RESULTS_DIR:-$home/results}

need() { [ "$1" -ge 2 ] || die "$2 needs a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint)        need $# "$1"; endpoint=$2; shift 2 ;;
    --model)           need $# "$1"; model=$2; shift 2 ;;
    --api-key)         need $# "$1"; BETTERBENCH_API_KEY=$2; export BETTERBENCH_API_KEY; shift 2 ;;
    --server-cmd-file) need $# "$1"; server_cmd_file=$2; shift 2 ;;
    --env-file)        need $# "$1"; env_file=$2; shift 2 ;;
    --results-dir)     need $# "$1"; results_dir=$2; shift 2 ;;
    --tp)              need $# "$1"; tp=$2; shift 2 ;;
    --pp)              need $# "$1"; pp=$2; shift 2 ;;
    --quant)           need $# "$1"; quant=$2; shift 2 ;;
    --kv)              need $# "$1"; kv=$2; shift 2 ;;
    --spec)            need $# "$1"; spec=$2; shift 2 ;;
    --thinking)        need $# "$1"; thinking=$2; shift 2 ;;
    --power-limit)     need $# "$1"; power_limit=$2; shift 2 ;;
    --submitter)       need $# "$1"; submitter=$2; shift 2 ;;
    --pcie)            need $# "$1"; pcie=$2; shift 2 ;;
    --special-notes)   need $# "$1"; special=$2; shift 2 ;;
    --mem-bw)          need $# "$1"; mem_bw=$2; shift 2 ;;
    --quick)           quick=1; shift ;;
    --full)            quick=0; shift ;;
    --no-concurrency)  no_conc=1; shift ;;
    --dry-run)         dry=1; shift ;;
    --verbose)         verbose=1; shift ;;
    -h|--help)         usage 0 ;;
    *) echo "bench.sh: unknown argument: $1 (other BetterBench flags are not allowed)" >&2; usage 2 ;;
  esac
done

# ---------------------------------------------------------------- checks
matches() { printf '%s\n' "$1" | grep -Eqx "$2"; }

command -v python3 >/dev/null || die "python3 not found"
have_ver=$(betterbench --version 2>/dev/null | awk '{print $NF}') || have_ver=""
python3 -c 'import re, sys
v = lambda s: tuple(int(x) for x in re.findall(r"\d+", s)[:3])
sys.exit(not sys.argv[1] or v(sys.argv[1]) < v(sys.argv[2]))' "$have_ver" "$BETTERBENCH_MIN_VERSION" ||
  die "BetterBench $BETTERBENCH_MIN_VERSION or newer required, found '${have_ver:-not installed}'" \
      "(see https://github.com/GGZ14/BetterBench)"
command -v curl >/dev/null || die "curl not found"
# The exact build: two installs can both say 0.6.1. pip records where it came
# from: a git commit, or a local checkout (then git describes it).
bb_build=$(bb=$(command -v betterbench) && py=$(sed -n '1s/^#!//p' "$(readlink -f "$bb")") &&
  $py -c 'import importlib.metadata as m, json
u = json.loads(m.distribution("betterbench").read_text("direct_url.json") or "{}")
c = (u.get("vcs_info") or {}).get("commit_id")
print(c[:12] if c else u.get("url", "")[7:] if u.get("url", "").startswith("file://") else "")' 2>/dev/null) || bb_build=""
if [ -d "$bb_build" ]; then
  bb_build=$(git -c safe.directory='*' -C "$bb_build" describe --tags --always --dirty 2>/dev/null) || bb_build=""
fi

missing=""
for f in endpoint model tp pp quant kv spec thinking submitter; do
  eval "v=\${$f}"
  [ -n "$v" ] || missing="$missing --$(printf '%s' "$f" | tr _ -)"
done
[ -z "$missing" ] || die "missing mandatory value(s):$missing"

matches "$tp" '[1-9][0-9]*'               || die "--tp must be a positive integer"
matches "$pp" '[1-9][0-9]*'               || die "--pp must be a positive integer"
matches "$spec" 'none|(mtp|eagle|eagle3|ngram|draft):[1-9][0-9]*' ||
  die "--spec must be none or METHOD:N (mtp|eagle|eagle3|ngram|draft), e.g. mtp:5"
matches "$thinking" 'off|on|low|medium|high|default' ||
  die "--thinking must be off|on|low|medium|high|default (the server-side mode)"
[ -z "$power_limit" ] || matches "$power_limit" 'stock|[0-9]+(\.[0-9]+)?' ||
  die "--power-limit must be watts or 'stock'"
[ -z "$pcie" ] || matches "$pcie" 'Gen[1-6] x(1|2|4|8|16|32)' || die "--pcie must look like 'Gen4 x16'"
[ -z "$mem_bw" ] || matches "$mem_bw" '[1-9][0-9]*(\.[0-9]+)?' || die "--mem-bw must be GB/s, e.g. 456"
case "$special" in *"
"*) die "--special-notes must be a single line" ;; esac
for f in quant kv submitter; do
  eval "v=\${$f}"
  matches "$v" '[^[:space:]=]+' || die "--$f must be one word without '='"
done
if [ -n "$server_cmd_file" ]; then [ -r "$server_cmd_file" ] || die "cannot read $server_cmd_file"; fi
if [ -n "$env_file" ]; then [ -r "$env_file" ] || die "cannot read $env_file"; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

hdr="$tmp/hdr"
if [ -n "${BETTERBENCH_API_KEY:-}" ]; then printf 'Authorization: Bearer %s\n' "$BETTERBENCH_API_KEY" > "$hdr"
else printf 'Accept: application/json\n' > "$hdr"; fi

# ---------------------------------------------------------------- warm up
# The first request through llama-swap loads the model (minutes); it must not
# land inside a timed pass, and the engine probe needs it loaded anyway.
echo "warming up $model (a cold model behind llama-swap takes minutes to load) ..." >&2
body=$(python3 -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 1,
  "messages": [{"role": "user", "content": "hi"}]}))' "$model")
code=$(curl -s -m "${WARM_TIMEOUT:-900}" -o "$tmp/warm" -w '%{http_code}' -H @"$hdr" \
       -H 'Content-Type: application/json' -d "$body" "${endpoint%/}/chat/completions") || code=000
case "$code" in 2*) ;; *) die "warm-up request failed (HTTP $code): $(head -c 300 "$tmp/warm" 2>/dev/null)" ;; esac

# ---------------------------------------------------------------- greedy check
# The same prompt three times at temperature 0 must give the same text. The
# hash of that text also lets two setups of the same model be compared: a
# different hash means different numerics (quant, kernels, spec decoding bug).
# Whether the replies hold reasoning is checked against --thinking below.
echo "greedy check: 3 requests at temperature 0 ..." >&2
ENDPOINT=$endpoint MODEL=$model python3 - > "$tmp/measured" <<'PY'
import hashlib, json, os, urllib.request
PROMPT = ("List the first twelve prime numbers, then explain in three sentences "
          "why there are infinitely many primes.")
body = json.dumps({"model": os.environ["MODEL"], "temperature": 0, "seed": 0,
                   "max_tokens": 128, "stream": False,
                   "messages": [{"role": "user", "content": PROMPT}]}).encode()
hdr = {"Content-Type": "application/json"}
if os.environ.get("BETTERBENCH_API_KEY"):
    hdr["Authorization"] = "Bearer " + os.environ["BETTERBENCH_API_KEY"]
outs, thought = [], []
try:
    for _ in range(3):
        req = urllib.request.Request(os.environ["ENDPOINT"].rstrip("/") + "/chat/completions",
                                     body, hdr)
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["choices"][0]["message"]
        # thinking models may spend all 128 tokens on reasoning
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        content = msg.get("content") or ""
        outs.append(reasoning + "\0" + content)
        # a separate reasoning field (server with a reasoning parser), or
        # think tags left in the text (server without one)
        thought.append(bool(reasoning.strip()) or "</think>" in content or "<think>" in content)
except Exception as e:
    print("greedy_check=failed")
    print("greedy_error=" + " ".join(str(e).split())[:200])
else:
    hashes = [hashlib.sha256(o.encode()).hexdigest()[:12] for o in outs]
    print("greedy_check=" + ("identical" if len(set(hashes)) == 1 else "differs"))
    print(f"greedy_variants={len(set(hashes))}\ngreedy_runs={len(hashes)}\ngreedy_hash={hashes[0]}")
    print("greedy_reasoning=" + ("yes" if all(thought) else "no" if not any(thought) else "mixed"))
PY

# ---------------------------------------------------------------- collect
if [ -n "$env_file" ]; then
  cp "$env_file" "$tmp/env"
else
  set -- --endpoint "$endpoint" --model "$model" --server-cmd-out "$tmp/cmd.auto"
  [ -n "$server_cmd_file" ] && set -- "$@" --server-cmd-file "$server_cmd_file"
  sh "$here/collect-env.sh" "$@" > "$tmp/env"
fi

get() { sed -n "s/^$1=//p" "$tmp/env" | head -n1; }
pl=$(get gpu_power_limit_w)
power_src=declared
if [ -z "$power_limit" ]; then
  matches "$pl" '[0-9]+(\.[0-9]+)?' ||
    die "GPU power limit not detected here; pass --power-limit W (or 'stock')"
  power_limit=$pl power_src=detected
fi

# Raw declared values; meta.json keeps these and the collected ones as is.
{
  printf 'submitter=%s\ntp=%s\npp=%s\nquant=%s\nkv=%s\nspec=%s\nthinking=%s\npower_limit=%s\n' \
    "$submitter" "$tp" "$pp" "$quant" "$kv" "$spec" "$thinking" "$power_limit"
  printf 'power_limit_source=%s\n' "$power_src"
  [ -n "$pcie" ] && printf 'pcie=%s\n' "$pcie"
  [ -n "$special" ] && printf 'special_notes=%s\n' "$special"
  [ -n "$mem_bw" ] && printf 'mem_bw=%s\n' "$mem_bw"
  printf 'betterbench=%s\nbetterbench_build=%s\n' "$have_ver" "$bb_build"
  printf 'bench_standard=%s\nrun_mode=%s\nconcurrency=%s\n' "$STANDARD_VERSION" \
    "$([ $quick = 1 ] && echo quick || echo full)" "$([ $no_conc = 1 ] && echo off || echo on)"
} > "$tmp/mandatory"
grep -Ev '^[[:space:]]*(#|$)' "$tmp/env" > "$tmp/collected" || true
if grep -Evq '^[a-z0-9_]+=' "$tmp/collected"; then
  die "malformed line in collected env (want key=value, lowercase keys)"
fi
for k in $(cut -d= -f1 "$tmp/mandatory"); do
  if grep -q "^$k=" "$tmp/collected"; then die "collected env must not set mandatory key '$k'"; fi
done

# Launch command: the user's file wins, else what llama-swap reports.
# Secrets in env vars / flags are redacted either way.
cmd_src=""
if [ -n "$server_cmd_file" ]; then cmd_src="file:$(basename "$server_cmd_file")"; cp "$server_cmd_file" "$tmp/cmd"
elif [ -s "$tmp/cmd.auto" ]; then cmd_src="llama-swap /running"; cp "$tmp/cmd.auto" "$tmp/cmd"; fi
: > "$tmp/cmd.red"
if [ -n "$cmd_src" ]; then
  python3 - "$tmp/cmd" "$tmp/cmd.red" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8", errors="replace").read()
s = re.sub(r'(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY)[A-Z0-9_]*=)\S+', r'\1<redacted>', s)
s = re.sub(r'(?i)(--(?:api-key|hf-token|token)[= ])\S+', r'\1<redacted>', s)
open(sys.argv[2], "w", encoding="utf-8").write(s if s.endswith("\n") else s + "\n")
PY
fi

# The notes are grouped for reading: one chip per topic, nothing twice.
# The context size is left out - BetterBench shows it from --max-model-len.
# launch.json: the settings parsed from the launch command, for meta.json.
python3 "$here/notes.py" "$tmp/mandatory" "$tmp/collected" "$tmp/measured" "$tmp/cmd.red" \
  "$tmp/launch.json" "$here/models.tsv" > "$tmp/notes"

engine=$(get engine) gpu=$(get gpu_model) ctx=$(get ctx)
[ "$engine" != unknown ] || echo "bench.sh: warning: engine not detected; result is hard to compare" >&2
if [ "$power_src" = declared ] && [ "$power_limit" != stock ] && matches "$pl" '[0-9.]+' &&
   [ "$(awk -v a="$pl" -v b="$power_limit" 'BEGIN{print (a-b>1||b-a>1)}')" = 1 ]; then
  echo "bench.sh: warning: --power-limit $power_limit W but the GPU reports $pl W" >&2
fi
# File names: <YYMMDD>_<submitter>_<model>_<engine>_<N>x-<gpu>[_quick][_2],
# with N the GPUs the model runs on (tp x pp); _2, _3, ... for another run of
# the same setup on the same day. Fields are joined with _ since
# model names have dashes. The run folder and the exported results.json/.html
# are named alike, so a file sent on its own still says what it is.
slug() { tr 'A-Z' 'a-z' | sed -e 's/[^a-z0-9.]\{1,\}/-/g' -e 's/^-//' -e 's/-$//'; }
gpu=$(printf '%s' "$gpu" | slug)
name="$(printf '%s' "$model" | slug)_$(printf '%s' "$engine" | slug)_$((tp * pp))x-${gpu:-unknown}"
run_id="$(date +%y%m%d)_$(printf '%s' "$submitter" | slug)_$name"
[ $quick = 1 ] && run_id="${run_id}_quick"
base=$run_id i=1
while [ -e "$results_dir/$run_id" ] || [ -e "$results_dir/$run_id.failed" ]; do
  i=$((i + 1)); run_id="${base}_$i"
done
run_dir="$results_dir/$run_id"
results="$run_dir/$run_id.json"

# ---------------------------------------------------------------- run
set -- run --no-update-check --endpoint "$endpoint" --model "$model" \
       --config "$run_dir/config.json" --name "$name" --out "$results"
[ $quick = 1 ] && set -- "$@" --quick
[ $no_conc = 1 ] && set -- "$@" --no-concurrency
# Through a proxy /v1/models often lacks the context size; hand over what the
# engine itself reported so prefill depths that don't fit are skipped cleanly.
matches "$ctx" '[1-9][0-9]*' && set -- "$@" --max-model-len "$ctx"
while IFS= read -r line; do set -- "$@" --note "$line"; done < "$tmp/notes"

if [ $dry = 1 ]; then
  echo "dry run - would write $run_dir/ and run:"
  printf '  betterbench'
  for a; do case $a in *[[:space:]]*) printf " '%s'" "$a" ;; *) printf ' %s' "$a" ;; esac; done
  echo
  exit 0
fi

mkdir -p "$results_dir"
mkdir "$run_dir" || die "cannot create $run_dir"
cp "$here/config.json" "$run_dir/config.json"
[ -n "$cmd_src" ] && cp "$tmp/cmd.red" "$run_dir/server-cmd.txt"

STANDARD_VERSION=$STANDARD_VERSION CMD_SRC=$cmd_src ENDPOINT=$endpoint MODEL=$model NAME=$name NOTES=$tmp/notes \
MEASURED=$tmp/measured LAUNCH=$tmp/launch.json \
python3 - "$tmp/mandatory" "$tmp/collected" "$run_dir/meta.json" "$@" <<'PY'
import json, os, sys, time
def kv(path):
    with open(path, encoding="utf-8") as f:
        return dict(l.rstrip("\n").split("=", 1) for l in f if "=" in l)
src = os.environ["CMD_SRC"]
meta = {
    "bench_standard": os.environ["STANDARD_VERSION"],
    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "endpoint": os.environ["ENDPOINT"],
    "model": os.environ["MODEL"],
    "name": os.environ["NAME"],
    "notes": kv(os.environ["NOTES"]),
    "mandatory": kv(sys.argv[1]),
    "collected": kv(sys.argv[2]),
    "measured": kv(os.environ["MEASURED"]),
    "launch": json.load(open(os.environ["LAUNCH"], encoding="utf-8")),
    "server_cmd_file": "server-cmd.txt" if src else None,
    "server_cmd_source": src or None,
    "betterbench_argv": ["betterbench"] + sys.argv[4:],
}
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY

# Power is sampled where the GPU is: only when BetterBench runs on the server.
sampler=""
if [ -z "$env_file" ]; then
  sh "$here/gpu-power.sh" 2 > "$run_dir/power.csv" 2>/dev/null &
  sampler=$!
  trap 'kill "$sampler" 2>/dev/null || true; rm -rf "$tmp"' EXIT INT TERM
fi
echo "running betterbench -> $run_dir" >&2
[ $verbose = 1 ] && sed 's/^/  note: /' "$tmp/notes" >&2
t_start=$(date +%s.%N)
# Full output always goes to betterbench.log; the terminal gets all of it
# (--verbose) or the progress lines and anything that looks like trouble.
# timeline.tsv stamps each progress line, for the energy of each phase.
{ PYTHONUNBUFFERED=1 betterbench "$@" 2>&1 || echo $? > "$tmp/bb_rc"; } |
  tee "$run_dir/betterbench.log" |
  python3 -uc 'import sys, time
tl = open(sys.argv[1], "w", encoding="utf-8")
for line in iter(sys.stdin.readline, ""):
    if line.startswith("["):
        tl.write(f"{time.time():.3f}\t{line}"); tl.flush()
    sys.stdout.write(line); sys.stdout.flush()' "$run_dir/timeline.tsv" |
  if [ $verbose = 1 ]; then cat; else grep --line-buffered -E '^\[|^phases|warn|WARN|rror|Traceback' || true; fi >&2
if [ -s "$tmp/bb_rc" ]; then
  mv "$run_dir" "$run_dir.failed" 2>/dev/null || true
  die "betterbench failed (exit $(cat "$tmp/bb_rc")); output in $run_dir.failed/betterbench.log"
fi
t_end=$(date +%s.%N)
if [ -n "$sampler" ]; then
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true
  [ -s "$run_dir/power.csv" ] || rm -f "$run_dir/power.csv"
fi

# Energy (whole run and per phase) and decode as % of memory bandwidth, from
# what the run measured. They come after the run, so they go to meta.json
# ("post_run"), not into the notes of the results file.
T0=$t_start T1=$t_end python3 "$here/postrun.py" "$run_dir" "$results" ||
  echo "bench.sh: warning: post-run figures failed" >&2

# BetterBench with GGZ14/BetterBench#4 fingerprints PCIe and BAR itself
# (env.gpu.pci), on the machine BetterBench runs on. Say so when that is not
# the benchmarked GPU, or when it disagrees with collect-env.sh.
ENV_FILE=$env_file python3 - "$results" "$tmp/collected" <<'PY' || true
import json, os, sys
pci = ((json.load(open(sys.argv[1], encoding="utf-8")).get("env") or {}).get("gpu") or {}).get("pci")
if not pci:
    sys.exit()
with open(sys.argv[2], encoding="utf-8") as f:
    c = dict(l.rstrip("\n").split("=", 1) for l in f if "=" in l)
bdf, rebar = c.get("gpu_bdf", "unknown"), c.get("rebar", "unknown")
mine = next((g for g in pci if g.get("pci_address") == bdf), None)
if mine is None:
    where = "expected with --env-file" if os.environ["ENV_FILE"] else "different GPU or machine?"
    print(f"bench.sh: note: BetterBench's own GPU fingerprint (env.gpu.pci) lists "
          f"{', '.join(g.get('pci_address', '?') for g in pci)}, not the benchmarked GPU {bdf} "
          f"({where}); the gpu/pcie notes describe the server", file=sys.stderr)
elif rebar != "unknown" and mine.get("resizable_bar", "").split()[-1:] != [rebar]:
    print(f"bench.sh: warning: ReBAR {rebar} per collect-env.sh, but BetterBench reads "
          f"{mine.get('resizable_bar')} ({mine.get('largest_bar_mib')} MiB BAR)", file=sys.stderr)
PY
echo
echo "done: $run_dir"
ls -1 "$run_dir"
