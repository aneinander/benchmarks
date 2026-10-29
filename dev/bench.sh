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
outs = []
try:
    for _ in range(3):
        req = urllib.request.Request(os.environ["ENDPOINT"].rstrip("/") + "/chat/completions",
                                     body, hdr)
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["choices"][0]["message"]
        # thinking models may spend all 128 tokens on reasoning
        outs.append((msg.get("reasoning_content") or msg.get("reasoning") or "") +
                    "\0" + (msg.get("content") or ""))
except Exception as e:
    print("greedy_check=failed")
    print("greedy_error=" + " ".join(str(e).split())[:200])
else:
    hashes = [hashlib.sha256(o.encode()).hexdigest()[:12] for o in outs]
    print("greedy_check=" + ("identical" if len(set(hashes)) == 1 else "differs"))
    print(f"greedy_variants={len(set(hashes))}\ngreedy_runs={len(hashes)}\ngreedy_hash={hashes[0]}")
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
  printf 'betterbench=%s\n' "$have_ver"
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
python3 - "$tmp/mandatory" "$tmp/collected" "$tmp/measured" "$tmp/cmd.red" "$tmp/launch.json" \
  > "$tmp/notes" <<'PY'
import json, shlex, sys
def kv(path):
    with open(path, encoding="utf-8") as f:
        return dict(l.rstrip("\n").split("=", 1) for l in f if "=" in l)
m, c, meas = kv(sys.argv[1]), kv(sys.argv[2]), kv(sys.argv[3])
cmd = open(sys.argv[4], encoding="utf-8").read().strip()
def g(k):
    v = c.get(k, "")
    return "" if v in ("", "unknown") else v
def join(*parts):
    return " · ".join(p for p in parts if p) or "unknown"

vendor = {"intel": "Intel", "nvidia": "NVIDIA", "amd": "AMD"}.get(g("gpu_vendor"), g("gpu_vendor"))
count = g("gpu_count")
gpu = join(" ".join(p for p in (count and count + "x", vendor if vendor != "none" else "",
                                g("gpu_model")) if p),
           g("gpu_vram_gb") and g("gpu_vram_gb") + " GB",
           g("gpu_driver") and "driver " + g("gpu_driver"))

link, top = g("pcie_link"), g("pcie_max")
if m.get("pcie"):
    pcie = m["pcie"] + " (declared)"
elif link and top and link != top:
    pcie = f"{link} now, {top} max"   # idle GPUs drop the link speed
else:
    pcie = link or "unknown"
if g("rebar"):
    pcie += " · ReBAR " + g("rebar")

pl, det = m["power_limit"], g("gpu_power_limit_w")
if pl == "stock":
    power = join("stock", det and det + " W")
else:
    power = pl + " W"
    if det and abs(float(det) - float(pl)) > 1:
        power += f" (GPU reports {det} W)"

virt = g("virt")
host = join(g("cpu_model"), g("ram_gb") and g("ram_gb") + " GB RAM", g("os"),
            g("kernel") and "kernel " + g("kernel"), virt != "none" and virt and "VM " + virt)

# Settings in the launch command that swing results but no API reports:
# batch sizes, flash attention, KV types, env flags. Flags left out of the
# command are the engine's defaults, which change between versions.
try:
    words = shlex.split(cmd.replace("\\\n", " "))
except ValueError:
    words = cmd.split()
env, flags, prog = {}, {}, None
BOOL = {"--no-mmap", "--mlock", "--enforce-eager", "--enable-prefix-caching",
        "--no-enable-prefix-caching", "--enable-chunked-prefill",
        "--no-enable-chunked-prefill", "--cont-batching", "-cb", "--no-cont-batching", "-nocb",
        "-cmoe", "--cpu-moe", "-rm", "--rm", "-d", "--detach", "-it", "-i", "--privileged"}
FA_VALUES = {"on", "off", "auto", "1", "0", "true", "false"}
i = 0
while i < len(words):
    w = words[i]
    nxt = words[i + 1] if i + 1 < len(words) else None
    if prog is None and "=" in w and not w.startswith("-") and w.split("=", 1)[0].isidentifier():
        k, v = w.split("=", 1); env[k] = v                     # FOO=1 cmd
    elif prog is None and not w.startswith("-") and w != "env":
        prog = w
    elif w in ("-e", "--env") and nxt is not None and not nxt.startswith("-"):  # docker -e K=V
        k, _, v = nxt.partition("="); env[k] = v if _ else "(inherited)"; i += 1
    elif w.startswith("--env="):
        k, _, v = w[6:].partition("="); env[k] = v if _ else "(inherited)"
    elif w.startswith("-e") and len(w) > 2 and "=" in w and w[2:].split("=")[0].isidentifier():
        k, v = w[2:].split("=", 1); env[k] = v
    elif w.startswith("-"):
        if "=" in w and w.startswith("--"):
            k, v = w.split("=", 1); flags[k] = v
        elif w in ("-fa", "--flash-attn"):
            if nxt is not None and nxt.lower() in FA_VALUES:
                flags[w] = nxt; i += 1
            else:
                flags[w] = "on"                                   # older bare -fa
        elif w in BOOL or nxt is None or nxt.startswith("-"):
            flags[w] = True
        else:
            flags[w] = nxt; i += 1
    i += 1

def flag(*names):
    for n in names:
        if n in flags:
            return flags[n]
    return None

LLAMA = [("b", ("-b", "--batch-size")), ("ub", ("-ub", "--ubatch-size")),
         ("fa", ("-fa", "--flash-attn")), ("ctk", ("-ctk", "--cache-type-k")),
         ("ctv", ("-ctv", "--cache-type-v")), ("ngl", ("-ngl", "--gpu-layers", "--n-gpu-layers")),
         ("np", ("-np", "--parallel")), ("t", ("-t", "--threads")),
         ("sm", ("-sm", "--split-mode")), ("ts", ("-ts", "--tensor-split")),
         ("ot", ("-ot", "--override-tensor")), ("ncmoe", ("-ncmoe", "--n-cpu-moe")),
         ("cmoe", ("-cmoe", "--cpu-moe")), ("no-mmap", ("--no-mmap",)), ("mlock", ("--mlock",)),
         ("nocb", ("-nocb", "--no-cont-batching"))]
LLAMA_ALWAYS = {"b", "ub", "fa", "ctk", "ctv"}   # shown as "default" when not set
VLLM = ["--max-num-batched-tokens", "--max-num-seqs", "--kv-cache-dtype",
        "--gpu-memory-utilization", "--dtype", "--quantization", "--block-size",
        "--enforce-eager", "--enable-chunked-prefill", "--no-enable-chunked-prefill",
        "--enable-prefix-caching", "--no-enable-prefix-caching", "--attention-backend",
        "--compilation-config", "-O", "--speculative-config"]
launch_kind = g("engine")
if launch_kind not in ("llama.cpp", "vllm"):
    launch_kind = ("llama.cpp" if prog and "llama-server" in prog or flag("-ngl", "-ub", "-ctk")
                   else "vllm" if any(f in flags for f in VLLM) else "")
settings = {}
if launch_kind == "llama.cpp":
    for name, al in LLAMA:
        v = flag(*al)
        if v is not None:
            settings[name] = "on" if v is True else v
        elif name in LLAMA_ALWAYS:
            settings[name] = "default"
elif launch_kind == "vllm":
    for f in VLLM:
        if f in flags:
            settings[f.lstrip("-")] = "on" if flags[f] is True else "".join(str(flags[f]).split())
if not cmd:
    launch = "unknown (pass --server-cmd-file)"
else:
    launch = " · ".join(k if v == "on" and k in ("no-mmap", "mlock", "nocb", "cmoe") or
                        k.startswith(("enable-", "no-enable-", "enforce-")) else f"{k} {v}"
                        for k, v in settings.items()) or "no tuning flags"
# Offline switches and cache paths don't change the numbers.
QUIET = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HOME",
         "VLLM_CACHE_ROOT", "PATH", "HOME", "TZ", "LANG", "LC_ALL"}
env_flags = (" · ".join(f"{k}={v}" for k, v in env.items() if k not in QUIET) or "none"
             ) if cmd else "unknown"
json.dump({"engine": launch_kind or None, "settings": settings, "env": env, "flags": flags},
          open(sys.argv[5], "w", encoding="utf-8"), indent=2, default=str)

# The declared --kv should be what the command sets.
kvset = [x for x in (settings.get("ctk"), settings.get("ctv"), settings.get("kv-cache-dtype"))
         if x and x not in ("default", "auto")]
if kvset and m["kv"] not in kvset + ["/".join(kvset), "_".join(kvset)]:
    print(f"bench.sh: warning: --kv {m['kv']} but the launch command sets "
          f"{'/'.join(kvset)}", file=sys.stderr)

mb = g("model_bytes")
model_file = join(g("model_file"), mb and f"{int(mb) / 1e9:.2f} GB")
mem_bw = join(g("gpu_mem_bw_gbs") and
              f'{g("gpu_mem_bw_gbs")} GB/s measured ({g("gpu_mem_bw_tool") or "?"})',
              m.get("mem_bw") and m["mem_bw"] + " GB/s declared")
if meas.get("greedy_check") == "identical":
    greedy = f'identical {meas["greedy_runs"]}/{meas["greedy_runs"]} · {meas["greedy_hash"]}'
elif meas.get("greedy_check") == "differs":
    greedy = f'differs: {meas["greedy_variants"]} variants in {meas["greedy_runs"]} · first {meas["greedy_hash"]}'
else:
    greedy = "failed"

eng = g("engine")
engine = join(" ".join(p for p in (eng, g("engine_version")) if p),
              g("engine_slots") and g("engine_slots") + " slots",
              g("engine_image") and "image " + g("engine_image"),
              g("proxy") not in ("", "none") and "via " + g("proxy"))

notes = [
    ("submitter", m["submitter"]),
    ("gpu", gpu),
    ("pcie", pcie),
    ("power_limit", power),
    ("host", host),
    ("engine", engine),
    ("model_file", model_file),
    ("mem_bw", mem_bw),
    ("parallel", f"tp {m['tp']} · pp {m['pp']}"),
    ("precision", f"weights {m['quant']} · kv {m['kv']}"),
    ("launch", launch),
    ("env_flags", env_flags),
    ("spec", m["spec"]),
    ("thinking", m["thinking"]),
    ("greedy_check", greedy),
    ("run", join(m["run_mode"], "concurrency " + m["concurrency"],
                 "bench-standard " + m["bench_standard"], "betterbench " + m["betterbench"])),
]
if m.get("special_notes"):
    notes.append(("special_notes", m["special_notes"]))
for k, v in notes:
    print(f"{k}={v}")
PY

engine=$(get engine) gpu=$(get gpu_model) count=$(get gpu_count) ctx=$(get ctx)
[ "$engine" != unknown ] || echo "bench.sh: warning: engine not detected; result is hard to compare" >&2
if [ "$power_src" = declared ] && [ "$power_limit" != stock ] && matches "$pl" '[0-9.]+' &&
   [ "$(awk -v a="$pl" -v b="$power_limit" 'BEGIN{print (a-b>1||b-a>1)}')" = 1 ]; then
  echo "bench.sh: warning: --power-limit $power_limit W but the GPU reports $pl W" >&2
fi
matches "$count" '[2-9]|[1-9][0-9]+' && gpu="${count}x-$gpu"
slug() { tr 'A-Z' 'a-z' | sed -e 's/[^a-z0-9.]\{1,\}/-/g' -e 's/^-//' -e 's/-$//'; }
name=$(printf '%s-%s-%s' "$model" "$engine" "$gpu" | slug)

run_dir="$results_dir/$(date +%Y%m%d-%H%M%S)-$(printf '%s' "$submitter" | slug)-$name"

# ---------------------------------------------------------------- run
set -- run --no-update-check --endpoint "$endpoint" --model "$model" \
       --config "$run_dir/config.json" --name "$name" --out "$run_dir/results.json"
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
{ PYTHONUNBUFFERED=1 betterbench "$@" 2>&1 || echo $? > "$tmp/bb_rc"; } |
  tee "$run_dir/betterbench.log" |
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

# Energy per token and decode as % of memory bandwidth, from what the run
# measured. They come after the run, so they go to meta.json ("post_run"),
# not into the notes of results.json.
T0=$t_start T1=$t_end python3 - "$run_dir" <<'PY' || echo "bench.sh: warning: post-run figures failed" >&2
import json, os, statistics, sys
d = sys.argv[1]
res = json.load(open(f"{d}/results.json", encoding="utf-8"))
meta = json.load(open(f"{d}/meta.json", encoding="utf-8"))
c, m = meta["collected"], meta["mandatory"]
post = {"wall_s": round(float(os.environ["T1"]) - float(os.environ["T0"]), 1)}

# Output tokens of the measured passes and the concurrency sweep; warm-up
# passes and prefill probes are not in results.json but draw power too.
single = [r for recs in (res.get("single_stream") or {}).values() for r in recs
          if r.get("ok", True)]
tok = sum(r.get("completion_tokens") or 0 for r in single)
tok += sum(round(x.get("aggregate_tps", 0) * x.get("wall_s", 0))
           for x in res.get("concurrency") or [])

rows = []
if os.path.exists(f"{d}/power.csv"):
    for line in open(f"{d}/power.csv"):
        p = line.strip().split(",")
        try:
            rows.append((float(p[0]), float(p[1]), float(p[2]) if p[2] else None,
                         float(p[3]) if len(p) > 3 and p[3] else 0.0))
        except (ValueError, IndexError):
            pass
if len(rows) >= 2:
    span = rows[-1][0] - rows[0][0]
    if rows[0][2] is not None and rows[-1][2] is not None and rows[-1][2] >= rows[0][2]:
        joules, src = rows[-1][2] - rows[0][2], "energy counter"
    else:
        joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(rows, rows[1:]))
        src = "integrated power"
    e = {"joules": round(joules), "source": src, "sampled_s": round(span, 1),
         "avg_w": round(joules / span, 1) if span else None,
         "peak_w": max(r[1] for r in rows), "output_tokens": tok,
         "j_per_output_token": round(joules / tok, 3) if tok else None}
    # A clock-gated idle GPU reads 0 MHz; average the samples where it runs.
    clk = [r[3] for r in rows if r[3] > 0]
    if clk:
        e.update(avg_mhz=round(statistics.mean(clk)), min_mhz=round(min(clk)),
                 max_mhz=round(max(clk)), clock_samples=f"{len(clk)}/{len(rows)}")
    post["energy"] = e

# Batch-1 decode reads all weights once per token, so decode t/s x weight
# bytes is the bandwidth it achieves. Dense models only; spec decoding or MoE
# can go above 100%.
tps = [r["decode_tps"] for r in single if r.get("decode_tps")]
# measured where there is a figure: what the card really reaches
bw = c.get("gpu_mem_bw_gbs") if c.get("gpu_mem_bw_gbs") not in (None, "", "unknown") else m.get("mem_bw")
mb = c.get("model_bytes")
if tps and bw not in (None, "", "unknown") and mb not in (None, "", "unknown"):
    med = statistics.median(tps)
    eff = med * int(mb) / 1e9
    post["roofline"] = {"decode_tps_median": round(med, 2), "model_bytes": int(mb),
                        "mem_bw_gbs": float(bw), "effective_gbs": round(eff, 1),
                        "pct_of_mem_bw": round(100 * eff / float(bw), 1)}

# The summary: headline numbers plus the setup that makes them comparable.
def med(xs):
    xs = [x for x in xs if x]
    return statistics.median(xs) if xs else None
def f(x, unit, d=0):
    return "—" if x is None else f"{x:,.{d}f} {unit}"
notes = meta["notes"]
L = [f"  {meta['model']} · {notes.get('engine', '?')}",
     f"  {notes.get('gpu', '?')} · {notes.get('power_limit', '?')} limit",
     f"  {notes.get('run', '?')} · greedy {notes.get('greedy_check', '?')}",
     f"  decode (batch 1)  {f(med(tps), 't/s', 1)}    TTFT p50  "
     f"{f(med(r.get('ttft_ms') for r in single), 'ms')}"]
conc = res.get("concurrency") or []
if conc:
    L.append("  concurrency       " + "   ".join(
        f"{x['level']}: {x.get('aggregate_tps', 0):,.0f} t/s" for x in conc))
live = [p for p in res.get("prefill") or [] if not p.get("skipped")]
if live:
    L.append("  prefill           " + "   ".join(
        f"{p['target_depth'] // 1000}k: {f(med(p.get('pp_tps') or []), 't/s')}" for p in live))
if "roofline" in post:
    r = post["roofline"]
    L.append(f"  bandwidth         {r['effective_gbs']:.0f} GB/s effective = "
             f"{r['pct_of_mem_bw']:.0f}% of {r['mem_bw_gbs']:.0f} GB/s ({notes.get('mem_bw', '')})")
if "energy" in post:
    e = post["energy"]
    clock = (f" · clock avg {e['avg_mhz']} MHz ({e['min_mhz']}-{e['max_mhz']})"
             if "avg_mhz" in e else "")
    L.append(f"  power             avg {e['avg_w']} W, peak {e['peak_w']:.0f} W{clock}")
    L.append(f"  energy            {e['joules'] / 1000:.1f} kJ, "
             f"{e['j_per_output_token']} J per output token (whole run)")
print("\n" + "\n".join(L), file=sys.stderr)

meta["post_run"] = post
with open(f"{d}/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY

# BetterBench with GGZ14/BetterBench#4 fingerprints PCIe and BAR itself
# (env.gpu.pci), on the machine BetterBench runs on. Say so when that is not
# the benchmarked GPU, or when it disagrees with collect-env.sh.
ENV_FILE=$env_file python3 - "$run_dir/results.json" "$tmp/collected" <<'PY' || true
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
