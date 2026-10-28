#!/bin/sh
# bench.sh - the single entry point for a standardized BetterBench run (see README.md).
#
# usage: bench.sh --endpoint URL --model ID
#                 --tp N --pp N --quant Q --kv KV --spec none|METHOD:N
#                 --thinking MODE --submitter NAME
#                 [--power-limit W|stock] [--api-key KEY] [--server-cmd-file FILE]
#                 [--env-file FILE] [--pcie GenN xW] [--special-notes TEXT]
#                 [--results-dir DIR] [--full] [--no-concurrency] [--dry-run]
#
# Default is a quick smoke run (BetterBench --quick); --full is the run that
# counts for submission.
# Mandatory values may also come from the environment: BENCH_TP, BENCH_PP,
# BENCH_QUANT, BENCH_KV, BENCH_SPEC, BENCH_THINKING, BENCH_SUBMITTER.
# --power-limit (or BENCH_POWER_LIMIT): read from the GPU when omitted; required
# only where it can't be detected. API key: --api-key or $BETTERBENCH_API_KEY.
# --pcie (or BENCH_PCIE): declare the slot link where it can't be read (VMs).
# --special-notes (or BENCH_SPECIAL_NOTES): free text for anything unusual
# about this run, e.g. "custom k8v4 kernels" or "undervolted".
# --env-file: output of collect-env.sh run on the server, when BetterBench
# runs on another machine (default: run collect-env.sh here).

set -eu
umask 002
here=$(cd "$(dirname "$0")" && pwd)
. "$here/standard.env"
# Machine-local defaults (e.g. BENCH_RESULTS_DIR), not part of the standard.
[ -f "$here/bench.local.env" ] && . "$here/bench.local.env"

die() { echo "bench.sh: $*" >&2; exit 1; }
usage() { sed -n '2,22p' "$0"; exit "${1:-0}"; }

endpoint="" model="" server_cmd_file="" env_file="" quick=1 no_conc=0 dry=0
tp=${BENCH_TP:-} pp=${BENCH_PP:-} quant=${BENCH_QUANT:-} kv=${BENCH_KV:-}
spec=${BENCH_SPEC:-} thinking=${BENCH_THINKING:-} power_limit=${BENCH_POWER_LIMIT:-}
submitter=${BENCH_SUBMITTER:-} pcie=${BENCH_PCIE:-} special=${BENCH_SPECIAL_NOTES:-}
results_dir=${BENCH_RESULTS_DIR:-$here/results}

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
    --quick)           quick=1; shift ;;
    --full)            quick=0; shift ;;
    --no-concurrency)  no_conc=1; shift ;;
    --dry-run)         dry=1; shift ;;
    -h|--help)         usage 0 ;;
    *) echo "bench.sh: unknown argument: $1 (other BetterBench flags are not allowed)" >&2; usage 2 ;;
  esac
done

# ---------------------------------------------------------------- checks
matches() { printf '%s\n' "$1" | grep -Eqx "$2"; }

have_ver=$(betterbench --version 2>/dev/null | awk '{print $NF}') || have_ver=""
[ "$have_ver" = "$BETTERBENCH_VERSION" ] ||
  die "BetterBench $BETTERBENCH_VERSION required, found '${have_ver:-not installed}'" \
      "(git checkout v$BETTERBENCH_VERSION && pip install -e .)"
command -v python3 >/dev/null || die "python3 not found"
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

# ---------------------------------------------------------------- collect
if [ -n "$env_file" ]; then
  cp "$env_file" "$tmp/env"
else
  sh "$here/collect-env.sh" --endpoint "$endpoint" --model "$model" \
     --server-cmd-out "$tmp/cmd.auto" > "$tmp/env"
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
# The notes are grouped for reading: one chip per topic, nothing twice.
# The context size is left out - BetterBench shows it from --max-model-len.
python3 - "$tmp/mandatory" "$tmp/collected" > "$tmp/notes" <<'PY'
import sys
def kv(path):
    with open(path, encoding="utf-8") as f:
        return dict(l.rstrip("\n").split("=", 1) for l in f if "=" in l)
m, c = kv(sys.argv[1]), kv(sys.argv[2])
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

eng = g("engine")
engine = join(" ".join(p for p in (eng, g("engine_version")) if p),
              g("engine_slots") and g("engine_slots") + " slots",
              g("proxy") not in ("", "none") and "via " + g("proxy"))

notes = [
    ("submitter", m["submitter"]),
    ("gpu", gpu),
    ("pcie", pcie),
    ("power_limit", power),
    ("host", host),
    ("engine", engine),
    ("model_file", g("model_file") or "unknown"),
    ("parallel", f"tp {m['tp']} · pp {m['pp']}"),
    ("precision", f"weights {m['quant']} · kv {m['kv']}"),
    ("spec", m["spec"]),
    ("thinking", m["thinking"]),
    ("run", join(m["run_mode"], "concurrency " + m["concurrency"],
                 "bench-standard " + m["bench_standard"])),
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
       --config "$here/config.json" --name "$name" --out "$run_dir/results.json"
[ $quick = 1 ] && set -- "$@" --quick
[ $no_conc = 1 ] && set -- "$@" --no-concurrency
# Through a proxy /v1/models often lacks the context size; hand over what the
# engine itself reported so prefill depths that don't fit are skipped cleanly.
matches "$ctx" '[1-9][0-9]*' && set -- "$@" --max-model-len "$ctx"
while IFS= read -r line; do set -- "$@" --note "$line"; done < "$tmp/notes"

if [ $dry = 1 ]; then
  echo "dry run - would write $run_dir/ and run:"
  printf '  betterbench'; printf ' %s' "$@"; echo
  exit 0
fi

mkdir -p "$results_dir"
mkdir "$run_dir" || die "cannot create $run_dir"

# Launch command: the user's file wins, else what llama-swap reports.
# Secrets in env vars / flags are redacted either way.
cmd_src=""
if [ -n "$server_cmd_file" ]; then cmd_src="file:$(basename "$server_cmd_file")"; cp "$server_cmd_file" "$tmp/cmd"
elif [ -s "$tmp/cmd.auto" ]; then cmd_src="llama-swap /running"; cp "$tmp/cmd.auto" "$tmp/cmd"; fi
if [ -n "$cmd_src" ]; then
  python3 - "$tmp/cmd" "$run_dir/server-cmd.txt" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8", errors="replace").read()
s = re.sub(r'(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY)[A-Z0-9_]*=)\S+', r'\1<redacted>', s)
s = re.sub(r'(?i)(--(?:api-key|hf-token|token)[= ])\S+', r'\1<redacted>', s)
open(sys.argv[2], "w", encoding="utf-8").write(s if s.endswith("\n") else s + "\n")
PY
fi

STANDARD_VERSION=$STANDARD_VERSION CMD_SRC=$cmd_src ENDPOINT=$endpoint MODEL=$model NAME=$name NOTES=$tmp/notes \
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
    "server_cmd_file": "server-cmd.txt" if src else None,
    "server_cmd_source": src or None,
    "betterbench_argv": ["betterbench"] + sys.argv[4:],
}
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PY

echo "running betterbench -> $run_dir" >&2
if ! betterbench "$@"; then
  mv "$run_dir" "$run_dir.failed" 2>/dev/null || true
  die "betterbench failed; partial output kept in $run_dir.failed"
fi
echo
echo "done: $run_dir"
ls -1 "$run_dir"
