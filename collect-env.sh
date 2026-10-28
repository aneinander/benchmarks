#!/bin/sh
# collect-env.sh - run ON THE INFERENCE SERVER. Prints a flat key=value list
# describing the hardware and the inference engine behind an endpoint.
# Never aborts on detection: anything it cannot find is reported as "unknown".
#
# usage: collect-env.sh [--endpoint URL] [--model ID] [--api-key KEY] [--server-cmd-out FILE]
#   --endpoint        OpenAI-compatible base, e.g. http://localhost:8080/v1 (omit: hardware only)
#   --model           model id as sent to the endpoint (needed behind llama-swap)
#   --server-cmd-out  behind llama-swap: write the model's launch command (raw) to FILE
# env: BETTERBENCH_API_KEY, PROBE_TIMEOUT (s, default 10), LOAD_TIMEOUT (s, default 600)

set -u
LC_ALL=C; export LC_ALL

endpoint="" model="" api_key="${BETTERBENCH_API_KEY:-}" cmd_out=""
PROBE_TIMEOUT=${PROBE_TIMEOUT:-10}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-600}

need() { [ "$1" -ge 2 ] || { echo "collect-env: $2 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint)       need $# "$1"; endpoint=$2; shift 2 ;;
    --model)          need $# "$1"; model=$2; shift 2 ;;
    --api-key)        need $# "$1"; api_key=$2; shift 2 ;;
    --server-cmd-out) need $# "$1"; cmd_out=$2; shift 2 ;;
    -h|--help)        sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "collect-env: unknown argument: $1" >&2; exit 2 ;;
  esac
done

tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT INT TERM
# Header via a file so the key never shows up in `ps`.
if [ -n "$api_key" ]; then printf 'Authorization: Bearer %s\n' "$api_key" > "$tmp/hdr"
else printf 'Accept: application/json\n' > "$tmp/hdr"; fi

have() { command -v "$1" >/dev/null 2>&1; }

# jget FILE EXPR - evaluate a python expression over the parsed JSON `j`
# (and `arg` = $JARG); prints nothing on any failure. Tolerates junk before
# the JSON (xpu-smi prints "[Error] ..." lines on stdout).
jget() {
  have python3 || return 0
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, os, sys
try:
    raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    try:
        j = json.loads(raw)
    except ValueError:
        j = json.loads(raw[raw.index("{"):])
    v = eval(sys.argv[2], {"j": j, "arg": os.environ.get("JARG", "")})
    if v is not None and v != "":
        print(" ".join(str(v).split()))
except Exception:
    pass
PY
}

# fetch URL OUTFILE [TIMEOUT] - succeeds only on HTTP 2xx.
fetch() {
  have curl || return 1
  code=$(curl -s -m "${3:-$PROBE_TIMEOUT}" -o "$2" -w '%{http_code}' -H @"$tmp/hdr" "$1" 2>/dev/null) || code=000
  case "$code" in 2*) return 0 ;; *) return 1 ;; esac
}

# Strip vendor boilerplate so the same card reads the same everywhere.
norm_gpu() {
  sed -e 's/([RT]M*)//g' -e 's/^ *Intel *//' -e 's/^ *NVIDIA *//' -e 's/^ *AMD *//' \
      -e 's/ Graphics *$//' -e 's/  */ /g' -e 's/^ //' -e 's/ $//'
}

kv() { printf '%s=%s\n' "$1" "$(printf '%s' "${2:-unknown}" | tr '\n\r' '  ')"; }

# ---------------------------------------------------------------- GPU
gpu_vendor=none gpu_model="" gpu_count=0 gpu_vram_gb="" gpu_driver="" gpu_power_limit_w=""
if have nvidia-smi && q=$(nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit \
                          --format=csv,noheader,nounits 2>/dev/null) && [ -n "$q" ]; then
  gpu_vendor=nvidia
  gpu_count=$(printf '%s\n' "$q" | grep -c .)
  gpu_model=$(printf '%s\n' "$q" | cut -d, -f1 | norm_gpu | sort -u | paste -sd+ -)
  first=$(printf '%s\n' "$q" | head -n1)
  gpu_vram_gb=$(printf '%s' "$first" | cut -d, -f2 | awk '$1+0>0 {printf "%d", $1/1024+0.5}')
  gpu_driver=$(printf '%s' "$first" | cut -d, -f3 | tr -d ' ')
  gpu_power_limit_w=$(printf '%s' "$first" | cut -d, -f4 | awk '$1+0>0 {printf "%g", $1}')
elif have rocm-smi && rocm-smi --showproductname --json > "$tmp/rp" 2>/dev/null; then
  gpu_vendor=amd
  rocm-smi --showmeminfo vram --json > "$tmp/rm" 2>/dev/null
  rocm-smi --showdriverversion --json > "$tmp/rd" 2>/dev/null
  rocm-smi --showmaxpower --json > "$tmp/rw" 2>/dev/null
  gpu_count=$(jget "$tmp/rp" 'len([k for k in j if k.startswith("card")])')
  gpu_model=$(jget "$tmp/rp" '"+".join(sorted({v.get("Card Series") or v.get("Card series") or v.get("Card model", "") for k, v in j.items() if k.startswith("card")}))' | norm_gpu)
  gpu_vram_gb=$(jget "$tmp/rm" 'round(int(next(v["VRAM Total Memory (B)"] for k, v in j.items() if k.startswith("card"))) / 2**30)')
  gpu_driver=$(jget "$tmp/rd" 'j["system"]["Driver version"]')
  gpu_power_limit_w=$(jget "$tmp/rw" '"%g" % float(next(v for k, v in next(c for n, c in j.items() if n.startswith("card")).items() if "Power" in k))')
elif have xpu-smi && xpu-smi discovery -j > "$tmp/xl" 2>/dev/null; then
  gpu_vendor=intel
  gpu_count=$(jget "$tmp/xl" 'len(j["device_list"])')
  gpu_model=$(jget "$tmp/xl" '"+".join(sorted({d["device_name"] for d in j["device_list"]}))' | norm_gpu)
  dev0=$(jget "$tmp/xl" 'j["device_list"][0]["device_id"]')
  if [ -n "$dev0" ]; then
    xpu-smi discovery -d "$dev0" -j > "$tmp/xd" 2>/dev/null
    gpu_vram_gb=$(jget "$tmp/xd" 'round(int(j["memory_physical_size_byte"]) / 2**30)')
    gpu_driver=$(jget "$tmp/xd" 'j["driver_version"]')
    # Newer xpu-smi takes named metrics; the old numeric -m syntax is gone.
    gpu_power_limit_w=$(xpu-smi dump --device "$dev0" --metrics power.limit --number 1 2>/dev/null \
                        | awk -F', *' 'NR>1 && $3+0>0 {printf "%g", $3; exit}')
  fi
fi
[ -n "$gpu_count" ] || gpu_count=unknown

# ---------------------------------------------------------------- host
cpu_model=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -n1)
[ -n "$cpu_model" ] || cpu_model=$(sed -n 's/^model name[[:space:]]*: *//p' /proc/cpuinfo 2>/dev/null | head -n1)
ram_gb=$(awk '/^MemTotal:/ {printf "%d", $2/1048576+0.5}' /proc/meminfo 2>/dev/null)
os=$( (. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-}") )
[ -n "$os" ] || os=$(uname -s 2>/dev/null)
kernel=$(uname -r 2>/dev/null)

# ---------------------------------------------------------------- engine
proxy=none engine=unknown engine_version="" model_file="" ctx="" engine_slots=""
if [ -n "$endpoint" ]; then
  base=${endpoint%/}; base=${base%/v1}
  prefix=$base
  t1=$PROBE_TIMEOUT
  if fetch "$base/running" "$tmp/running" && [ -n "$(jget "$tmp/running" '"running" in j and "y"')" ]; then
    proxy=llama-swap
    if [ -n "$model" ]; then
      prefix="$base/upstream/$model"
      export JARG="$model"
      state=$(jget "$tmp/running" 'next((r.get("state") for r in j["running"] if r.get("model") == arg), "")')
      if [ "$state" != ready ]; then
        echo "collect-env: $model is not loaded; llama-swap loads it on this probe (up to ${LOAD_TIMEOUT}s)" >&2
        t1=$LOAD_TIMEOUT
      fi
    fi
  fi

  if fetch "$prefix/props" "$tmp/props" "$t1" \
     && [ -n "$(jget "$tmp/props" '("build_info" in j or "default_generation_settings" in j) and "y"')" ]; then
    engine=llama.cpp
    engine_version=$(jget "$tmp/props" 'j.get("build_info")')
    model_file=$(jget "$tmp/props" 'j.get("model_path")')
    ctx=$(jget "$tmp/props" '(j.get("default_generation_settings") or {}).get("n_ctx") or j.get("n_ctx")')
    engine_slots=$(jget "$tmp/props" 'j.get("total_slots")')
  elif fetch "$prefix/version" "$tmp/version" "$t1" \
       && v=$(jget "$tmp/version" 'j.get("version")') && [ -n "$v" ]; then
    engine_version=$v
    if fetch "$prefix/v1/models" "$tmp/models"; then
      export JARG="$model"
      # Behind a proxy the served name differs from the routed id, so a single
      # entry is taken as is.
      pick='next((m for m in j["data"] if m.get("id") == arg), j["data"][0] if len(j["data"]) == 1 else {})'
      [ "$(jget "$tmp/models" "($pick).get('owned_by')")" = vllm ] && engine=vllm
      ctx=$(jget "$tmp/models" "($pick).get('max_model_len')")
      model_file=$(jget "$tmp/models" "($pick).get('root')")
    fi
  fi

  # Behind llama-swap the launch command is known: keep it, and map a
  # container path (-v HOST:/model) back to the host path.
  if [ "$proxy" = llama-swap ] && [ -n "$model" ] && fetch "$base/running" "$tmp/running"; then
    export JARG="$model"
    jget "$tmp/running" 'next((r.get("cmd") for r in j["running"] if r.get("model") == arg), "")' > "$tmp/cmd"
    if [ -s "$tmp/cmd" ]; then
      [ -n "$cmd_out" ] && cp "$tmp/cmd" "$cmd_out"
      if [ -n "$model_file" ] && have python3; then
        host=$(python3 - "$tmp/cmd" "$model_file" <<'PY' 2>/dev/null
import re, sys
cmd, path = open(sys.argv[1]).read(), sys.argv[2]
for src, dst in re.findall(r'(?:-v|--volume)[ =](\S+?):(/[^\s:]*)', cmd):
    if path == dst or path.startswith(dst.rstrip("/") + "/"):
        print(src + path[len(dst):]); break
PY
)
        [ -n "$host" ] && model_file=$host
      fi
    fi
  fi
fi

kv gpu_vendor "$gpu_vendor"
kv gpu_model "$gpu_model"
kv gpu_count "$gpu_count"
kv gpu_vram_gb "$gpu_vram_gb"
kv gpu_driver "$gpu_driver"
kv gpu_power_limit_w "$gpu_power_limit_w"
kv cpu_model "$cpu_model"
kv ram_gb "$ram_gb"
kv os "$os"
kv kernel "$kernel"
kv proxy "$proxy"
kv engine "$engine"
kv engine_version "$engine_version"
kv engine_slots "$engine_slots"
kv model_file "$model_file"
kv ctx "$ctx"
exit 0
