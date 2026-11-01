#!/bin/sh
# shellcheck disable=SC2034  # variables are read inside the eval'd check strings
# test.sh - end-to-end tests of the one-file bench.sh, without a GPU: a fake
# OpenAI-compatible server (dev/test/fake_server.py) and a fake betterbench
# (dev/test/betterbench) stand in for the real ones.
#
# usage: sh dev/test.sh      (needs python3, curl; rebuilds nothing, checks the build)

set -u
dev=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$dev/.." && pwd)
t=$(mktemp -d)
pids=""
trap 'for p in $pids; do kill "$p" 2>/dev/null; done; rm -rf "$t"' EXIT
pass=0 fail=0
ok()  { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; [ -f "$t/out" ] && sed 's/^/     | /' "$t/out" | tail -n 25; }
check() { if (eval "$2"); then ok "$1"; else bad "$1"; fi; }   # check NAME SHELL-TEST

PATH="$dev/test:$PATH"; export PATH
MEASURE_BW=0 XDG_CACHE_HOME="$t/cache" BENCH_HOME="$t/home"; export MEASURE_BW XDG_CACHE_HOME BENCH_HOME
unset BETTERBENCH_API_KEY BENCH_SUBMITTER BENCH_QUANT BENCH_KV BENCH_SPEC BENCH_THINKING 2>/dev/null || true
mkdir -p "$t/home" "$t/models" "$t/tmpdir"
printf 'fake gguf weights\n' > "$t/models/test.gguf"

# serve PORT ENGINE REASONING MODEL_PATH
serve() {
  python3 "$dev/test/fake_server.py" "$@" & pids="$pids $!"
  i=0; until curl -s "http://127.0.0.1:$1/v1/models" >/dev/null || [ $i -gt 50 ]; do sleep 0.1; i=$((i + 1)); done
}
serve 18431 llama.cpp 0 "$t/models/test.gguf"
serve 18432 vllm 0 "Org/Repo"
serve 18433 llama.cpp 1 "/models/test.gguf"
LL=http://127.0.0.1:18431/v1 VL=http://127.0.0.1:18432/v1 LR=http://127.0.0.1:18433/v1

# bench ENDPOINT CMDFILE ARGS... - the one-file bench.sh with the declared values
bench() {
  e=$1 c=$2; shift 2
  TMPDIR="$t/tmpdir" sh "$root/bench.sh" --endpoint "$e" --model test-model --tp 1 --pp 1 \
    --quant Q4_K_M --kv f16 --spec none --thinking default --submitter tester --power-limit 200 \
    --results-dir "$t/res" ${c:+--server-cmd-file "$c"} "$@" > "$t/out" 2>&1
}
note() { sed -n "s/^$1=//p" "$t/notes"; }
# dry-run notes: the --note values the dry run would pass
dry_notes() { bench "$@" --dry-run; python3 - "$t/out" > "$t/notes" <<'PY'
import shlex, sys
line = next(l for l in open(sys.argv[1]) if l.startswith("  betterbench"))
w = shlex.split(line)
print("\n".join(w[i + 1] for i, x in enumerate(w) if x == "--note"))
PY
}

# ------------------------------------------------------------------ build
check "bench.sh is built from dev/" 'sh "$dev/build.sh" --check >/dev/null 2>&1'
check "shell syntax" 'for f in "$root/bench.sh" "$dev"/*.sh; do sh -n "$f" || exit 1; done'

# ------------------------------------------------------------------ argument checks
bench "$LL" "" --spec bogus; check "rejects a bad --spec" 'grep -q "spec must be" "$t/out"'
TMPDIR="$t/tmpdir" sh "$root/bench.sh" --endpoint "$LL" > "$t/out" 2>&1
check "lists missing mandatory values" 'grep -q "missing mandatory value(s): --model --tp" "$t/out"'
bench "$LL" "" --temperature 0; check "refuses BetterBench flags" 'grep -q "not allowed" "$t/out"'

# ------------------------------------------------------------------ a quick run
printf 'llama-server -m %s -ngl 99 -fa on -b 4096 -t -1 -e HF_TOKEN=hf_secret\n' "$t/models/test.gguf" > "$t/cmd1"
bench "$LL" "$t/cmd1"; rc=$?
run=$(sed -n 's/^done: //p' "$t/out")
check "quick run finishes (no GPU, the sampler exits at once)" '[ $rc = 0 ] && [ -d "$run" ]'
check "run folder has every file" 'for f in results.json results.html meta.json config.json server-cmd.txt summary.txt betterbench.log; do [ -s "$run/$f" ] || exit 1; done'
check "no empty power.csv without a GPU" '[ ! -e "$run/power.csv" ]'
check "summary printed" 'grep -q "decode (batch 1)  52.0 t/s" "$t/out"'
check "meta.json has post-run figures and warnings" 'python3 -c "import json,sys; m=json.load(open(sys.argv[1])); assert \"post_run\" in m and \"warnings\" in m" "$run/meta.json"'
check "secrets redacted" '! grep -q hf_secret "$run/server-cmd.txt" && grep -q "HF_TOKEN=<redacted>" "$run/server-cmd.txt"'
check "results in --results-dir, helpers cleaned up" '[ -z "$(ls -A "$t/tmpdir")" ] && [ ! -e "$t/home/results" ]'
check "model hashed, flagged as non-standard" 'grep -q "not one of the standard models" "$t/out"'
check "model hash is cached" 'bench "$LL" "$t/cmd1" --dry-run; ! grep -q "collect-env: hashing" "$t/out"'

# ------------------------------------------------------------------ launch flags and cross-checks
dry_notes "$LL" "$t/cmd1"
check "launch note parsed (negative values stay values)" '[ "$(note launch)" = "b 4096 · ub default · fa on · ctk default · ctv default · ngl 99 · t -1" ]'
check "model_file note has size and hash" 'note model_file | grep -q "0.00 GB · sha256 [0-9a-f]\{12\} (not a standard model)"'

printf 'llama-server -m /m.gguf -ctk q8_0\n' > "$t/cmd2"
bench "$LL" "$t/cmd2" --kv q8_0 --dry-run
check "--kv: K set, V default -> q8_0/f16" 'grep -q "declare --kv q8_0/f16" "$t/out"'
bench "$LL" "$t/cmd2" --kv q8_0/f16 --dry-run
check "--kv q8_0/f16 accepted" '! grep -q "warning: --kv" "$t/out"'

printf 'llama-server -m /m.gguf -md /d.gguf --draft-max 8 --reasoning-budget 0\n' > "$t/cmd3"
bench "$LL" "$t/cmd3" --dry-run
check "--spec checked against a draft model" 'grep -q "declare --spec draft:8" "$t/out"'
check "--thinking checked against --reasoning-budget 0" 'grep -q "declare --thinking off" "$t/out"'
check "dry run recaps warnings" 'grep -q "warning(s) - fix or explain" "$t/out"'

cat > "$t/cmd4" <<'EOF'
VLLM_USE_V1=1 vllm serve Org/Repo -tp 2 --max-num-batched-tokens=8192 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' --api-key sk-secret
EOF
bench "$VL" "$t/cmd4" --dry-run
check "vllm: --spec from --speculative-config" 'grep -q "declare --spec mtp:5" "$t/out"'
check "vllm: --tp from -tp" 'grep -q "declare --tp 2" "$t/out"'
dry_notes "$VL" "$t/cmd4" --spec mtp:5 --tp 2
check "vllm: matching values give no warning" '! grep -q "warning: --" "$t/out"'
check "vllm: --max-num-batched-tokens=N not redacted" 'note launch | grep -q "max-num-batched-tokens 8192"'
check "vllm: env flags recorded" '[ "$(note env_flags)" = "VLLM_USE_V1=1" ]'

bench "$LR" "" --thinking off --dry-run
check "--thinking off vs reasoning in the replies" 'grep -q "thinking off, but the greedy replies contain reasoning" "$t/out"'

# ------------------------------------------------------------------ finding the weights
# docker without llama-swap: the container path maps back through -v
printf 'docker run --rm -v %s:/models img -m /models/test.gguf\n' "$t/models" > "$t/cmd5"
dry_notes "$LR" "$t/cmd5"
check "docker -v mapping without llama-swap" 'note model_file | grep -q "^/models/test.gguf · 0.00 GB · sha256"'

# vLLM repo id -> HF cache snapshot, safetensors as symlinks into blobs/
hf="$t/hf/hub/models--Org--Repo"; mkdir -p "$hf/blobs" "$hf/refs" "$hf/snapshots/abc123"
printf 'A%.0s' $(seq 1000) > "$hf/blobs/b1"; printf 'B%.0s' $(seq 2000) > "$hf/blobs/b2"
ln -s ../../blobs/b1 "$hf/snapshots/abc123/model-00001-of-00002.safetensors"
ln -s ../../blobs/b2 "$hf/snapshots/abc123/model-00002-of-00002.safetensors"
printf 'abc123' > "$hf/refs/main"
HF_HOME="$t/hf" sh "$root/bench.sh" collect-env --endpoint "$VL" --model test-model > "$t/env" 2>/dev/null
want=$(cd "$hf/snapshots/abc123" && sha256sum *.safetensors | LC_ALL=C sort -k2 | sha256sum | cut -c1-64)
check "HF repo id resolved, symlink sizes followed" 'grep -qx "model_bytes=3000" "$t/env"'
check "repo hash equals the README recipe" 'grep -qx "model_sha256=$want" "$t/env"'

# ------------------------------------------------------------------ failures and signals
FAKE_BB_FAIL=3 bench "$LL" "$t/cmd1"; rc=$?
check "betterbench failure -> .failed folder, exit 1" '[ $rc = 1 ] && grep -q "betterbench failed (exit 3)" "$t/out" && ls -d "$t"/res/*.failed >/dev/null 2>&1'

FAKE_BB_SLEEP=60 TMPDIR="$t/tmpdir" sh "$root/bench.sh" --endpoint "$LL" --model test-model --tp 1 --pp 1 \
  --quant Q4_K_M --kv f16 --spec none --thinking default --submitter tester --power-limit 200 \
  --results-dir "$t/res2" > "$t/out" 2>&1 &
p=$!
i=0; until grep -q "running betterbench" "$t/out" || [ $i -gt 100 ]; do sleep 0.1; i=$((i + 1)); done
sleep 0.5; s0=$(date +%s); kill -TERM "$p"; wait "$p"; rc=$?; s1=$(date +%s)
check "TERM stops a run at once, exit 143" '[ $rc = 143 ] && [ $((s1 - s0)) -lt 5 ]'
check "interrupted run folder marked, no done:" 'ls -d "$t"/res2/*.interrupted >/dev/null 2>&1 && ! grep -q "^done:" "$t/out"'
check "no betterbench left running" '! pgrep -f "FAKE_BB_SLEEP|dev/test/betterbench run" >/dev/null 2>&1'
check "helpers cleaned up after a signal" '[ -z "$(ls -A "$t/tmpdir")" ]'

# ------------------------------------------------------------------ the settings file
# plain bench.sh (no declared values on the command line) with BENCH_HOME=$t/home
plain() { TMPDIR="$t/tmpdir" sh "$root/bench.sh" "$@" > "$t/out" 2>&1; }
plain init; rc=$?
check "init writes bench.local.env next to bench.sh" '[ $rc = 0 ] && grep -q "^BENCH_SUBMITTER=" "$t/home/bench.local.env"'
plain init; rc=$?
check "init doesn't overwrite" '[ $rc = 1 ] && grep -q "exists" "$t/out"'
plain --dry-run
check "an unfilled file names what's missing and how to save it" 'grep -q "missing mandatory value(s): --model --quant --submitter" "$t/out" && grep -q "bench.sh init" "$t/out"'
s="$t/home/bench.local.env"
sed -e "s|^BENCH_SUBMITTER=.*|BENCH_SUBMITTER=filer   # me|" \
    -e "s|^BENCH_ENDPOINT=.*|BENCH_ENDPOINT=$LL|" -e "s|^BENCH_MODEL=.*|BENCH_MODEL=test-model|" \
    -e "s|^BENCH_SERVER_CMD_FILE=.*|BENCH_SERVER_CMD_FILE=cmd.txt|" -e "s|^BENCH_QUANT=.*|BENCH_QUANT=Q4_K_M|" \
    -e 's|^# BENCH_POWER_LIMIT=250 .*|export BENCH_POWER_LIMIT=200|' \
    -e 's|^# BENCH_SPECIAL_NOTES=.*|BENCH_SPECIAL_NOTES="two words # not a comment"  # a comment|' \
    -e 's|^# BENCH_RESULTS_DIR=.*|BENCH_RESULTS_DIR=res|' "$s" > "$s.new" && mv "$s.new" "$s"
printf 'llama-server -m /m.gguf -ngl 99\n' > "$t/home/cmd.txt"
(cd / && TMPDIR="$t/tmpdir" sh "$root/bench.sh" --dry-run > "$t/out" 2>&1); rc=$?
python3 - "$t/out" > "$t/notes" <<'PY'
import shlex, sys
line = next((l for l in open(sys.argv[1]) if l.startswith("  betterbench")), "")
w = shlex.split(line)
print("\n".join(w[i + 1] for i, x in enumerate(w) if x == "--note"))
PY
check "a filled file runs with no flags at all" '[ $rc = 0 ] && grep -q "settings from .*bench.local.env: BENCH_SUBMITTER" "$t/out"'
check "file: comments stripped, quotes kept, export accepted" '[ "$(note submitter)" = filer ] && [ "$(note special_notes)" = "two words # not a comment" ] && note power_limit | grep -q "^200 W"'
check "file: relative paths are relative to the file" 'note launch | grep -q "ngl 99" && grep -q "would write $t/home/res/" "$t/out"'
plain --dry-run --kv q8_0
check "a flag wins over the file" 'grep -q "weights Q4_K_M · kv q8_0" "$t/out"'
BENCH_SUBMITTER=envname plain --dry-run
check "the environment wins over the file" 'grep -q "submitter=envname" "$t/out"'
sed 's|^BENCH_SUBMITTER=.*|BENCH_SUBMITTER=other|' "$s" > "$t/home/other.env"
plain --settings "$t/home/other.env" --dry-run
check "--settings picks another file" 'grep -q "submitter=other" "$t/out"'
printf 'BETTERBENCH_API_KEY=sk-x\n' > "$t/key.env"; plain --settings "$t/key.env" --dry-run; rc=$?
check "an API key in the file is refused" '[ $rc = 1 ] && grep -q "keep the API key out of files" "$t/out"'
printf 'BENCH_KV=$(touch %s/pwned)\n' "$t" > "$t/code.env"; plain --settings "$t/code.env" --dry-run
check "the file is data, never run" '[ ! -e "$t/pwned" ]'
rm -f "$s"

# ------------------------------------------------------------------ power sampling
f="$t/sys/bus/pci/drivers/amdgpu/0000:03:00.0/hwmon/hwmon1"; mkdir -p "$f"
echo 150000000 > "$f/power1_input"; echo 2100000000 > "$f/freq1_input"
GPU_POWER_SYSFS="$t/sys" timeout 2 sh "$dev/gpu-power.sh" 1 > "$t/power" 2>/dev/null
check "amdgpu power via sysfs" 'grep -q "^[0-9.]*,150.00,,2100$" "$t/power"'

echo
echo "$pass passed, $fail failed"
[ $fail = 0 ]
