# Details

What `bench.sh` records, how, and how it's built. For running it, see
[`AGENTS.md`](../AGENTS.md) or `./bench.sh --help`.

## What a run does

1. One warm-up request. Behind llama-swap the first request loads the model, which
   can take minutes.
2. **Greedy check:** the same prompt 3 times at temperature 0.
3. `collect-env.sh` (also `bench.sh collect-env`) detects the hardware and the engine, and measures memory bandwidth.
4. BetterBench runs while `gpu-power.sh` (also `bench.sh power`) samples power, energy and the core and memory clocks every 2 s.
5. A summary prints. Energy and % of bandwidth are written to `meta.json`.

## Files in a run folder

`<run>` is the folder's name, `<YYMMDD>_<submitter>_<model>_<engine>_<N>x-<gpu>[_quick][_2]`
with `N` = tp × pp; see the README.

| file | what |
|---|---|
| `<run>.json` | BetterBench results; the notes below are under `env.notes` |
| `<run>.html` | the standalone report |
| `config.json` | the BetterBench config the run used |
| `meta.json` | every raw value: `mandatory` (declared), `collected` (detected), `measured`, `launch`, `post_run` |
| `server-cmd.txt` | the server launch command, secrets redacted |
| `power.csv` | `epoch_s,watts,joules,mhz,mem_mhz` during the run |
| `timeline.tsv` | when each BetterBench progress line appeared, for the energy per phase |
| `betterbench.log` | BetterBench's full output |

## Notes

One note per topic, so the report and the compare page stay readable. Anything that
can't be detected is left out or shows `unknown`.

| note | example |
|---|---|
| `submitter` | `pascal` |
| `gpu` | `1x Intel Arc Pro B65 · 32 GB · driver 1.17.39395 · max 2800 MHz core · 19000 MHz mem` (VRAM per GPU, max clocks of the first) |
| `pcie` | `Gen4 x16 · ReBAR on`, `Gen1 x16 now, Gen4 x16 max · ReBAR on`, `unknown (VM) · ReBAR on` |
| `power_limit` | `200 W`, or `180 W (GPU reports 200 W)` |
| `host` | `AMD EPYC 7282 16-Core Processor · 60 GB RAM · Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic · VM kvm` |
| `engine` | `llama.cpp b11913-7aca0d2ea · 4 slots · image ghcr.io/ggml-org/llama.cpp:server-intel@1a2b3c4d5e6f · via llama-swap` |
| `model_file` | `/models/Qwen3.8-27B-UD-Q4_K_M.gguf · 16.46 GB · standard: Qwen3.8-27B UD-Q4_K_M`, `... · not a standard model (sha256 4f58a79550e8)` |
| `mem_bw` | `532 GB/s measured (torch copy, random, 1024 MiB) · 608 GB/s declared` |
| `parallel` | `tp 1 · pp 1` |
| `precision` | `weights Q4_0 · kv f16` |
| `max_ctx` | `412k tokens (engine KV cache) · 262k set` (vLLM), `≈ 181k tokens est. · 32k set · 9.1 GB free · 64 KiB/token` (llama.cpp) |
| `launch` | `b 4096 · ub 1024 · fa on · ctk q8_0 · ctv q8_0 · ngl 99 · np 4` |
| `env_flags` | `GGML_SYCL_DISABLE_OPT=0 · ZES_ENABLE_SYSMAN=1`, or `none` |
| `spec` | `mtp:5`, `none` |
| `thinking` | `default · reasoning seen`, `off · no reasoning seen` (declared · observed) |
| `greedy_check` | `identical 3/3 · 1d5105fe09dc`, `differs: 2 variants in 3 · first ...` |
| `run` | `full · concurrency on · bench-standard 0.6.0 · betterbench 0.6.1`, or `... · betterbench 0.6.1 @ v0.6.1-3-gabc1234-dirty` for a build that isn't the release |
| `special_notes` | only when given |

The context size appears as BetterBench's own chip.

## How things are detected

**GPU:** `nvidia-smi`, then `rocm-smi`, then `xpu-smi`.

**Clocks:** the max core and memory clocks come from `nvidia-smi` (`clocks.max.*`), the
driver's sysfs (xe/i915 `RP0`, amdgpu `pp_dpm_sclk`/`pp_dpm_mclk`) or `xpu-smi discovery`.
The run samples the current ones. `xpu-smi` versions without a memory clock metric leave
`mem_mhz` empty, and the rest of the sampling still works.

**Max context** is the longest context this configuration holds in memory. vLLM logs its
KV cache size in tokens ("GPU KV cache size: N tokens"), shared by all requests running at
once. llama.cpp allocates the cache for `-c` when it loads, so this is an estimate: `-c`
plus the VRAM still free (read after the warm-up, over all GPUs) over the KV bytes per
token. Those come from the GGUF header and `-ctk`/`-ctv`: only layers whose cache grows
with the context count. Recurrent layers (Mamba, linear attention) don't, and neither do
sliding-window layers unless `--swa-full` is set. Compute buffers grow a little with the
context too, so treat it as an upper bound. The note is left out when the model file
can't be read on this machine.

**PCIe** is the link into the slot. Cards with their own PCIe switch (Intel Arc, some AMD)
are read at the switch's upstream port. Inside a VM the link can't be read: run
`bench.sh collect-env` on the hypervisor, or pass `--pcie`.

**ReBAR** is on when the GPU's largest BAR is bigger than 256 MiB (about the size of VRAM).
This works in a VM too, because passthrough keeps BAR sizes.

**Engine:** llama.cpp via `/props` (its build includes the commit), vLLM via `/version` and
`/v1/models`. Behind llama-swap these go through `/upstream/<model>/`, which also gives the
launch command. For a docker server the image is recorded with its ID, because tags like
`server-intel` move daily.

**Launch settings** come from the launch command, because no API reports them:
- llama.cpp: `b`, `ub`, `fa`, `ctk` and `ctv` always show, as `default` when not set,
  because the defaults change between builds.
- vLLM: batching, KV type, compilation and spec config.
- **Env flags** come from `VAR=x cmd`, `env VAR=x` and docker `-e VAR=x`. Settings in
  llama-swap's `env:` block, or in the shell that started the server, aren't visible;
  put them in the command.
- If `--kv` doesn't match the command, `bench.sh` warns.

**Model file:** the path comes from the launch command. `model_hash.py` hashes it the
way the README's Models section does (a GGUF file, or the `*.safetensors` of a
directory) and looks the hash up in `models.tsv`. Not a standard model, or a `--quant`
that differs from the table, gives a warning. Per-file hashes are cached in
`~/.cache/bench-standard/sha256.tsv` by path, size and mtime, so only the first run
waits (about a minute for 16 GB). `HASH_MODEL=0` skips it, and the note then reads
`hash not checked`. When the server runs in docker, the path is the one inside the
container. If that path doesn't exist on the host, the hash can't be checked either.

**BetterBench build:** read from pip's record of where it was installed from: the git
commit for `pip install git+...`, or `git describe` of the checkout for an editable
install. A release tag adds nothing to the `run` note; a commit or `-dirty` does.

**Memory bandwidth:** `mem-bw.py` copies 1 GiB of random data with torch, inside the
server's container (same GPU, env and torch), or with `$BW_PYTHON`. It uses random
data because Arc compresses uniform buffers: zeros read 1627 GB/s on a B65 that really
does 532. The fallback is `clpeak`, which may read high on Arc. `MEASURE_BW=0` skips the
test. `--mem-bw` adds the spec figure next to the measured one.

**Greedy check:** temperature 0, seed 0, 128 tokens, 3 times. `identical` is expected.
`differs` points to nondeterministic kernels or a broken spec-decoding path. The hash
lets you compare setups: the same model and quant should give the same hash. A
different hash isn't automatically wrong, since kernels round differently. The check
also records whether the replies held reasoning: a separate reasoning field, or think
tags in the text. `bench.sh` warns when `--thinking off` meets reasoning, or when
`on`/`low`/`medium`/`high` meets none. With `default` the observation is only recorded.

## After the run

These go to `meta.json` under `post_run` and the summary, not to the notes, which are
fixed before BetterBench starts.

- **energy:** kJ, average and peak W, J per output token over the whole run, and
  average, min and max core and memory clock. The clock counts only samples where the GPU is running.
  Power is sampled only when BetterBench runs on the server. On a split setup, run
  `bench.sh power > power.csv` on the server yourself.
- **energy per phase** (`energy.phases`): each BetterBench progress line is stamped in
  `timeline.tsv`, and a piece of work lasts until the next line. That gives joules and
  average W for decode (J per output token), prefill (J per 1000 prompt tokens) and each
  concurrency level (J per token). The warm-up passes draw power but aren't in
  the results, so their tokens are estimated from the measured passes of the same
  piece (`tokens_incl_warmup_est`). These compare across quick and full runs; the
  whole-run figure doesn't.
- **roofline:** median batch-1 decode t/s × weight bytes ÷ memory bandwidth = % of
  bandwidth. The weight bytes are what vLLM logs as loaded ("Model loading took X GiB",
  per GPU, × tp), which is right for models quantized at load (fp8 from bf16 files reads
  half of what's stored). Otherwise the stored size, without `*mtp*` files, since plain
  decoding doesn't read an MTP head. With tp > 1 the bandwidth is per GPU × tp. It holds
  for dense models; spec decoding and MoE can exceed 100%.

## BetterBench with PR #4

[GGZ14/BetterBench#4](https://github.com/GGZ14/BetterBench/pull/4) adds BetterBench's
own GPU/PCIe/ReBAR probe under `env.gpu` (still version 0.6.1). That probe describes the
machine BetterBench runs on, so the notes above are what count. `bench.sh` warns when
the two disagree. The probe adds an `intel`-style chip, so `compare` shows a `gpu`
mismatch between runs with and without the PR.

## Building

People download only the root `bench.sh`. It's generated: `dev/build.sh` packs
`dev/bench.sh`, `collect-env.sh`, `gpu-power.sh`, `gguf_kv.py`, `mem-bw.py`, `model_hash.py`,
`notes.py`, `postrun.py`, `models.tsv`, `config.json` and `standard.env` into it. At run time it unpacks them to a temp dir and runs
`dev/bench.sh` from there, with `BENCH_HOME` set to its own folder, so `results/`
and `bench.local.env` stay next to the downloaded file.

```sh
sh dev/build.sh            # after any change in dev/; commit dev/ and bench.sh together
sh dev/build.sh --check    # fails if bench.sh is out of date
sh dev/bench.sh --help     # dev/bench.sh also runs directly, results in dev/results/
pytest dev/tests           # the notes, model hash and post-run figures
```

CI (`.github/workflows/ci.yml`) runs shellcheck, `build.sh --check` and the tests.
A model added to the README's table goes into `dev/models.tsv` too.

BetterBench 0.6.1 is the oldest version that works (`BETTERBENCH_MIN_VERSION` in
`standard.env`). Newer ones run too, and their version goes into the `run` note.
Bump `STANDARD_VERSION` when `config.json`, the minimum BetterBench or the recorded
fields change.
