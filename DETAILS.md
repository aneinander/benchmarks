# Details

What `bench.sh` records, and how. For running it, see [`README.md`](README.md).

## What a run does

1. One warm-up request. Behind llama-swap the first request loads the model, which
   can take minutes.
2. **Greedy check:** the same prompt 3 times at temperature 0.
3. `collect-env.sh` detects the hardware and the engine, and measures memory bandwidth.
4. BetterBench runs while `gpu-power.sh` samples power, energy and clock every 2 s.
5. A summary prints. Energy and % of bandwidth are written to `meta.json`.

## Files in a run folder

| file | what |
|---|---|
| `results.json` | BetterBench results; the notes below are under `env.notes` |
| `results.html` | the standalone report |
| `meta.json` | every raw value: `mandatory` (declared), `collected` (detected), `measured`, `launch`, `post_run` |
| `server-cmd.txt` | the server launch command, secrets redacted |
| `power.csv` | `epoch_s,watts,joules,mhz` during the run |
| `betterbench.log` | BetterBench's full output |

## Notes

One note per topic, so the report and the compare page stay readable. Anything that
can't be detected is left out or shows `unknown`.

| note | example |
|---|---|
| `submitter` | `pascal` |
| `gpu` | `1x Intel Arc Pro B65 · 32 GB · driver 1.17.39395` (VRAM per GPU) |
| `pcie` | `Gen4 x16 · ReBAR on`, `Gen1 x16 now, Gen4 x16 max · ReBAR on`, `unknown (VM) · ReBAR on` |
| `power_limit` | `200 W`, or `180 W (GPU reports 200 W)` |
| `host` | `AMD EPYC 7282 16-Core Processor · 60 GB RAM · Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic · VM kvm` |
| `engine` | `llama.cpp b11913-7aca0d2ea · 4 slots · image ghcr.io/ggml-org/llama.cpp:server-intel@1a2b3c4d5e6f · via llama-swap` |
| `model_file` | `/models/Qwen3.8-27B-Q4_0.gguf · 15.60 GB` (weights as stored) |
| `mem_bw` | `532 GB/s measured (torch copy, random, 1024 MiB) · 608 GB/s declared` |
| `parallel` | `tp 1 · pp 1` |
| `precision` | `weights Q4_0 · kv f16` |
| `launch` | `b 4096 · ub 1024 · fa on · ctk q8_0 · ctv q8_0 · ngl 99 · np 4` |
| `env_flags` | `GGML_SYCL_DISABLE_OPT=0 · ZES_ENABLE_SYSMAN=1`, or `none` |
| `spec`, `thinking` | `mtp:5`, `default` |
| `greedy_check` | `identical 3/3 · 1d5105fe09dc`, `differs: 2 variants in 3 · first ...` |
| `run` | `full · concurrency on · bench-standard 0.3.0` |
| `special_notes` | only when given |

The context size appears as BetterBench's own chip.

## How things are detected

**GPU:** `nvidia-smi`, then `rocm-smi`, then `xpu-smi`.

**PCIe** is the link into the slot. Cards with their own PCIe switch (Intel Arc, some AMD)
are read at the switch's upstream port. Inside a VM the link can't be read: run
`collect-env.sh` on the hypervisor, or pass `--pcie`.

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

**Memory bandwidth:** `mem-bw.py` copies 1 GiB of random data with torch, inside the
server's container (same GPU, env and torch), or with `$BW_PYTHON`. It uses random
data because Arc compresses uniform buffers: zeros read 1627 GB/s on a B65 that really
does 532. The fallback is `clpeak`, which may read high on Arc. `MEASURE_BW=0` skips the
test. `--mem-bw` adds the spec figure next to the measured one.

**Greedy check:** temperature 0, seed 0, 128 tokens, 3 times. `identical` is expected.
`differs` points to nondeterministic kernels or a broken spec-decoding path. The hash
lets you compare setups: the same model and quant should give the same hash. A
different hash isn't automatically wrong, since kernels round differently.

## After the run

These go to `meta.json` under `post_run` and the summary, not to the notes, which are
fixed before BetterBench starts.

- **energy:** kJ, average and peak W, J per output token, and average, min and max clock.
  The clock counts only samples where the GPU is running. The figures cover the
  **whole run**, warm-up and prefill included, so compare only runs of the same mode.
  Power is sampled only when BetterBench runs on the server. On a split setup, run
  `gpu-power.sh > power.csv` on the server yourself.
- **roofline:** median batch-1 decode t/s × weight bytes ÷ memory bandwidth = % of
  bandwidth. It holds for dense models. Spec decoding and MoE can exceed 100%. So
  can models quantized at load (e.g. vLLM fp8 from bf16 files), because they read
  fewer bytes than are stored. For safetensors the size includes every file, so the
  RedHatAI models' MTP head adds about 5%.

## BetterBench with PR #4

[GGZ14/BetterBench#4](https://github.com/GGZ14/BetterBench/pull/4) adds BetterBench's
own GPU/PCIe/ReBAR probe under `env.gpu` (still version 0.6.1). That probe describes the
machine BetterBench runs on, so the notes above are what count. `bench.sh` warns when
the two disagree. The probe adds an `intel`-style chip, so `compare` shows a `gpu`
mismatch between runs with and without the PR.
