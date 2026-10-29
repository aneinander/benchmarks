# bench-standard

A thin wrapper around [BetterBench](https://github.com/GGZ14/BetterBench) so everyone in the group
benchmarks the same way and every result records the setup it was measured on.
BetterBench itself is used unmodified, only through its CLI.

## The rules

1. **BetterBench 0.6.1**, exactly (tag `v0.6.1`). `bench.sh` refuses any other version.
   The pin lives in `standard.env`.
2. **Models:** the ~9B model and Qwen3.8-27B, exact checkpoints in [`models.md`](models.md).
   Verify the SHA256 before you benchmark.
3. **Config:** `config.json`, identical to BetterBench's `config/default.json` (no
   deviations). max_tokens comes from the corpus, so it matches as long as the corpus does.
4. **Server settings you have to fix and declare** (BetterBench cannot set or detect them):

   | flag | meaning | example |
   |---|---|---|
   | `--tp` / `--pp` | tensor / pipeline parallel size | `1` / `1` |
   | `--quant` | weight quantization | `Q4_K_M`, `gptq-int4` |
   | `--kv` | KV cache dtype | `f16`, `q8_0`, `int8_k_int4_v` |
   | `--spec` | speculative decoding: `none` or `METHOD:N` (`mtp`, `eagle`, `eagle3`, `ngram`, `draft`) | `mtp:5` |
   | `--thinking` | thinking mode **configured on the server**: `off`, `on`, `low`, `medium`, `high`, `default` | `default` |
   | `--submitter` | who ran it | `pascal` |

   The GPU power limit is read from the GPU. Pass `--power-limit W` (or `stock`) only
   where it can't be detected, or to declare it; a mismatch is noted in the result.

   BetterBench can't send `chat_template_kwargs` / `reasoning_effort`, so the thinking
   mode must be fixed in the server's chat template or launch flags and declared here.
5. **Results come from `bench.sh` only.** It adds everything it can detect itself
   (GPU, driver, PCIe/ReBAR, memory bandwidth, CPU, RAM, OS, engine and version,
   model file and size, launch flags and env flags, context) as notes.
6. **Memory bandwidth is measured** with `mem-bw.py`, a torch copy of random data run
   inside the server's container (vLLM images have torch). Elsewhere set `BW_PYTHON`
   to a python with GPU torch, or install `clpeak` as a fallback. Pass the card's
   spec figure as `--mem-bw GBPS`; both are recorded.

## Running

On the inference server (or any machine that reaches it and has BetterBench 0.6.1):

```sh
./bench.sh --endpoint http://localhost:8080/v1 --model vllm/lfm3-1.7b \
  --tp 1 --pp 1 --quant bf16 --kv bf16 --spec none --thinking default \
  --submitter pascal --server-cmd-file my-server-cmd.txt
```

That is a quick smoke run (BetterBench `--quick`, a few minutes) to check the setup.
Add `--full` for the run that counts; it takes much longer and occupies the GPU.

- `--full` - the complete, publishable run; recorded as `run=full · ...`.
- `--no-concurrency` - skip the concurrency sweep; recorded as `concurrency=off`.
- `--dry-run` - print the BetterBench command and exit.
- `--verbose` - stream BetterBench's full output and list the notes. By default you
  see the progress lines and a summary at the end. The full output is in
  `betterbench.log` either way, and what gets recorded is identical.
- `--pcie "Gen4 x16"` - declare the slot link where it can't be read (VMs).
- `--special-notes "..."` - free text for anything unusual about the run (custom
  kernels, undervolt, patched engine). Recorded as `special_notes`.
- `--mem-bw GBPS` - the card's declared (spec sheet) memory bandwidth.
  `MEASURE_BW=0` skips the measurement (a few seconds, 2 GiB of free VRAM).
- `--server-cmd-file FILE` - the server launch command; needed for the `launch` and
  `env_flags` notes unless the server runs behind llama-swap.
- `--api-key KEY` or `$BETTERBENCH_API_KEY` - never stored.
- Each mandatory flag can also come from `BENCH_<NAME>` (e.g. `BENCH_SUBMITTER`).
- Results go to `results/` next to this file, or `--results-dir` / `$BENCH_RESULTS_DIR`.
  Machine-local defaults can go in `bench.local.env` (sourced, not committed).

`bench.sh` first sends one warm-up request (through llama-swap the first request loads
the model, which takes minutes), then the greedy check, then `collect-env.sh`, then
BetterBench while `gpu-power.sh` samples GPU power and clock.

**BetterBench on a different machine than the server:** run `./collect-env.sh --endpoint
http://localhost:PORT/v1 --model ID > env.txt` on the server and pass `--env-file env.txt`
to `bench.sh` on the client.

Each run produces `results/<timestamp>-<submitter>-<model>-<engine>-<gpu>/` with
`results.json`, `results.html`, `meta.json`, `server-cmd.txt`, `power.csv` and
`betterbench.log`. See
[`SUBMIT.md`](SUBMIT.md).

## Recorded notes

The notes are grouped, one per topic, so the report and compare page stay readable:

| note | example |
|---|---|
| `submitter` | `pascal` |
| `gpu` | `1x Intel Arc Pro B65 · 32 GB · driver 1.17.39395` (VRAM per GPU) |
| `pcie` | `Gen4 x16 · ReBAR on`, `Gen1 x16 now, Gen4 x16 max · ReBAR on` (idle downclock), `Gen4 x16 (declared) · ReBAR off` |
| `power_limit` | `200 W`, or `180 W (GPU reports 200 W)` when they disagree |
| `host` | `AMD EPYC 7282 16-Core Processor · 60 GB RAM · Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic · VM kvm` |
| `engine` | `vllm 0.32.0 · image lfm3-vllm-xpu:v0.32.0-lfm3-fp8fix@ad5083ed33a7 · via llama-swap`, `llama.cpp b11913-7aca0d2ea · 4 slots · image ghcr.io/ggml-org/llama.cpp:server-intel@1a2b3c4d5e6f` |
| `model_file` | `/models/Qwen3.8-27B-Q4_0.gguf · 15.60 GB` (weights as stored) |
| `mem_bw` | `532 GB/s measured (torch copy, random, 1024 MiB) · 608 GB/s declared` |
| `parallel` | `tp 1 · pp 1` |
| `precision` | `weights Q4_0 · kv f16` |
| `launch` | llama.cpp: `b 4096 · ub 1024 · fa on · ctk q8_0 · ctv q8_0 · ngl 99 · np 4`; vLLM: `max-num-batched-tokens 4224 · max-num-seqs 4 · kv-cache-dtype int8_k_int4_v · ...` |
| `env_flags` | `GGML_SYCL_DISABLE_OPT=0 · ZES_ENABLE_SYSMAN=1`, `none` |
| `spec`, `thinking` | `mtp:5`, `default` |
| `greedy_check` | `identical 3/3 · 1d5105fe09dc`, `differs: 2 variants in 3 · first ...` |
| `run` | `full · concurrency on · bench-standard 0.3.0` |
| `special_notes` | only when given: free text from `--special-notes` |

The context size shows as BetterBench's own context chip. Parts that can't be detected
are left out. `meta.json` keeps every raw value (`mandatory`, `collected` from
`collect-env.sh`) for scripts.

PCIe is the link into the slot. Cards with their own PCIe switch, like Intel Arc, are
read at the switch's upstream port. Inside a VM the link can't be read: run
`collect-env.sh` on the hypervisor or pass `--pcie "Gen4 x16"`.

ReBAR comes from the GPU's largest BAR: on when it's bigger than 256 MiB (roughly the
size of VRAM), off otherwise. This works inside a VM too, because passthrough keeps
the BAR sizes. The raw size is kept in `meta.json` as `gpu_bar_mib`.

**BetterBench with [#4](https://github.com/GGZ14/BetterBench/pull/4)** (Intel GPU,
PCIe and ReBAR detection; still reports version 0.6.1) also records its own probe in
`results.json` under `env.gpu` (`pci`, `xpu_smi_discovery`), and the report shows a
vendor chip such as `intel`. That probe describes the machine **BetterBench** runs
on, not necessarily the server, and inside a VM it reads the placeholder link. The
`gpu`/`pcie` notes are what counts. After the run, `bench.sh` checks that probe: it
prints a note if the probe lists a different GPU (expected with `--env-file`), and a
warning if the two disagree on ReBAR. Measurements are unaffected, but the vendor
chip makes a run with #4 show a `gpu` mismatch in `compare` against one without it.

**Launch settings.** Batch/ubatch, flash attention, KV types and env flags can swing
results by 10-50%, and no API reports them, so they are read from the launch command
(llama-swap's, or `--server-cmd-file`). Env flags are `VAR=x` prefixes, `env VAR=x`
and docker `-e VAR=x`. Settings in llama-swap's `env:` block, or set in the shell
that started the server, are invisible: put them in the command or the cmd file.
For llama.cpp, `b`, `ub`, `fa`, `ctk` and `ctv` show `default` when they aren't set,
because the defaults change between builds. Offline switches and cache paths
(`HF_HUB_OFFLINE`, `VLLM_CACHE_ROOT`, ...) are left out. `meta.json` keeps everything
parsed under `launch`. If `--kv` doesn't match the KV type the command sets,
`bench.sh` prints a warning.

**Greedy check.** Before the benchmark, the same prompt runs 3 times at temperature 0
(seed 0, 128 tokens). `identical` is expected. `differs` points to nondeterministic
kernels or batching, or a broken spec-decoding path. The hash makes the output
comparable across runs: the same model and quant should give the same hash on
another engine or card. A different hash isn't automatically wrong (kernels round
differently), but check spec decoding or a new kernel against a run without it.

**After the run** (in `meta.json` under `post_run`, printed at the end; these can't be
notes because the notes are fixed before BetterBench starts):

- `energy` - from `power.csv` (GPU power, energy counter and graphics clock every 2 s, via
  `nvidia-smi` or `xpu-smi`): kJ, average/peak W, and J per output token. This
  covers the **whole run** (warm-up passes and prefill included), so it is a gross
  figure, comparable only between runs of the same mode. Only measured when
  BetterBench runs on the server; on a split setup run `gpu-power.sh > power.csv` on
  the server yourself.
  Also average/min/max clock, counted only over samples where the GPU is running
  (a clock-gated idle GPU reads 0 MHz).
- `roofline` - median batch-1 decode t/s × weight bytes ÷ measured `mem_bw` (else the
  declared one) = % of memory
  bandwidth. Valid for dense models whose stored weights are what the GPU reads.
  Spec decoding and MoE can exceed 100%. A model quantized at load (e.g. vLLM
  `--quantization fp8` from bf16 files) reads fewer bytes than are stored, so its
  figure comes out too high.

**Why random data:** Arc compresses uniform buffers, so a test on zeros reads far too
high. On the B65 here zeros give 1627 GB/s and random data 532 GB/s, against 608
declared. A docker server's image is recorded with its ID, because tags like
llama.cpp's `server-intel` move daily. The llama.cpp build and commit are in the
`engine` note anyway (`b11913-7aca0d2ea`, from `/props`).

Engine detection: llama.cpp via `/props`, vLLM via `/version` + `/v1/models`; behind
llama-swap through `/upstream/<model>/...`, which also yields the launch command.
GPU detection: `nvidia-smi`, then `rocm-smi`, then `xpu-smi`.

## Comparing

```sh
betterbench compare A/results.json B/results.json
```

Every note that differs between two runs shows up as a mismatch chip. Compare like
with like.
