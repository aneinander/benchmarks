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
   (GPU, driver, CPU, RAM, OS, engine and version, model file, context) as notes.

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
- `--pcie "Gen4 x16"` - declare the slot link where it can't be read (VMs).
- `--special-notes "..."` - free text for anything unusual about the run (custom
  kernels, undervolt, patched engine). Recorded as `special_notes`.
- `--api-key KEY` or `$BETTERBENCH_API_KEY` - never stored.
- Each mandatory flag can also come from `BENCH_<NAME>` (e.g. `BENCH_SUBMITTER`).
- Results go to `results/` next to this file, or `--results-dir` / `$BENCH_RESULTS_DIR`.
  Machine-local defaults can go in `bench.local.env` (sourced, not committed).

`bench.sh` first sends one warm-up request (through llama-swap the first request loads
the model, which takes minutes), then runs `collect-env.sh`, then BetterBench.

**BetterBench on a different machine than the server:** run `./collect-env.sh --endpoint
http://localhost:PORT/v1 --model ID > env.txt` on the server and pass `--env-file env.txt`
to `bench.sh` on the client.

Each run produces `results/<timestamp>-<submitter>-<model>-<engine>-<gpu>/` with
`results.json`, `results.html`, `meta.json` and `server-cmd.txt`. See [`SUBMIT.md`](SUBMIT.md).

## Recorded notes

The notes are grouped, one per topic, so the report and compare page stay readable:

| note | example |
|---|---|
| `submitter` | `pascal` |
| `gpu` | `1x Intel Arc Pro B65 · 32 GB · driver 1.17.39395` (VRAM per GPU) |
| `pcie` | `Gen4 x16`, `Gen1 x16 now, Gen4 x16 max` (idle downclock), `Gen4 x16 (declared)` |
| `power_limit` | `200 W`, or `180 W (GPU reports 200 W)` when they disagree |
| `host` | `AMD EPYC 7282 16-Core Processor · 60 GB RAM · Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic · VM kvm` |
| `engine` | `vllm 0.30.0 · via llama-swap`, `llama.cpp b11913-7aca0d2ea · 4 slots · via llama-swap` |
| `model_file` | `/models/Qwen3.8-27B-Q4_0.gguf` |
| `parallel` | `tp 1 · pp 1` |
| `precision` | `weights Q4_0 · kv f16` |
| `spec`, `thinking` | `mtp:5`, `default` |
| `run` | `full · concurrency on · bench-standard 0.2.0` |
| `special_notes` | only when given: free text from `--special-notes` |

The context size shows as BetterBench's own context chip. Parts that can't be detected
are left out. `meta.json` keeps every raw value (`mandatory`, `collected` from
`collect-env.sh`) for scripts.

PCIe is the link into the slot. Cards with their own PCIe switch, like Intel Arc, are
read at the switch's upstream port. Inside a VM the link can't be read: run
`collect-env.sh` on the hypervisor or pass `--pcie "Gen4 x16"`.

Engine detection: llama.cpp via `/props`, vLLM via `/version` + `/v1/models`; behind
llama-swap through `/upstream/<model>/...`, which also yields the launch command.
GPU detection: `nvidia-smi`, then `rocm-smi`, then `xpu-smi`.

## Comparing

```sh
betterbench compare A/results.json B/results.json
```

Every note that differs between two runs shows up as a mismatch chip. Compare like
with like.
