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
   | `--power-limit` | GPU power limit in W, or `stock` | `200` |
   | `--submitter` | who ran it | `pascal` |

   BetterBench can't send `chat_template_kwargs` / `reasoning_effort`, so the thinking
   mode must be fixed in the server's chat template or launch flags and declared here.
5. **Results come from `bench.sh` only.** It adds everything it can detect itself
   (GPU, driver, CPU, RAM, OS, engine and version, model file, context) as notes.

## Running

On the inference server (or any machine that reaches it and has BetterBench 0.6.1):

```sh
./bench.sh --endpoint http://localhost:8080/v1 --model llamacpp-sycl/qwen3.8-27b-q4-mtp \
  --tp 1 --pp 1 --quant Q4_K_M --kv f16 --spec mtp:5 --thinking default \
  --power-limit 200 --submitter pascal \
  --server-cmd-file my-server-cmd.txt
```

- `--quick` - smoke run (BetterBench `--quick`); recorded as `run_mode=quick`, not for submission.
- `--no-concurrency` - skip the concurrency sweep; recorded as `concurrency=off`.
- `--dry-run` - print the BetterBench command and exit.
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

Declared: `submitter tp pp quant kv spec thinking power_limit`, plus `bench_standard
run_mode concurrency`. Detected (by `collect-env.sh`): `gpu_vendor gpu_model gpu_count
gpu_vram_gb gpu_driver gpu_power_limit_w cpu_model ram_gb os kernel proxy engine
engine_version engine_slots model_file ctx`. `gpu_vram_gb` is per GPU. Anything
undetectable is `unknown`.

Engine detection: llama.cpp via `/props`, vLLM via `/version` + `/v1/models`; behind
llama-swap through `/upstream/<model>/...`, which also yields the launch command.
GPU detection: `nvidia-smi`, then `rocm-smi`, then `xpu-smi`.

## Comparing

```sh
betterbench compare A/results.json B/results.json
```

Every note that differs between two runs shows up as a mismatch chip. Compare like
with like.
