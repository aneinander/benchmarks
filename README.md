# bench-standard

Benchmark your LLM server the way everyone in the group does, so results are
comparable. It's one script around [BetterBench](https://github.com/GGZ14/BetterBench).
It runs the benchmark and records the setup the numbers were measured on: GPU,
PCIe, power, engine and version, launch flags, env flags, memory bandwidth and
more. Two results can only be compared when you know how they differ.

## With a coding agent

Copy this into Claude Code, Codex, Cursor or any agent that can run commands on the
machine with your GPU:

```text
Help me benchmark my LLM server with bench-standard. Read these two files first:
https://raw.githubusercontent.com/aneinander/benchmarks/main/AGENTS.md
https://raw.githubusercontent.com/aneinander/benchmarks/main/README.md
Then follow AGENTS.md step by step: install what's missing, find my server and model,
check the model hash, work out the values to declare from my server's launch command,
and do a dry run and a quick run. Ask me for anything you can't find out yourself,
and ask me before the --full run.
```

The agent follows [`AGENTS.md`](AGENTS.md), a step-by-step procedure with rules that keep
the result valid. Without an agent, do it yourself as described below.

## Quick start

On the machine with the GPU and the inference server (Linux, Python 3.10+):

```sh
pip install git+https://github.com/GGZ14/BetterBench   # BetterBench 0.6.1 or newer
curl -O https://raw.githubusercontent.com/aneinander/benchmarks/main/bench.sh && chmod +x bench.sh
```

Serve one of the [models](#models) with any OpenAI-compatible server (llama.cpp, vLLM,
...), then:

```sh
./bench.sh --endpoint http://localhost:8080/v1 --model MODEL_ID \
  --tp 1 --pp 1 --quant Q4_K_M --kv f16 --spec none --thinking default \
  --submitter YOUR_NAME --server-cmd-file server-cmd.txt
```

That's a **quick check** of a few minutes. If the summary at the end looks right,
run the same command with `--full`. That's the result that counts. It takes much
longer and keeps the GPU busy the whole time.

`./bench.sh --help` lists every option.

## Models

Results only count for these files. Check the hash before you benchmark: a
different file is a different benchmark.

| model | engine | Hugging Face repo | file | `--quant` |
|---|---|---|---|---|
| Qwen3.5-9B | llama.cpp | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `Qwen3.5-9B-Q4_K_M.gguf` | `Q4_K_M` |
| Qwen3.8-27B | llama.cpp | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `UD-Q4_K_M` |
| Qwen3.5-9B | vLLM & co. | [`RedHatAI/Qwen3.5-9B-quantized.w4a16`](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16) | the whole repo | `w4a16` |
| Qwen3.8-27B | vLLM & co. | [`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | the whole repo | `w4a16` |

| file | SHA256 |
|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| `RedHatAI/Qwen3.5-9B-quantized.w4a16` | `cca16f9855b9f92eee5876ddc8656ed6397316ffc65b9f17ca1939a9549e4c26` |
| `RedHatAI/Qwen3.8-27B-INT4` | `bffa6b23fa11be557fe9de4b122a8bb99f537ad7a1a4effa702f7d0c677778f7` |

```sh
sha256sum Qwen3.5-9B-Q4_K_M.gguf                                       # a GGUF
cd MODEL_DIR && sha256sum *.safetensors | LC_ALL=C sort -k2 | sha256sum  # a repo
```

For a repo, the hash is over the per-file hashes in name order. Keep `LC_ALL=C`:
other locales sort the files differently and give a different hash.

- The RedHatAI checkpoints are int4 weights with 16-bit activations
  (compressed-tensors). They include an MTP head, so `--spec mtp:N` works.
- The 27B GGUF repo has no plain Q4_K_M, only Unsloth's dynamic `UD-Q4_K_M`.
- Hashes are from Hugging Face as of 2026-10-29. If a repo changes its files,
  this table gets updated.

## What you declare

These settings change the numbers, but no API reports them, so you state them. Most
can be read off the command you started the server with:

| flag | what | llama.cpp | vLLM |
|---|---|---|---|
| `--quant` | weight quantization | see [models](#models) | see [models](#models) |
| `--kv` | KV cache type | `-ctk`/`-ctv`; `f16` if unset; `q8_0/q4_0` if K and V differ | `--kv-cache-dtype`; `auto` or unset = model dtype, e.g. `bf16` |
| `--tp` / `--pp` | tensor / pipeline parallel size | `1` / `1` on one GPU | `--tensor-parallel-size` / `--pipeline-parallel-size` |
| `--spec` | speculative decoding | `-md` + `--draft-max N` → `draft:N`; else `none` | `--speculative-config` → `mtp:5`, `eagle3:4`, `ngram:4`, ...; else `none` |
| `--thinking` | thinking mode set **on the server** | `--reasoning-budget 0` → `off` | chat template / default kwargs; nothing set → `default` |
| `--submitter` | your name | | |

`--spec` methods are `mtp`, `eagle`, `eagle3`, `ngram` and `draft`. `--thinking` values are
`off`, `on`, `low`, `medium`, `high` and `default`. BetterBench can't switch thinking on or
off per request, so it has to be set in the server's chat template or launch flags.

**`--server-cmd-file`** is a text file with the exact command that started your server,
including env vars (`VAR=x ...`, docker `-e VAR=x`). It's how batch sizes, flash
attention and env flags get recorded, and those swing results by 10-50%. Behind
llama-swap you can leave it out, because the command is read automatically.

## Options

| flag | for |
|---|---|
| `--full` | the run that counts (default: a quick check) |
| `--verbose` | show all of BetterBench's output, not just progress and the summary |
| `--dry-run` | detect everything, print what would run, stop |
| `--mem-bw 608` | the card's spec memory bandwidth in GB/s (it's measured too) |
| `--power-limit 250` | when the power limit can't be read from the GPU (or `stock`) |
| `--pcie "Gen4 x16"` | inside a VM, where the PCIe link can't be read |
| `--special-notes "..."` | anything unusual: custom kernels, undervolt, patched engine |
| `--api-key KEY` | if the server needs one; or `$BETTERBENCH_API_KEY`. Never stored |
| `--no-concurrency` | skip the concurrency sweep |
| `--results-dir DIR` | where run folders go (default `results/` next to `bench.sh`) |

The declared values can also come from the environment (`BENCH_SUBMITTER=pascal`,
`BENCH_QUANT=...`, ...). Machine-local defaults like these go in a `bench.local.env`
file next to `bench.sh`.

## What a run does

1. **Warm-up:** one request. Behind llama-swap that loads the model, which can take minutes.
2. **Greedy check:** the same prompt 3 times at temperature 0. The output should be
   identical. Its hash lets you compare setups: the same model and quant should give
   the same text everywhere.
3. **Detection:** GPU, driver, power limit, PCIe link and ReBAR, CPU, RAM, OS, the
   engine and its exact build, the docker image, the model file and size, launch and
   env flags. **Memory bandwidth** is measured with random data inside the server's
   container. Zeros compress on Arc and read about 3× too high.
4. **BetterBench:** single-stream decode per category, a prefill sweep and a
   concurrency sweep. GPU power, energy and clock are sampled throughout.
5. **Summary:** decode t/s, TTFT, concurrency and prefill throughput, % of memory
   bandwidth reached, average power and clock, and J per token.

Everything is recorded as notes in `results.json`, so it shows in the report and in
`compare`. Example notes:

```
gpu          1x Intel Arc Pro B65 · 32 GB · driver 1.17.39395
pcie         Gen4 x16 · ReBAR on
engine       llama.cpp b11913-7aca0d2ea · 4 slots · image ghcr.io/ggml-org/llama.cpp:server-intel@1a2b3c4d5e6f
launch       b 4096 · ub 1024 · fa on · ctk q8_0 · ctv q8_0 · ngl 99 · np 4
env_flags    GGML_SYCL_DISABLE_OPT=0 · ZES_ENABLE_SYSMAN=1
mem_bw       532 GB/s measured (torch copy, random, 1024 MiB) · 608 GB/s declared
greedy_check identical 3/3 · 1d5105fe09dc
run          full · concurrency on · bench-standard 0.4.0 · betterbench 0.6.1
```

## Your result

Each run makes a folder `results/<date>-<you>-<model>-<engine>-<gpu>/`:

| file | what |
|---|---|
| `results.html` | the report; open it in a browser |
| `results.json` | the results, with all notes |
| `meta.json` | every raw value, plus energy and % of bandwidth |
| `server-cmd.txt` | your server's launch command, secrets redacted |
| `config.json`, `power.csv`, `betterbench.log` | the config used, power samples, the full output |

**Send the whole folder of a `--full` run.** Check `server-cmd.txt` for anything private
first. Tokens, keys and passwords are redacted automatically.

Compare two runs:

```sh
betterbench compare A/results.json B/results.json
```

Every recorded difference shows up as a mismatch chip, so you can see exactly what
differs between the two setups.

## BetterBench on another machine

If BetterBench can't run on the GPU machine, run this there:

```sh
./bench.sh collect-env --endpoint http://localhost:8080/v1 --model MODEL_ID > env.txt
```

Then add `--env-file env.txt` to the `bench.sh` command on the other machine. Power
isn't sampled in that case. Run `./bench.sh power > power.csv` on the GPU machine
during the benchmark if you want it.

## If something goes wrong

| message | fix |
|---|---|
| `BetterBench 0.6.1 or newer required` | install or upgrade BetterBench (quick start) |
| `warm-up request failed` | is the server up? Are `--endpoint` and `--model` right? `curl URL/models` lists the ids. For a slow first load, try `WARM_TIMEOUT=1800` |
| `missing mandatory value(s)` | pass the listed flags |
| `GPU power limit not detected` | pass `--power-limit W` (or `stock`) |
| `--kv X but the launch command sets Y` | your `--kv` is wrong; use what the command sets |
| `greedy_check=differs` | not fatal. Kernels or spec decoding aren't deterministic; mention it |
| `launch=unknown` | pass `--server-cmd-file` |
| `betterbench failed` | see `betterbench.log` in the `.failed` folder, usually a server crash or OOM |

## Rules

- Use `bench.sh` unmodified, and only the models above.
- Don't edit results.
- Only `--full` runs count.

## Developing

`bench.sh` is generated: the sources are in [`dev/`](dev/), and `sh dev/build.sh` packs
them into the one file. Commit `dev/` and the rebuilt `bench.sh` together. What each
recorded value means and how it's detected: [`dev/DETAILS.md`](dev/DETAILS.md).
