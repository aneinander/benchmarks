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
Then follow AGENTS.md step by step. Start with its round 1 questions, all in one
numbered message, before you run anything. Install what's missing, find my server and
model, check the model hash, work out the values to declare from my server's launch
command, and do a dry run. Show me the round 2 table of values to confirm, then do a
quick run, and ask me the round 3 questions before the --full run.
```

The agent follows [`AGENTS.md`](AGENTS.md), a step-by-step procedure with rules that keep
the result valid. It asks you three rounds of numbered questions with defaults already
filled in, so you mostly confirm:

1. **Before it starts:** your name, the server's address, whether it needs an API key
   (you set that yourself, it never goes in the chat), how the server was started, the
   model, the number of GPUs, the power limit, anything unusual.
2. **After the dry run:** a table of every value it will declare and where each came
   from, plus every warning.
3. **Before the full run:** whether to start it, and in tmux or the background.

Without an agent, do it yourself as described below.

## Quick start

On the machine with the GPU and the inference server (Linux, Python 3.10+):

```sh
pip install git+https://github.com/GGZ14/BetterBench   # BetterBench 0.6.1 or newer
curl -O https://raw.githubusercontent.com/aneinander/benchmarks/main/bench.sh && chmod +x bench.sh
```

Serve one of the [models](#models) with any OpenAI-compatible server (llama.cpp, vLLM,
...). Then copy this into a text editor, fill in the top part, and paste it into the
terminal in the folder with `bench.sh`. [What you declare](#what-you-declare) explains
how to read each value off your server's launch command.

```sh
# ─── fill these in ──────────────────────────────────────────────────────────
SUBMITTER=your-name                 # you
ENDPOINT=http://localhost:8080/v1   # the server's base URL, ending in /v1
MODEL=MODEL_ID                      # the id exactly as `curl $ENDPOINT/models` lists it
QUANT=Q4_K_M                        # from the model table: Q4_K_M, UD-Q4_K_M or w4a16
KV=f16                              # KV cache type: f16, q8_0, q8_0/f16 (K/V), bf16, fp8
TP=1                                # tensor parallel size (1 on one GPU)
PP=1                                # pipeline parallel size (1 on one GPU)
SPEC=none                           # speculative decoding: none, mtp:5, draft:8, ...
THINKING=default                    # set on the server: default, off, on, low, medium, high
SERVER_CMD=server-cmd.txt           # file with the command that started the server;
                                    #   leave empty behind llama-swap
# export BETTERBENCH_API_KEY=...    # only if the server needs a key; never saved
# ─── then run (add --full for the run that counts) ──────────────────────────
./bench.sh --submitter "$SUBMITTER" --endpoint "$ENDPOINT" --model "$MODEL" \
  --quant "$QUANT" --kv "$KV" --tp "$TP" --pp "$PP" --spec "$SPEC" --thinking "$THINKING" \
  ${SERVER_CMD:+--server-cmd-file "$SERVER_CMD"}
```

Add `--dry-run` first to see what it detects and every warning without running the
benchmark. To write the launch command into `server-cmd.txt`, paste it into the file
exactly as you typed it, env vars and docker `-e` flags included, or for a running
process:

```sh
ps -o args= -p "$(pgrep -f 'llama-server|vllm' | head -n1)" > server-cmd.txt
```

### Save your values

So you don't fill these in every time, save them once:

```sh
./bench.sh init        # writes bench.local.env next to bench.sh, every value explained
nano bench.local.env   # fill in the empty ones
./bench.sh --dry-run   # then just this; --full for the run that counts
```

`bench.local.env` holds one `BENCH_...=value` per line (`BENCH_SUBMITTER`,
`BENCH_ENDPOINT`, `BENCH_MODEL`, `BENCH_SERVER_CMD_FILE`, `BENCH_QUANT`, `BENCH_KV`,
`BENCH_TP`, `BENCH_PP`, `BENCH_SPEC`, `BENCH_THINKING`, and optional ones like
`BENCH_POWER_LIMIT`). It's read as plain values, never run as a script. Relative paths
in it are relative to the file. A flag on the command line wins over an environment
variable, which wins over the file. Every run prints which values came from the file.
For a second server or model, keep a second file and pass `--settings other.env`.
The API key never goes in it: `bench.sh` refuses `BETTERBENCH_API_KEY` there. Export
it in your shell.

### The run

Without `--full`, that's a **quick check** of a few minutes. If the summary at the end looks right,
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

`bench.sh` does this for you when it finds the weights on the machine (also through
docker `-v` mounts and the Hugging Face cache). It records the hash in the `model_file`
note and warns when the file isn't in this table or `--quant` doesn't match. The hash is
cached, so only the first run spends time on it; `HASH_MODEL=0` skips it.

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

`bench.sh` checks `--kv`, `--spec`, `--thinking` and, for vLLM, `--tp`/`--pp` against the
launch command, and warns with the value the command implies.

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

| `--settings FILE` | read saved values from FILE instead of `bench.local.env` |

Every value can also be saved, see [Save your values](#save-your-values).

## What a run does

1. **Warm-up:** one request. Behind llama-swap that loads the model, which can take minutes.
2. **Greedy check:** the same prompt 3 times at temperature 0. The output should be
   identical. Its hash lets you compare setups: the same model and quant should give
   the same text everywhere. The replies also show whether the model reasoned, and
   `bench.sh` warns if that contradicts your `--thinking`.
3. **Detection:** GPU, driver, power limit, PCIe link and ReBAR, CPU, RAM, OS, the
   engine and its exact build, the docker image, the model file, its size and hash,
   launch and env flags. **Memory bandwidth** is measured with random data inside the server's
   container. Zeros compress on Arc and read about 3× too high.
4. **BetterBench:** single-stream decode per category, a prefill sweep and a
   concurrency sweep. GPU power, energy and clock are sampled throughout.
5. **Summary:** decode t/s, TTFT, concurrency and prefill throughput, % of memory
   bandwidth reached, average power and clock, J per token, and every warning again.

Ctrl-C or a `kill` stops the run at once; the folder is renamed `.interrupted` and
doesn't count.

Everything is recorded as notes in `results.json`, so it shows in the report and in
`compare`. Example notes:

```
gpu          1x Intel Arc Pro B65 · 32 GB · driver 1.17.39395
pcie         Gen4 x16 · ReBAR on
engine       llama.cpp b11913-7aca0d2ea · 4 slots · image ghcr.io/ggml-org/llama.cpp:server-intel@1a2b3c4d5e6f
model_file   /models/Qwen3.5-9B-Q4_K_M.gguf · 5.68 GB · sha256 03b74727a860 (standard: Qwen3.5-9B-Q4_K_M.gguf)
launch       b 4096 · ub 1024 · fa on · ctk q8_0 · ctv q8_0 · ngl 99 · np 4
env_flags    GGML_SYCL_DISABLE_OPT=0 · ZES_ENABLE_SYSMAN=1
mem_bw       532 GB/s measured (torch copy, random, 1024 MiB) · 608 GB/s declared
greedy_check identical 3/3 · 1d5105fe09dc
thinking     default · reasoning seen
run          full · concurrency on · bench-standard 0.5.0 · betterbench 0.6.1
```

## Your result

Each run makes a folder `results/<date>-<you>-<model>-<engine>-<gpu>/`:

| file | what |
|---|---|
| `results.html` | the report; open it in a browser |
| `results.json` | the results, with all notes |
| `meta.json` | every raw value, plus energy, % of bandwidth and the warnings |
| `summary.txt` | the summary printed at the end |
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
| `--X A but the launch command sets ...: declare --X B` | the declared value contradicts the server's launch command; check it and use B |
| `not one of the standard models` | the served file's hash isn't in the [table](#models): wrong file or quant |
| `greedy check failed` | the temperature-0 requests failed; the error is in the message |
| `greedy_check=differs` | not fatal. Kernels or spec decoding aren't deterministic; mention it |
| `--thinking off, but the greedy replies contain reasoning` | thinking is still on in the server's chat template or flags. Fix the server, or declare what it does |
| `launch=unknown` | pass `--server-cmd-file` |
| `betterbench failed` | see `betterbench.log` in the `.failed` folder, usually a server crash or OOM |
| `interrupted` | the run was stopped; the `.interrupted` folder doesn't count, run again |

## Rules

- Use `bench.sh` unmodified, and only the models above.
- Don't edit results.
- Only `--full` runs count.

## Developing

`bench.sh` is generated: the sources are in [`dev/`](dev/), and `sh dev/build.sh` packs
them into the one file. Commit `dev/` and the rebuilt `bench.sh` together. `sh dev/test.sh`
runs the end-to-end tests against a fake server and a fake BetterBench, no GPU needed.

Bugs and ideas: open an issue on [GitHub](https://github.com/aneinander/benchmarks/issues).
The GitHub repo is a mirror, updated automatically from the maintainers' source repo.
Changes are made there, and anything pushed or merged on GitHub directly gets overwritten. What each
recorded value means and how it's detected: [`dev/DETAILS.md`](dev/DETAILS.md).
