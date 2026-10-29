# AGENTS.md

Instructions for a coding agent that helps someone benchmark their LLM server with
this repo. The human-facing docs are [`README.md`](README.md) and
[`DETAILS.md`](DETAILS.md).

Your job: get one correct, comparable result folder out of `bench.sh`, and tell your
human what it says. A wrong declared value makes the result worthless, and a skipped
one makes the run fail. Correct is more important than fast.

## Rules

- **Run benchmarks only through `./bench.sh`.** Never call `betterbench run` directly,
  and never add BetterBench flags. `bench.sh` refuses them.
- **Don't change the standard to make a run pass.** Leave `config.json`,
  `standard.env`, `bench.sh`, `collect-env.sh`, `gpu-power.sh` and `mem-bw.py` alone. If one
  of them seems wrong, tell your human instead.
- **Never edit anything in a results folder.**
- **Never guess a declared value silently.** Derive it from the launch command (below),
  show your human what you derived, and ask when the command doesn't say.
  `--submitter` always comes from the human.
- **Ask before `--full`.** It takes much longer than the quick run, and the GPU is busy
  the whole time.
- **Don't restart, reconfigure or stop the inference server** without asking. Don't
  start other GPU work while a benchmark runs.
- **API keys** go in `$BETTERBENCH_API_KEY`, never into files or committed commands.
- **Don't commit results.** `results/` and `bench.local.env` are git-ignored on purpose.

## Procedure

### 1. Check the tools

```sh
betterbench --version        # must print 0.6.1
python3 --version            # 3.10+
```

If BetterBench is missing or a different version, install it as the README shows
(`git checkout v0.6.1 && pip install -e .`). If an existing install has a different
version, ask before touching it.

### 2. Find the endpoint and the model id

```sh
curl -s http://localhost:8080/v1/models     # ids the server accepts
curl -s http://localhost:8080/running       # llama-swap only: loaded models + launch commands
```

`--endpoint` is the base URL ending in `/v1`. `--model` is the id exactly as listed.

### 3. Get the launch command

- **Behind llama-swap** (`/running` answers): nothing to do, `bench.sh` reads it.
- **Otherwise:** write the exact command the server was started with into
  `server-cmd.txt`, including any `VAR=value` env prefixes and docker `-e` flags. Pass it
  with `--server-cmd-file server-cmd.txt`. Get the command from your human, a
  launch script, `docker inspect`, or `ps -o args= -p PID`.

Without it, the `launch` and `env_flags` notes read `unknown`, and those settings
swing results by 10-50%.

### 4. Check the model, derive the declared values

Results only count for the models in [`models.md`](models.md). Check the served file's
SHA256 against that table, with the commands at its end. On a mismatch, tell your
human before benchmarking. For those models, `--quant` is listed in `models.md`.

For the rest, use the launch command and confirm with your human:

| flag | llama.cpp | vLLM |
|---|---|---|
| `--quant` | quant in the GGUF filename (`Q4_K_M`, `Q8_0`, ...) | `--quantization`, or the checkpoint's format (`gptq-int4`, `awq`, `fp8`); `bf16`/`fp16` when unquantized |
| `--kv` | `-ctk`/`-ctv`; `f16` when unset; `q8_0/q4_0` when K and V differ | `--kv-cache-dtype`; with `auto` (or unset), the model dtype (`bf16`) |
| `--tp` / `--pp` | `1` / `1` on one GPU; multi-GPU: ask | `--tensor-parallel-size` / `--pipeline-parallel-size`; default `1` |
| `--spec` | `-md`/`--model-draft` with `--draft-max N` → `draft:N`; otherwise `none` | `--speculative-config` method and `num_speculative_tokens` → e.g. `mtp:5`, `eagle3:4`, `ngram:4`; otherwise `none` |
| `--thinking` | `--reasoning-budget 0` or `enable_thinking: false` in the template kwargs → `off`; unclear → ask | `enable_thinking`/effort in the chat template or default kwargs; nothing set → `default`; unclear → ask |
| `--submitter` | ask | ask |

`--spec` methods are `mtp`, `eagle`, `eagle3`, `ngram` and `draft`. `--thinking` values are
`off`, `on`, `low`, `medium`, `high` and `default`.

### 5. Dry run

```sh
./bench.sh --endpoint URL --model ID --tp 1 --pp 1 --quant Q --kv KV \
  --spec none --thinking default --submitter NAME [--server-cmd-file server-cmd.txt] --dry-run
```

This sends the warm-up and the greedy check, detects everything, prints the notes it
would record, and stops. Read the notes:
- `unknown` values should have a reason (VM PCIe, no cmd file, ...).
- Any `bench.sh: warning:` line needs fixing or explaining before the real run.

### 6. Quick run, then full run

Run the same command without `--dry-run`. This is a quick check of a few minutes,
ending in a summary. If it looks sane, ask your human, then run again with `--full`.
That is the result that counts.

For long runs, use a terminal multiplexer or a background job, so a dropped
session doesn't kill the benchmark.

### 7. Report back

Tell your human:
- the result folder (printed after `done:`)
- the summary block: decode t/s, TTFT, concurrency, prefill, bandwidth %, power, energy
- every warning, and whether the greedy check was `identical`
- that the folder is what they send in, and that they should check `server-cmd.txt` for
  anything private first

## When something goes wrong

| message | what to do |
|---|---|
| `BetterBench 0.6.1 required, found ...` | install 0.6.1 (step 1) |
| `warm-up request failed (HTTP ...)` | server not up, or wrong `--endpoint`/`--model` (step 2). Behind llama-swap the first load can take minutes; `WARM_TIMEOUT=1800` gives it longer |
| `missing mandatory value(s)` | pass the listed flags (step 4) |
| `GPU power limit not detected here` | ask for the limit, pass `--power-limit W` or `stock` |
| `--kv X but the launch command sets Y` | the declared `--kv` is wrong. Use what the command sets |
| `greedy_check=differs` | not a failure, but report it. It means nondeterministic kernels or a broken spec-decoding path |
| `mem_bw` has no measured figure | no torch in the server container. Set `BW_PYTHON` to a python with GPU torch, or install `clpeak`, or leave it |
| `launch=unknown (pass --server-cmd-file)` | step 3 |
| `betterbench failed (exit N)` | read `betterbench.log` in the `.failed` folder. Usually the server crashed or ran out of memory |
| `ReBAR ... but BetterBench reads ...` | BetterBench runs on another machine than the GPU. Use `--env-file` (see README) |

BetterBench on a different machine than the GPU: run `collect-env.sh` on the GPU
machine and pass its output with `--env-file` (README, "More").
