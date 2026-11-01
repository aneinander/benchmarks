# AGENTS.md

Instructions for a coding agent that helps someone benchmark their LLM server with
`bench.sh`. The human guide is [`README.md`](README.md). Read its "What you declare"
and "Models" sections too.

Your job: get one correct, comparable result folder out of `bench.sh`, and tell your
human what it says. A wrong declared value makes the result worthless, and a skipped
one makes the run fail. Correct is more important than fast.

## Rules

- **Run benchmarks only through `./bench.sh`.** Never call `betterbench run` directly,
  and never add BetterBench flags. `bench.sh` refuses them.
- **Don't edit `bench.sh`** to make a run pass. If it seems wrong, tell your human.
- **Never edit anything in a results folder.**
- **Never guess a declared value silently.** Derive it from the launch command (step 4),
  show your human what you derived, and ask when the command doesn't say.
  `--submitter` always comes from the human.
- **Ask before `--full`.** It takes much longer than the quick run, and the GPU is busy
  the whole time.
- **Don't restart, reconfigure or stop the inference server** without asking. Don't
  start other GPU work while a benchmark runs.
- **API keys** go in `$BETTERBENCH_API_KEY`, never into files or commands that get saved.

## Models

Results only count for the files in the [README's model table](README.md#models).
`bench.sh` hashes the served file when it can find it on this machine (the `model_file`
note in the dry run shows `sha256 ... (standard: NAME)`), and warns when it isn't a
standard model or `--quant` doesn't match. When the note has no hash (BetterBench on
another machine, weights it can't locate), check the SHA256 yourself with the commands
below the table. On a mismatch, tell your human before benchmarking. `--quant` for each
model is in that table.

## Procedure

### 1. Check the tools

```sh
./bench.sh --version          # bench-standard version and the BetterBench it needs
betterbench --version         # 0.6.1 or newer
python3 --version             # 3.10+
```

If something is missing, install it as the README's quick start shows. If an older one is installed, ask
before upgrading it.

### 2. Find the endpoint and the model id

```sh
curl -s http://localhost:8080/v1/models     # ids the server accepts
curl -s http://localhost:8080/running       # llama-swap only: loaded models + launch commands
```

`--endpoint` is the base URL ending in `/v1`. `--model` is the id exactly as listed.

### 3. Get the launch command

- **Behind llama-swap** (`/running` answers): nothing to do, `bench.sh` reads it.
- **Otherwise:** write the exact command the server was started with into
  `server-cmd.txt`, including `VAR=value` env prefixes and docker `-e` flags. Pass it
  with `--server-cmd-file server-cmd.txt`. Get the command from your human, a launch
  script, `docker inspect`, or `ps -o args= -p PID`.

Without it, the `launch` and `env_flags` notes read `unknown`, and those settings
swing results by 10-50%.

### 4. Derive the declared values

`--quant` comes from the README's model table. For the rest, use the launch command and confirm
with your human:

| flag | llama.cpp | vLLM |
|---|---|---|
| `--kv` | `-ctk`/`-ctv`; `f16` when unset; `q8_0/q4_0` when K and V differ | `--kv-cache-dtype`; with `auto` (or unset), the model dtype (`bf16`) |
| `--tp` / `--pp` | `1` / `1` on one GPU; multi-GPU: ask | `--tensor-parallel-size` / `--pipeline-parallel-size`; default `1` |
| `--spec` | `-md`/`--model-draft` with `--draft-max N` → `draft:N`; otherwise `none` | `--speculative-config` method and `num_speculative_tokens` → e.g. `mtp:5`, `eagle3:4`, `ngram:4`; otherwise `none` |
| `--thinking` | `--reasoning-budget 0` or `enable_thinking: false` in the template kwargs → `off`; unclear → ask | `enable_thinking`/effort in the chat template or default kwargs; nothing set → `default`; unclear → ask |
| `--submitter` | ask | ask |

`bench.sh` checks `--kv`, `--spec`, `--thinking` (and `--tp`/`--pp` for vLLM) against the
launch command and warns `--X A but the launch command sets ...: declare --X B`. Don't
just switch to B: show your human the warning and confirm.

### 5. Dry run

```sh
./bench.sh --endpoint URL --model ID --tp 1 --pp 1 --quant Q --kv KV \
  --spec none --thinking default --submitter NAME [--server-cmd-file server-cmd.txt] --dry-run
```

This sends the warm-up and the greedy check, detects everything, prints what it would
run, and stops. Check the `--note` values in that output:
- `unknown` values should have a reason (VM PCIe, no cmd file, ...).
- Any `bench.sh: warning:` line needs fixing or explaining before the real run. The dry
  run repeats them all at the end.

### 6. Quick run, then full run

Run the same command without `--dry-run`. This is a quick check of a few minutes,
ending in a summary. If it looks sane, ask your human, then run again with `--full`.
That is the result that counts. For long runs, use a terminal multiplexer or a
background job, so a dropped session doesn't kill the benchmark.

### 7. Report back

Tell your human:
- the result folder (printed after `done:`)
- the summary block: decode t/s, TTFT, concurrency, prefill, bandwidth %, power, energy
  (also in `summary.txt` in the folder)
- every warning (listed at the end of the summary), and whether the greedy check was
  `identical`
- that the folder is what they send in, and that they should check `server-cmd.txt` for
  anything private first (tokens, keys and passwords are already redacted)

## When something goes wrong

| message | what to do |
|---|---|
| `BetterBench 0.6.1 or newer required` | install or upgrade BetterBench (step 1) |
| `warm-up request failed (HTTP ...)` | server not up, or wrong `--endpoint`/`--model` (step 2). Behind llama-swap the first load can take minutes; `WARM_TIMEOUT=1800` gives it longer |
| `missing mandatory value(s)` | pass the listed flags (step 4) |
| `GPU power limit not detected here` | ask for the limit, pass `--power-limit W` or `stock` |
| `--X A but the launch command sets ...: declare --X B` | the declared value contradicts the launch command. Confirm with your human, then use B |
| `not one of the standard models` | the served file's hash isn't in the README's table: wrong file or quant. Tell your human |
| `--quant X but the model file is NAME` | use the `--quant` the table lists for that file |
| `greedy check failed (...)` | the server rejected or timed out the temperature-0 requests; read the error |
| `greedy_check=differs` | not a failure, but report it. It means nondeterministic kernels or a broken spec-decoding path |
| `--thinking X, but the greedy replies ...` | the declared thinking mode doesn't match what the server does. Ask your human which is right; never change the server yourself |
| `mem_bw` has no measured figure | no torch in the server container. Set `BW_PYTHON` to a python with GPU torch, or install `clpeak`, or leave it |
| `launch=unknown (pass --server-cmd-file)` | step 3 |
| `betterbench failed (exit N)` | read `betterbench.log` in the `.failed` folder. Usually the server crashed or ran out of memory |
| `interrupted; partial output in ....interrupted` | the run was stopped (Ctrl-C, kill, dropped session). It doesn't count; run again |
| `ReBAR ... but BetterBench reads ...` | BetterBench runs on another machine than the GPU. Use `--env-file` (below) |

**BetterBench on another machine than the GPU:** run
`./bench.sh collect-env --endpoint URL --model ID > env.txt` on the GPU machine, then
add `--env-file env.txt` on the BetterBench machine. Power isn't sampled then. Run
`./bench.sh power > power.csv` on the GPU machine during the run if your human wants it.

What each recorded value means, and how it's detected:
[`dev/DETAILS.md`](dev/DETAILS.md).
