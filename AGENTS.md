# AGENTS.md

Instructions for a coding agent that helps someone benchmark their LLM server with
`bench.sh`. The human guide is [`README.md`](README.md). Read its "What you declare"
and "Models" sections too.

Your job: get one correct, comparable result folder out of `bench.sh`, and tell your
human what it says. A wrong declared value makes the result worthless, and a skipped
one makes the run fail. Correct is more important than fast.

## Rules

- **Ask in the three rounds below**, not one question at a time: round 1 before you
  start, round 2 after detecting, round 3 before `--full`. A round is one batch,
  asked before you do anything else, never spread across turns. Number the questions,
  and fill in what you already know as a default they only have to confirm. "Don't
  know, find it" is always a valid answer.
  - **With a question tool** (e.g. `ask_user_question` in Pi, `AskUserQuestion` in
    Claude Code), use it for each round. If a round has more questions than one call
    allows, split it into consecutive calls of the tool's maximum, in the original
    order, and keep the numbers: with 4 per call, round 1 is 1-4, 5-8, 9-10. Make the
    bracketed default the first option, marked as recommended. Leave the free-text
    answer ("Other") open for anything that needs typing: a name, a URL, a launch
    command, watts. A free-text question (1, 3, 5) has no real choices, so give it the
    default and "find it" as options.
  - **Without one,** ask in one numbered message, answered as `1: pascal, 2: yes, ...`.
  - **Round 2's table is always a plain message**, so it stays readable. Only its
    confirmation ("ok", or correct line N) may go through the tool.
- **Run benchmarks only through `./bench.sh`.** Never call `betterbench run` directly,
  and never add BetterBench flags. `bench.sh` refuses them.
- **Don't edit `bench.sh`** to make a run pass. If it seems wrong, tell your human.
- **Never edit anything in a results folder.**
- **Never guess a declared value silently.** Derive it from the launch command, show
  it in round 2, and ask when the command doesn't say. `--submitter` always comes from
  the human.
- **Ask before `--full`** (round 3). It takes much longer than the quick run, and the
  GPU is busy the whole time.
- **Don't restart, reconfigure or stop the inference server** without asking. Don't
  start other GPU work while a benchmark runs.
- **API keys** go in `$BETTERBENCH_API_KEY`, never into files, saved commands or the
  chat. Ask your human to set it themselves in the shell you use.

## Models

Results only count for the files in the [README's model table](README.md#models).
`bench.sh` hashes the served file when it can find it on this machine (the `model_file`
note in the dry run shows `sha256 ... (standard: NAME)`), and warns when it isn't a
standard model or `--quant` doesn't match. When the note has no hash (BetterBench on
another machine, weights it can't locate), check the SHA256 yourself with the commands
below the table. On a mismatch, tell your human before benchmarking. `--quant` for each
model is in that table.

## Procedure

### 1. Round 1: the setup questions

Ask these as one batch before running anything: with a question tool, in calls of its
maximum (1-4, 5-8, 9-10 with 4 per call); without one, in one message like this. Put the
default in brackets.

```text
Before I start, a few questions. Answer by number; "find it" is fine for any of them.

1. Your name for the results (--submitter):
2. Is this the machine with the GPU and the inference server? [yes]
3. The server's address [http://localhost:8080/v1]:
4. Does the server need an API key? [no]
   If yes, don't paste it here: run  export BETTERBENCH_API_KEY=...  in the
   terminal I work in, and tell me when it's set.
5. How was the server started? llama-swap / docker / a script / a command.
   Paste the command or the script's path if you have it [find it]:
6. Which model from the README table is loaded? [find it]
7. How many GPUs does the server use? [1]
8. Is the GPU power limit changed from stock? [stock / watts / don't know]
9. Anything unusual? A VM, undervolt, custom kernels, a patched engine [no]
10. May I install BetterBench if it's missing? [yes; I'll ask before upgrading one]
```

If `bench.local.env` exists next to `bench.sh` (your human saved values with
`./bench.sh init`), read it first and put its values in as the defaults. It never holds
the API key, so it's safe to read. Still ask every question; a saved value can be stale.

What the answers mean:
- **2 = no:** BetterBench runs elsewhere. Use `--env-file` ([below](#when-something-goes-wrong)).
- **4 = yes:** wait until they say it's set. Check with `[ -n "$BETTERBENCH_API_KEY" ]`,
  never by printing it.
- **5:** a pasted command goes into `server-cmd.txt` exactly as given (step 3).
- **7 > 1 with llama.cpp:** ask how the model is split (`-sm`, `-ts`) for `--tp`/`--pp`.
- **8:** watts → `--power-limit W`. `stock` or "don't know" → let `bench.sh` read it.
- **9:** a VM → ask for the PCIe link if they know it (`--pcie "Gen4 x16"`).
  Anything else → `--special-notes "..."`.

### 2. Check the tools

```sh
./bench.sh --version          # bench-standard version and the BetterBench it needs
betterbench --version         # 0.6.1 or newer
python3 --version             # 3.10+
```

If something is missing, install it as the README's quick start shows (if answer 10
allows it). If an older one is installed, ask before upgrading it.

### 3. Find the endpoint, the model id and the launch command

Start from the answers to 3, 5 and 6, and verify them:

```sh
curl -s http://localhost:8080/v1/models     # ids the server accepts
curl -s http://localhost:8080/running       # llama-swap only: loaded models + launch commands
```

`--endpoint` is the base URL ending in `/v1`. `--model` is the id exactly as listed.
Nothing answers: look for listening ports (`ss -ltnp`) and containers (`docker ps`).

The launch command:
- **Behind llama-swap** (`/running` answers): nothing to do, `bench.sh` reads it.
- **Otherwise:** write the exact command the server was started with into
  `server-cmd.txt`, including `VAR=value` env prefixes and docker `-e` flags. Pass it
  with `--server-cmd-file server-cmd.txt`. Get it from answer 5, a launch script,
  `docker inspect`, or `ps -o args= -p PID`. Env vars set in llama-swap's `env:` block
  or in the shell that started the server aren't in any of these: ask.

Without it, the `launch` and `env_flags` notes read `unknown`, and those settings
swing results by 10-50%.

### 4. Derive the declared values

`--quant` comes from the README's model table. The rest comes from the launch command:

| flag | llama.cpp | vLLM |
|---|---|---|
| `--kv` | `-ctk`/`-ctv`; `f16` when unset; `q8_0/q4_0` when K and V differ | `--kv-cache-dtype`; with `auto` (or unset), the model dtype (`bf16`) |
| `--tp` / `--pp` | `1` / `1` on one GPU; multi-GPU: ask | `--tensor-parallel-size` / `--pipeline-parallel-size`; default `1` |
| `--spec` | `-md`/`--model-draft` with `--draft-max N` → `draft:N`; otherwise `none` | `--speculative-config` method and `num_speculative_tokens` → e.g. `mtp:5`, `eagle3:4`, `ngram:4`; otherwise `none` |
| `--thinking` | `--reasoning-budget 0` or `enable_thinking: false` in the template kwargs → `off`; unclear → ask | `enable_thinking`/effort in the chat template or default kwargs; nothing set → `default`; unclear → ask |
| `--submitter` | answer 1 | answer 1 |

### 5. Dry run

```sh
./bench.sh --endpoint URL --model ID --tp 1 --pp 1 --quant Q --kv KV \
  --spec none --thinking default --submitter NAME [--server-cmd-file server-cmd.txt] --dry-run
```

This sends the warm-up and the greedy check, detects everything, prints what it would
run, and stops. `bench.sh` checks `--kv`, `--spec`, `--thinking` (and `--tp`/`--pp` for
vLLM) against the launch command and the model file against the README's table, and
lists every warning at the end.

### 6. Round 2: confirm the values

Show your human one table with every declared value, where it came from, and what the
dry run found, then ask them to confirm or correct it:

```text
Here is what I'll declare. Reply "ok", or correct a line by number.

  #  flag           value        from
  1  --endpoint     http://localhost:8080/v1   your answer, /v1/models answers
  2  --model        qwen3.5-9b   /v1/models
  3  --quant        Q4_K_M       README table; hash matches Qwen3.5-9B-Q4_K_M.gguf
  4  --kv           q8_0/f16     -ctk q8_0, no -ctv (V defaults to f16)
  5  --tp / --pp    1 / 1        one GPU
  6  --spec         none         no -md in the launch command
  7  --thinking     default      nothing in the command; greedy replies show reasoning
  8  --submitter    pascal       your answer
  9  --power-limit  (detected)   GPU reports 250 W
  10 launch         b 4096 · ub 1024 · fa on · ...   server-cmd.txt

Warnings from the dry run:
  - ... (each one, with what you think it means)

Notes that read unknown, and why:
  - ...
```

Every `bench.sh: warning:` needs fixing or an explanation before the real run. For
`--X A but the launch command sets ...: declare --X B`, don't just switch to B: show it
here and let your human decide. Rerun the dry run after any change.

Once the values are confirmed, offer to save them, so the next run needs no typing:
`./bench.sh init` writes `bench.local.env` (it refuses to overwrite one), then fill in
the confirmed values. Ask before changing an existing file. After that the run is
just `./bench.sh` plus `--dry-run` or `--full`.

### 7. Quick run

Run the same command without `--dry-run`. This is a quick check of a few minutes,
ending in a summary (also saved as `summary.txt` in the result folder).

### 8. Round 3: before the full run

```text
The quick run finished: decode X t/s, TTFT Y ms, N warnings (listed below).

1. Start the --full run now? It takes much longer and keeps the GPU busy. [yes]
2. Run it in tmux/screen or in the background, so a dropped session doesn't stop it? [yes]
3. Is anything else using the GPU right now that should be stopped first? [no]
```

### 9. Report back

Tell your human:
- the result folder (printed after `done:`)
- the summary block: decode t/s, TTFT, concurrency, prefill, bandwidth %, power, energy
- every warning (listed at the end of the summary), and whether the greedy check was
  `identical`
- that the folder is what they send in, and that they should check `server-cmd.txt` for
  anything private first (tokens, keys and passwords are already redacted)

## When something goes wrong

| message | what to do |
|---|---|
| `BetterBench 0.6.1 or newer required` | install or upgrade BetterBench (step 2) |
| `warm-up request failed (HTTP ...)` | server not up, or wrong `--endpoint`/`--model` (step 3). HTTP 401/403: the server needs an API key (round 1, question 4). Behind llama-swap the first load can take minutes; `WARM_TIMEOUT=1800` gives it longer |
| `missing mandatory value(s)` | pass the listed flags (step 4) |
| `GPU power limit not detected here` | ask for the limit, pass `--power-limit W` or `stock` |
| `--X A but the launch command sets ...: declare --X B` | the declared value contradicts the launch command. Confirm with your human (round 2), then use B |
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
