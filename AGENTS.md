# bench-standard

Benchmark your LLM server the way everyone in the group does. You need one file:

```sh
pip install git+https://github.com/GGZ14/BetterBench   # BetterBench 0.6.1 or newer
curl -O https://git.linxweiler.xyz/miga/benchmarks/raw/branch/main/bench.sh && chmod +x bench.sh
./bench.sh --help
```

**Humans:** `./bench.sh --help` has everything. Serve one of the [models](#models) below,
run it, and send in the `results/<run>/` folder of a `--full` run.

**Agents:** the rest of this file is for you.

---

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

Results only count for these files. Check the hash before benchmarking. On a mismatch,
tell your human.

| model | engine | Hugging Face repo | file | `--quant` | SHA256 |
|---|---|---|---|---|---|
| Qwen3.5-9B | llama.cpp | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Q4_K_M.gguf` | `Q4_K_M` | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| Qwen3.8-27B | llama.cpp | `unsloth/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `UD-Q4_K_M` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Qwen3.5-9B | vLLM & co. | `RedHatAI/Qwen3.5-9B-quantized.w4a16` | the repo | `w4a16` | `cca16f9855b9f92eee5876ddc8656ed6397316ffc65b9f17ca1939a9549e4c26` |
| Qwen3.8-27B | vLLM & co. | `RedHatAI/Qwen3.8-27B-INT4` | the repo | `w4a16` | `bffa6b23fa11be557fe9de4b122a8bb99f537ad7a1a4effa702f7d0c677778f7` |

```sh
sha256sum Qwen3.5-9B-Q4_K_M.gguf                                      # a GGUF
cd MODEL_DIR && sha256sum *.safetensors | LC_ALL=C sort -k2 | sha256sum # a repo
```

`LC_ALL=C` matters: other locales sort the files differently and give another hash.
The RedHatAI repos include an MTP head for `--spec mtp:N`. The 27B GGUF repo has no plain
Q4_K_M, only Unsloth's `UD-Q4_K_M`. Hashes are as of 2026-10-29.

## Procedure

### 1. Check the tools

```sh
./bench.sh --version          # bench-standard version and the BetterBench it needs
betterbench --version         # 0.6.1 or newer
python3 --version             # 3.10+
```

If BetterBench is missing, install it (see the top). If an older one is installed, ask
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

`--quant` comes from the model table. For the rest, use the launch command and confirm
with your human:

| flag | llama.cpp | vLLM |
|---|---|---|
| `--kv` | `-ctk`/`-ctv`; `f16` when unset; `q8_0/q4_0` when K and V differ | `--kv-cache-dtype`; with `auto` (or unset), the model dtype (`bf16`) |
| `--tp` / `--pp` | `1` / `1` on one GPU; multi-GPU: ask | `--tensor-parallel-size` / `--pipeline-parallel-size`; default `1` |
| `--spec` | `-md`/`--model-draft` with `--draft-max N` → `draft:N`; otherwise `none` | `--speculative-config` method and `num_speculative_tokens` → e.g. `mtp:5`, `eagle3:4`, `ngram:4`; otherwise `none` |
| `--thinking` | `--reasoning-budget 0` or `enable_thinking: false` in the template kwargs → `off`; unclear → ask | `enable_thinking`/effort in the chat template or default kwargs; nothing set → `default`; unclear → ask |
| `--submitter` | ask | ask |

### 5. Dry run

```sh
./bench.sh --endpoint URL --model ID --tp 1 --pp 1 --quant Q --kv KV \
  --spec none --thinking default --submitter NAME [--server-cmd-file server-cmd.txt] --dry-run
```

This sends the warm-up and the greedy check, detects everything, prints what it would
run, and stops. Check the `--note` values in that output:
- `unknown` values should have a reason (VM PCIe, no cmd file, ...).
- Any `bench.sh: warning:` line needs fixing or explaining before the real run.

### 6. Quick run, then full run

Run the same command without `--dry-run`. This is a quick check of a few minutes,
ending in a summary. If it looks sane, ask your human, then run again with `--full`.
That is the result that counts. For long runs, use a terminal multiplexer or a
background job, so a dropped session doesn't kill the benchmark.

### 7. Report back

Tell your human:
- the result folder (printed after `done:`)
- the summary block: decode t/s, TTFT, concurrency, prefill, bandwidth %, power, energy
- every warning, and whether the greedy check was `identical`
- that the folder is what they send in, and that they should check `server-cmd.txt` for
  anything private first (tokens, keys and passwords are already redacted)

## When something goes wrong

| message | what to do |
|---|---|
| `BetterBench 0.6.1 or newer required` | install or upgrade BetterBench (step 1) |
| `warm-up request failed (HTTP ...)` | server not up, or wrong `--endpoint`/`--model` (step 2). Behind llama-swap the first load can take minutes; `WARM_TIMEOUT=1800` gives it longer |
| `missing mandatory value(s)` | pass the listed flags (step 4) |
| `GPU power limit not detected here` | ask for the limit, pass `--power-limit W` or `stock` |
| `--kv X but the launch command sets Y` | the declared `--kv` is wrong. Use what the command sets |
| `greedy_check=differs` | not a failure, but report it. It means nondeterministic kernels or a broken spec-decoding path |
| `mem_bw` has no measured figure | no torch in the server container. Set `BW_PYTHON` to a python with GPU torch, or install `clpeak`, or leave it |
| `launch=unknown (pass --server-cmd-file)` | step 3 |
| `betterbench failed (exit N)` | read `betterbench.log` in the `.failed` folder. Usually the server crashed or ran out of memory |
| `ReBAR ... but BetterBench reads ...` | BetterBench runs on another machine than the GPU. Use `--env-file` (below) |

**BetterBench on another machine than the GPU:** run
`./bench.sh collect-env --endpoint URL --model ID > env.txt` on the GPU machine, then
add `--env-file env.txt` on the BetterBench machine. Power isn't sampled then. Run
`./bench.sh power > power.csv` on the GPU machine during the run if your human wants it.

What each recorded value means, and how it's detected:
[`dev/DETAILS.md`](dev/DETAILS.md).
