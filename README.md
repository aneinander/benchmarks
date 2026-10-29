# bench-standard

One script so everyone in the group benchmarks LLM servers the same way, with
[BetterBench](https://github.com/GGZ14/BetterBench). Every result records the setup it was
measured on: GPU, PCIe, engine, launch flags and more. Results stay comparable that way.

Using a coding agent? Point it at [`AGENTS.md`](AGENTS.md).

## 1. Install

On the machine with the GPU and the inference server (Linux, Python 3.10+):

```sh
git clone https://github.com/GGZ14/BetterBench && cd BetterBench
git checkout v0.6.1 && pip install -e . && cd ..
git clone https://git.linxweiler.xyz/miga/benchmarks bench-standard
```

## 2. Run

Serve one of the models in [`models.md`](models.md), Qwen3.5-9B or Qwen3.8-27B as
GGUF or w4a16, and check its SHA256. Any OpenAI-compatible server works (llama.cpp,
vLLM, ...). Then:

```sh
cd bench-standard
./bench.sh --endpoint http://localhost:8080/v1 --model MODEL_ID \
  --tp 1 --pp 1 --quant Q4_K_M --kv f16 --spec none --thinking default \
  --submitter YOUR_NAME --server-cmd-file server-cmd.txt
```

That's a quick check run of a few minutes. When it looks right, add `--full` for the
run that counts. It takes much longer and uses the GPU the whole time.

**You declare** what the script can't detect:

| flag | what | examples |
|---|---|---|
| `--tp` / `--pp` | tensor / pipeline parallel size | `1` / `1` |
| `--quant` | weight quantization | `Q4_K_M`, `gptq-int4`, `bf16` |
| `--kv` | KV cache type | `f16`, `q8_0` |
| `--spec` | speculative decoding | `none`, `mtp:5`, `eagle3:4`, `draft:8` |
| `--thinking` | thinking mode set **on the server** | `default`, `off`, `on`, `low`, `high` |
| `--submitter` | your name | `pascal` |

`--server-cmd-file` is a text file with the command you started the server with. Behind
llama-swap you can leave it out, because it's read automatically.

**Everything else is detected.** That includes the GPU, driver, power limit,
PCIe and ReBAR, measured memory bandwidth, CPU, OS, the engine and its exact
version, the model file, and launch and env flags. A temperature-0 output check
runs too. During the run it records power, clock and energy.

Optional flags:

| flag | for |
|---|---|
| `--full` | the real run (default is a quick check) |
| `--verbose` | show everything BetterBench prints |
| `--dry-run` | print what would run, then stop |
| `--mem-bw 608` | the card's spec memory bandwidth in GB/s |
| `--power-limit 250` | when the power limit can't be read |
| `--pcie "Gen4 x16"` | inside a VM, where the PCIe link can't be read |
| `--special-notes "..."` | anything unusual: custom kernels, undervolt, patched engine |
| `--api-key KEY` | if the server needs one (never stored) |

## 3. Send your result

Each run makes a folder `results/<date>-<you>-<model>-<engine>-<gpu>/`. Send the whole
folder. Open `results.html` in a browser to see the report.

- Only `--full` runs count.
- Check `server-cmd.txt` for anything private first. Tokens, keys and passwords
  are already redacted.

To compare two runs: `betterbench compare A/results.json B/results.json`

## Rules

- BetterBench **0.6.1** exactly, with the settings in `config.json`. `bench.sh` refuses
  any other version, and other BetterBench flags aren't allowed.
- Results come from `bench.sh` only. Don't edit them.

## More

- **BetterBench on another machine than the GPU:** run
  `./collect-env.sh --endpoint URL --model ID > env.txt` on the server, then add
  `--env-file env.txt` on the other machine.
- The declared values can also be set as `BENCH_<NAME>`, e.g. `BENCH_SUBMITTER=pascal`.
  Results go to `results/` (or `--results-dir`). Machine-local defaults like these
  go in `bench.local.env`.
- What each recorded field means and how it's detected: [`DETAILS.md`](DETAILS.md).
