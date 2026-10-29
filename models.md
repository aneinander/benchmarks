# Models

Benchmark only these files, and check the hash first. A different file is a different
benchmark.

| model | engine | Hugging Face repo | file(s) | `--quant` |
|---|---|---|---|---|
| Qwen3.5-9B | vLLM & co. | [`RedHatAI/Qwen3.5-9B-quantized.w4a16`](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16) | the repo (`model.safetensors`, `model_mtp.safetensors`) | `w4a16` |
| Qwen3.8-27B | vLLM & co. | [`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | the repo (`model.safetensors`, `model_mtp.safetensors`) | `w4a16` |
| Qwen3.5-9B | llama.cpp | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `Qwen3.5-9B-Q4_K_M.gguf` | `Q4_K_M` |
| Qwen3.8-27B | llama.cpp | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `UD-Q4_K_M` |

Both RedHatAI checkpoints are int4 weights with 16-bit activations (compressed-tensors),
and they include an MTP head (`model_mtp.safetensors`) for `--spec mtp:N`. The 27B GGUF
repo has no plain Q4_K_M, only Unsloth's dynamic `UD-Q4_K_M`.

## SHA256

| file | SHA256 |
|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| `RedHatAI/Qwen3.5-9B-quantized.w4a16` (directory) | `cca16f9855b9f92eee5876ddc8656ed6397316ffc65b9f17ca1939a9549e4c26` |
| `RedHatAI/Qwen3.8-27B-INT4` (directory) | `bffa6b23fa11be557fe9de4b122a8bb99f537ad7a1a4effa702f7d0c677778f7` |

The hashes are from the Hugging Face repos as of 2026-10-29 (revisions `3885219`,
`4ca7207`, `a398088`, `91bd022`). If a repo changes its files, this table gets updated.

## Check

A GGUF file:

```sh
sha256sum Qwen3.5-9B-Q4_K_M.gguf
```

A safetensors directory: one hash over the per-file hashes, in name order. `LC_ALL=C`
matters, because other locales sort `model_mtp` before `model.` and give a different hash.

```sh
cd /path/to/model-dir && sha256sum *.safetensors | LC_ALL=C sort -k2 | sha256sum
```
