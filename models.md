# Models

**TBD by the group.** Fill in exact checkpoints; nobody benchmarks a file whose hash
doesn't match this table.

| model | format | HF repo | file(s) | SHA256 |
|---|---|---|---|---|
| ~9B (TBD) | GGUF | `TBD` | `TBD.gguf` | `TBD` |
| ~9B (TBD) | safetensors | `TBD` | `model-*.safetensors` | `TBD` (see below) |
| Qwen3.8-27B | GGUF | `TBD` | `TBD.gguf` | `TBD` |
| Qwen3.8-27B | safetensors | `TBD` | `model-*.safetensors` | `TBD` (see below) |

Open: which quant(s) of Qwen3.8-27B are standard.

## Verify

Single file (GGUF):

```sh
sha256sum /path/to/model.gguf
```

Directory (safetensors) - one hash over the per-file hashes, in name order:

```sh
cd /path/to/model-dir && sha256sum *.safetensors | sort -k2 | sha256sum
```

The upstream hash of a single file on Hugging Face (the LFS etag is its SHA256):

```sh
curl -sI https://huggingface.co/REPO/resolve/main/FILE | grep -i x-linked-etag
```
