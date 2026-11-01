"""gguf_kv.py - KV cache elements per token of a GGUF model, from its header.

usage: gguf_kv.py FILE    (the first part of a split GGUF; prints nothing on failure)

Prints "K V SWA_K SWA_V": the K and V elements one token of context adds over
the layers whose cache grows with the context, and the same for sliding-window
layers, whose cache llama.cpp keeps at the window size (unless --swa-full).
Recurrent layers (Mamba, linear attention) keep a fixed state and count for
nothing. Bytes are elements x the size of the -ctk / -ctv type.
"""
import struct
import sys

# Sliding-window layouts llama.cpp sets in code when the GGUF doesn't say.
SWA_PATTERN = {"gemma2": 2, "gemma3": 6, "cohere2": 4, "gpt-oss": 2, "exaone4": 4}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_meta(path):
    """The key/value header, without the tensor infos. Big arrays (the
    tokenizer) are skipped, short ones kept."""
    with open(path, "rb") as f:
        def unpack(fmt):
            fmt = "<" + fmt
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(unpack("Q")).decode("utf-8", "replace")

        def value(t):
            if t in SCALAR:
                return unpack(SCALAR[t])
            if t == 8:
                return string()
            if t == 9:
                it, n = unpack("I"), unpack("Q")
                if it in SCALAR and n > 4096:
                    f.seek(n * struct.calcsize("<" + SCALAR[it]), 1)
                    return None
                items = [value(it) for _ in range(n)]
                return items if n <= 4096 else None
            raise ValueError(f"GGUF value type {t}")

        if f.read(4) != b"GGUF" or unpack("I") < 2:
            raise ValueError("not a GGUF v2+ file")
        unpack("Q")
        meta = {}
        for _ in range(unpack("Q")):
            k = string()
            meta[k] = value(unpack("I"))
        return meta


def kv_elems(meta):
    arch = meta["general.architecture"]
    a = lambda k, d=None: meta.get(f"{arch}.{k}", d)
    n_layer = a("block_count")
    per_layer = lambda v: v if isinstance(v, list) else [v] * n_layer
    n_head = per_layer(a("attention.head_count"))
    n_kv = per_layer(a("attention.head_count_kv", a("attention.head_count")))
    embd = a("embedding_length")
    k_len = a("attention.key_length") or embd // max(n_head)
    v_len = a("attention.value_length") or embd // max(n_head)
    if a("attention.kv_lora_rank") and not a("attention.key_length_mla"):
        # older MLA files (DeepSeek): one latent head of rank + rope dims
        rank = a("attention.kv_lora_rank")
        n_kv, k_len, v_len = [1] * n_layer, rank + a("rope.dimension_count", 0), rank
    interval = a("full_attention_interval")
    swa = [False] * n_layer
    if a("attention.sliding_window"):
        pat = a("attention.sliding_window_pattern", SWA_PATTERN.get(arch))
        if isinstance(pat, list):
            swa = [bool(x) for x in pat]
        elif pat:
            swa = [il % pat < pat - 1 for il in range(n_layer)]
    full, window = [0, 0], [0, 0]
    for il in range(n_layer):
        if not n_kv[il] or interval and (il + 1) % interval:
            continue
        d = window if swa[il] else full
        d[0] += n_kv[il] * k_len
        d[1] += n_kv[il] * v_len
    return full + window


if __name__ == "__main__":
    try:
        print(*kv_elems(read_meta(sys.argv[1])))
    except Exception:
        pass
