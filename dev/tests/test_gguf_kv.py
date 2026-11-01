"""gguf_kv.py: the header reader and the KV elements per token."""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf_kv  # noqa: E402


def write_gguf(path, meta):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def val(v):
        if isinstance(v, bool):
            return struct.pack("<I?", 7, v)
        if isinstance(v, int):
            return struct.pack("<II", 4, v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + s(v)
        if isinstance(v, list) and v and isinstance(v[0], str):
            return struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s(x) for x in v)
        if isinstance(v, list) and v and isinstance(v[0], bool):
            return struct.pack("<IIQ", 9, 7, len(v)) + b"".join(struct.pack("<?", x) for x in v)
        return struct.pack("<IIQ", 9, 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)

    body = b"".join(s(k) + val(v) for k, v in meta.items())
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta)) + body)
    return str(path)


def test_dense(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.block_count": 32, "llama.embedding_length": 4096,
        "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
        "tokenizer.ggml.tokens": ["a"] * 5000})          # a big array is skipped
    assert gguf_kv.kv_elems(gguf_kv.read_meta(p)) == [32 * 8 * 128, 32 * 8 * 128, 0, 0]


def test_hybrid_counts_only_attention_layers(tmp_path):
    # qwen3next-style: every 4th layer is full attention, the rest linear
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "qwen3next", "qwen3next.block_count": 8,
        "qwen3next.embedding_length": 2048, "qwen3next.attention.head_count": 16,
        "qwen3next.attention.head_count_kv": 2, "qwen3next.attention.key_length": 256,
        "qwen3next.attention.value_length": 256, "qwen3next.full_attention_interval": 4})
    assert gguf_kv.kv_elems(gguf_kv.read_meta(p)) == [2 * 2 * 256, 2 * 2 * 256, 0, 0]
    # per-layer KV heads with 0 for recurrent layers (lfm2, granite hybrid, ...)
    p = write_gguf(tmp_path / "h.gguf", {
        "general.architecture": "lfm2", "lfm2.block_count": 4, "lfm2.embedding_length": 1024,
        "lfm2.attention.head_count": 16, "lfm2.attention.head_count_kv": [0, 8, 0, 8]})
    assert gguf_kv.kv_elems(gguf_kv.read_meta(p)) == [2 * 8 * 64, 2 * 8 * 64, 0, 0]


def test_sliding_window_split(tmp_path):
    # gemma3: 5 of 6 layers are sliding-window (pattern set in code)
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "gemma3", "gemma3.block_count": 12, "gemma3.embedding_length": 2560,
        "gemma3.attention.head_count": 8, "gemma3.attention.head_count_kv": 4,
        "gemma3.attention.key_length": 256, "gemma3.attention.value_length": 256,
        "gemma3.attention.sliding_window": 1024})
    assert gguf_kv.kv_elems(gguf_kv.read_meta(p)) == [2 * 4 * 256, 2 * 4 * 256, 10 * 4 * 256, 10 * 4 * 256]


def test_not_gguf(tmp_path):
    (tmp_path / "x").write_bytes(b"nope")
    try:
        gguf_kv.read_meta(str(tmp_path / "x"))
    except ValueError:
        pass
    else:
        raise AssertionError
