"""model_hash.py must match the README's commands exactly."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_hash  # noqa: E402


def sh(cmd, cwd):
    return subprocess.run(["sh", "-c", cmd], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.split()[0]


def test_directory_hash_matches_sha256sum(tmp_path):
    # model_mtp sorts before model. in en_US but after it in C: the README pins C
    (tmp_path / "model.safetensors").write_bytes(b"weights" * 1000)
    (tmp_path / "model_mtp.safetensors").write_bytes(b"mtp head")
    (tmp_path / "config.json").write_text("{}")
    want = sh("sha256sum *.safetensors | LC_ALL=C sort -k2 | sha256sum", tmp_path)
    assert model_hash.model_sha256(str(tmp_path), cache={}) == want


def test_file_hash_and_cache(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"gguf" * 5000)
    cache = {}
    assert model_hash.model_sha256(str(f), cache) == sh("sha256sum m.gguf", tmp_path)
    key = next(iter(cache))
    cache[key] = "cached"
    assert model_hash.model_sha256(str(f), cache) == "cached"   # same size and mtime: no rehash
