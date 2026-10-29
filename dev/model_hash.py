"""model_hash.py - SHA256 of a model, as the README's Models section defines it.

usage: model_hash.py PATH    (prints the hash, or nothing when PATH can't be read)

A file: its SHA256, like `sha256sum FILE`. A directory: the SHA256 over the
per-file hashes of its *.safetensors, in byte order of the names, exactly like
`sha256sum *.safetensors | LC_ALL=C sort -k2 | sha256sum`.

Hashing a 16 GB file takes a while, so per-file hashes are cached in
~/.cache/bench-standard/sha256.tsv, keyed by path, size and mtime.
"""
import hashlib
import os
import sys

CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                     "bench-standard", "sha256.tsv")


def _load_cache():
    cache = {}
    try:
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 4:
                    cache[(p[0], p[1], p[2])] = p[3]
    except OSError:
        pass
    return cache


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        tmp = CACHE + f".{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            for (path, size, mtime), h in sorted(cache.items()):
                f.write(f"{path}\t{size}\t{mtime}\t{h}\n")
        os.replace(tmp, CACHE)
    except OSError:
        pass


def file_sha256(path, cache):
    st = os.stat(path)
    key = (os.path.realpath(path), str(st.st_size), str(st.st_mtime_ns))
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        cache[key] = h.hexdigest()
    return cache[key]


def model_sha256(path, cache=None):
    cache = _load_cache() if cache is None else cache
    if os.path.isdir(path):
        names = sorted((n for n in os.listdir(path) if n.endswith(".safetensors")),
                       key=lambda n: n.encode())
        if not names:
            return None
        listing = "".join(f"{file_sha256(os.path.join(path, n), cache)}  {n}\n" for n in names)
        return hashlib.sha256(listing.encode()).hexdigest()
    return file_sha256(path, cache)


if __name__ == "__main__":
    try:
        cache = _load_cache()
        h = model_sha256(sys.argv[1], cache)
        _save_cache(cache)
    except (OSError, IndexError):
        h = None
    if h:
        print(h)
