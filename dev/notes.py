"""notes.py - the notes bench.sh hands BetterBench, one per topic.

usage: notes.py MANDATORY COLLECTED MEASURED CMD LAUNCH_JSON MODELS_TSV > notes
The first three are key=value files (declared, detected by collect-env.sh,
measured by the greedy check), CMD the redacted launch command (may be
empty). Writes the parsed launch settings to LAUNCH_JSON and prints
key=value notes; warnings go to stderr.
"""
import json, shlex, sys
def kv(path):
    with open(path, encoding="utf-8") as f:
        return dict(l.rstrip("\n").split("=", 1) for l in f if "=" in l)
m, c, meas = kv(sys.argv[1]), kv(sys.argv[2]), kv(sys.argv[3])
models = [l.rstrip("\n").split("\t") for l in open(sys.argv[6], encoding="utf-8")
          if l.strip() and not l.startswith("#")]
cmd = open(sys.argv[4], encoding="utf-8").read().strip()
def g(k):
    v = c.get(k, "")
    return "" if v in ("", "unknown") else v
def join(*parts):
    return " · ".join(p for p in parts if p) or "unknown"

vendor = {"intel": "Intel", "nvidia": "NVIDIA", "amd": "AMD"}.get(g("gpu_vendor"), g("gpu_vendor"))
count = g("gpu_count")
gpu = join(" ".join(p for p in (count and count + "x", vendor if vendor != "none" else "",
                                g("gpu_model")) if p),
           g("gpu_vram_gb") and g("gpu_vram_gb") + " GB",
           g("gpu_driver") and "driver " + g("gpu_driver"),
           g("gpu_max_core_mhz") and "max " + g("gpu_max_core_mhz") + " MHz core",
           g("gpu_max_mem_mhz") and g("gpu_max_mem_mhz") + " MHz mem")

link, top = g("pcie_link"), g("pcie_max")
if m.get("pcie"):
    pcie = m["pcie"] + " (declared)"
elif link and top and link != top:
    pcie = f"{link} now, {top} max"   # idle GPUs drop the link speed
else:
    pcie = link or "unknown"
if g("rebar"):
    pcie += " · ReBAR " + g("rebar")

pl, det = m["power_limit"], g("gpu_power_limit_w")
if pl == "stock":
    power = join("stock", det and det + " W")
else:
    power = pl + " W"
    if det and abs(float(det) - float(pl)) > 1:
        power += f" (GPU reports {det} W)"

virt = g("virt")
host = join(g("cpu_model"), g("ram_gb") and g("ram_gb") + " GB RAM", g("os"),
            g("kernel") and "kernel " + g("kernel"), virt != "none" and virt and "VM " + virt)

# Settings in the launch command that swing results but no API reports:
# batch sizes, flash attention, KV types, env flags. Flags left out of the
# command are the engine's defaults, which change between versions.
try:
    words = shlex.split(cmd.replace("\\\n", " "))
except ValueError:
    words = cmd.split()
env, flags, prog = {}, {}, None
BOOL = {"--no-mmap", "--mlock", "--enforce-eager", "--enable-prefix-caching",
        "--no-enable-prefix-caching", "--enable-chunked-prefill",
        "--no-enable-chunked-prefill", "--cont-batching", "-cb", "--no-cont-batching", "-nocb",
        "-cmoe", "--cpu-moe", "-rm", "--rm", "-d", "--detach", "-it", "-i", "--privileged"}
FA_VALUES = {"on", "off", "auto", "1", "0", "true", "false"}
i = 0
while i < len(words):
    w = words[i]
    nxt = words[i + 1] if i + 1 < len(words) else None
    if prog is None and "=" in w and not w.startswith("-") and w.split("=", 1)[0].isidentifier():
        k, v = w.split("=", 1); env[k] = v                     # FOO=1 cmd
    elif prog is None and not w.startswith("-") and w != "env":
        prog = w
    elif w in ("-e", "--env") and nxt is not None and not nxt.startswith("-"):  # docker -e K=V
        k, _, v = nxt.partition("="); env[k] = v if _ else "(inherited)"; i += 1
    elif w.startswith("--env="):
        k, _, v = w[6:].partition("="); env[k] = v if _ else "(inherited)"
    elif w.startswith("-e") and len(w) > 2 and "=" in w and w[2:].split("=")[0].isidentifier():
        k, v = w[2:].split("=", 1); env[k] = v
    elif w.startswith("-"):
        if "=" in w and w.startswith("--"):
            k, v = w.split("=", 1); flags[k] = v
        elif w in ("-fa", "--flash-attn"):
            if nxt is not None and nxt.lower() in FA_VALUES:
                flags[w] = nxt; i += 1
            else:
                flags[w] = "on"                                   # older bare -fa
        elif w in BOOL or nxt is None or nxt.startswith("-"):
            flags[w] = True
        else:
            flags[w] = nxt; i += 1
    i += 1

def flag(*names):
    for n in names:
        if n in flags:
            return flags[n]
    return None

LLAMA = [("b", ("-b", "--batch-size")), ("ub", ("-ub", "--ubatch-size")),
         ("fa", ("-fa", "--flash-attn")), ("ctk", ("-ctk", "--cache-type-k")),
         ("ctv", ("-ctv", "--cache-type-v")), ("ngl", ("-ngl", "--gpu-layers", "--n-gpu-layers")),
         ("np", ("-np", "--parallel")), ("t", ("-t", "--threads")),
         ("sm", ("-sm", "--split-mode")), ("ts", ("-ts", "--tensor-split")),
         ("ot", ("-ot", "--override-tensor")), ("ncmoe", ("-ncmoe", "--n-cpu-moe")),
         ("cmoe", ("-cmoe", "--cpu-moe")), ("no-mmap", ("--no-mmap",)), ("mlock", ("--mlock",)),
         ("nocb", ("-nocb", "--no-cont-batching"))]
LLAMA_ALWAYS = {"b", "ub", "fa", "ctk", "ctv"}   # shown as "default" when not set
VLLM = ["--max-num-batched-tokens", "--max-num-seqs", "--kv-cache-dtype",
        "--gpu-memory-utilization", "--dtype", "--quantization", "--block-size",
        "--enforce-eager", "--enable-chunked-prefill", "--no-enable-chunked-prefill",
        "--enable-prefix-caching", "--no-enable-prefix-caching", "--attention-backend",
        "--compilation-config", "-O", "--speculative-config"]
launch_kind = g("engine")
if launch_kind not in ("llama.cpp", "vllm"):
    launch_kind = ("llama.cpp" if prog and "llama-server" in prog or flag("-ngl", "-ub", "-ctk")
                   else "vllm" if any(f in flags for f in VLLM) else "")
settings = {}
if launch_kind == "llama.cpp":
    for name, al in LLAMA:
        v = flag(*al)
        if v is not None:
            settings[name] = "on" if v is True else v
        elif name in LLAMA_ALWAYS:
            settings[name] = "default"
elif launch_kind == "vllm":
    for f in VLLM:
        if f in flags:
            settings[f.lstrip("-")] = "on" if flags[f] is True else "".join(str(flags[f]).split())
if not cmd:
    launch = "unknown (pass --server-cmd-file)"
else:
    launch = " · ".join(k if v == "on" and k in ("no-mmap", "mlock", "nocb", "cmoe") or
                        k.startswith(("enable-", "no-enable-", "enforce-")) else f"{k} {v}"
                        for k, v in settings.items()) or "no tuning flags"
# Offline switches and cache paths don't change the numbers.
QUIET = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HOME",
         "VLLM_CACHE_ROOT", "PATH", "HOME", "TZ", "LANG", "LC_ALL"}
env_flags = (" · ".join(f"{k}={v}" for k, v in env.items() if k not in QUIET) or "none"
             ) if cmd else "unknown"
json.dump({"engine": launch_kind or None, "settings": settings, "env": env, "flags": flags},
          open(sys.argv[5], "w", encoding="utf-8"), indent=2, default=str)

# The declared --kv should be what the command sets.
kvset = [x for x in (settings.get("ctk"), settings.get("ctv"), settings.get("kv-cache-dtype"))
         if x and x not in ("default", "auto")]
if kvset and m["kv"] not in kvset + ["/".join(kvset), "_".join(kvset)]:
    print(f"bench.sh: warning: --kv {m['kv']} but the launch command sets "
          f"{'/'.join(kvset)}", file=sys.stderr)

# The model file against the standard models: results only count for those.
mb, sha = g("model_bytes"), g("model_sha256")
std = next((r for r in models if len(r) >= 4 and r[3] == sha), None) if sha else None
if std:
    check = f"standard: {std[0]} {std[2]}"
    if m["quant"] != std[2]:
        print(f"bench.sh: warning: --quant {m['quant']}, but the file is the standard "
              f"{std[0]} {std[2]}; declare --quant {std[2]}", file=sys.stderr)
elif sha:
    check = f"not a standard model (sha256 {sha[:12]})"
    print("bench.sh: warning: the model file is not one of the standard models "
          "(README, Models), so the result won't count", file=sys.stderr)
else:
    check = "hash not checked"
model_file = join(g("model_file"), mb and f"{int(mb) / 1e9:.2f} GB", check)
mem_bw = join(g("gpu_mem_bw_gbs") and
              f'{g("gpu_mem_bw_gbs")} GB/s measured ({g("gpu_mem_bw_tool") or "?"})',
              m.get("mem_bw") and m["mem_bw"] + " GB/s declared")
# The longest context this configuration holds. vLLM logs its KV cache in
# tokens. llama.cpp allocates the cache for -c up front, so estimate: what's
# set, plus the VRAM still free over the KV bytes per token of -ctk/-ctv.
# Sliding-window layers keep a fixed cache unless --swa-full.
KV_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
def kilo(n):
    return f"{n / 1000:.0f}k" if n >= 10000 else f"{n:,}"
ctx = int(g("ctx")) if g("ctx").isdigit() else None
max_ctx = None
if g("kv_cache_tokens").isdigit():
    max_ctx = join(kilo(int(g("kv_cache_tokens"))) + " tokens (engine KV cache)", ctx and kilo(ctx) + " set")
elif g("kv_elems_per_token") and ctx and g("gpu_vram_used_mib").isdigit() and g("gpu_vram_total_mib").isdigit():
    ek, ev, sk, sv = (int(x) for x in g("kv_elems_per_token").split())
    if "--swa-full" in flags:
        ek, ev = ek + sk, ev + sv
    bk = KV_BYTES.get(str(settings.get("ctk", "f16")).replace("default", "f16"))
    bv = KV_BYTES.get(str(settings.get("ctv", "f16")).replace("default", "f16"))
    free = (int(g("gpu_vram_total_mib")) - int(g("gpu_vram_used_mib"))) * 2**20
    if bk and bv and ek + ev:
        per = ek * bk + ev * bv
        max_ctx = join("≈ " + kilo(ctx + int(max(free, 0) / per)) + " tokens est.", kilo(ctx) + " set",
                       f"{free / 1e9:.1f} GB free", f"{per / 1024:.0f} KiB/token")

if meas.get("greedy_check") == "identical":
    greedy = f'identical {meas["greedy_runs"]}/{meas["greedy_runs"]} · {meas["greedy_hash"]}'
elif meas.get("greedy_check") == "differs":
    greedy = f'differs: {meas["greedy_variants"]} variants in {meas["greedy_runs"]} · first {meas["greedy_hash"]}'
else:
    greedy = "failed"

# --thinking is declared, but the greedy replies show whether the model
# reasoned: check one against the other.
seen = meas.get("greedy_reasoning")
thinking = join(m["thinking"], {"yes": "reasoning seen", "no": "no reasoning seen",
                                "mixed": "reasoning in some replies"}.get(seen, ""))
if m["thinking"] == "off" and seen in ("yes", "mixed"):
    print("bench.sh: warning: --thinking off, but the greedy replies contain reasoning;"
          " check the server's chat template or reasoning flags", file=sys.stderr)
elif m["thinking"] in ("on", "low", "medium", "high") and seen == "no":
    print(f"bench.sh: warning: --thinking {m['thinking']}, but the greedy replies contain no"
          " reasoning; thinking may be off on the server", file=sys.stderr)

# The BetterBench build: a release tag says nothing more than the version, a
# commit or local changes do (two installs can share a version number).
build = m.get("betterbench_build", "")
betterbench = "betterbench " + m["betterbench"]
if build and build not in ("v" + m["betterbench"], m["betterbench"]):
    betterbench += " @ " + build

eng = g("engine")
engine = join(" ".join(p for p in (eng, g("engine_version")) if p),
              g("engine_slots") and g("engine_slots") + " slots",
              g("engine_image") and "image " + g("engine_image"),
              g("proxy") not in ("", "none") and "via " + g("proxy"))

notes = [
    ("submitter", m["submitter"]),
    ("gpu", gpu),
    ("pcie", pcie),
    ("power_limit", power),
    ("host", host),
    ("engine", engine),
    ("model_file", model_file),
    ("mem_bw", mem_bw),
    ("parallel", f"tp {m['tp']} · pp {m['pp']}"),
    ("precision", f"weights {m['quant']} · kv {m['kv']}"),
    ("max_ctx", max_ctx),
    ("launch", launch),
    ("env_flags", env_flags),
    ("spec", m["spec"]),
    ("thinking", thinking),
    ("greedy_check", greedy),
    ("run", join(m["run_mode"], "concurrency " + m["concurrency"],
                 "bench-standard " + m["bench_standard"], betterbench)),
]
if m.get("special_notes"):
    notes.append(("special_notes", m["special_notes"]))
for k, v in notes:
    if v:
        print(f"{k}={v}")
