"""notes.py: launch parsing, the model check, the thinking check, the run note."""
import json
import os
import subprocess
import sys

DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_GGUF = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"

MANDATORY = {"submitter": "pascal", "tp": "1", "pp": "1", "quant": "UD-Q4_K_M", "kv": "q8_0",
             "spec": "none", "thinking": "default", "power_limit": "200",
             "power_limit_source": "detected", "betterbench": "0.6.1", "betterbench_build": "v0.6.1",
             "bench_standard": "0.4.1", "run_mode": "full", "concurrency": "on"}
COLLECTED = {"engine": "llama.cpp", "engine_version": "b11913-7aca0d2ea", "gpu_vendor": "intel",
             "gpu_count": "1", "gpu_model": "Arc Pro B65", "model_file": "/models/x.gguf",
             "model_bytes": "16464440224", "model_sha256": STD_GGUF}
MEASURED = {"greedy_check": "identical", "greedy_runs": "3", "greedy_variants": "1",
            "greedy_hash": "1d5105fe09dc", "greedy_reasoning": "yes"}
LLAMA_CMD = ("GGML_SYCL_DISABLE_OPT=0 ZES_ENABLE_SYSMAN=1 HF_HUB_OFFLINE=1 /app/llama-server "
             "-m /models/x.gguf -c 32768 -ngl 99 -fa on -b 4096 -ub 1024 -ctk q8_0 -ctv q8_0 "
             "-np 4 --no-mmap --port 9000")
VLLM_CMD = ("docker run --rm --name v -e K8V4_PREFILL=onednn -e HF_TOKEN=<redacted> img /model "
            "--max-num-batched-tokens=4224 --max-num-seqs=4 --kv-cache-dtype=int8_k_int4_v "
            "--no-enable-prefix-caching "
            "'--compilation-config={\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'")


def run(tmp_path, cmd=LLAMA_CMD, **over):
    files = {}
    for name, base in (("m", MANDATORY), ("c", COLLECTED), ("x", MEASURED)):
        d = {**base, **over.get(name, {})}
        files[name] = tmp_path / name
        files[name].write_text("".join(f"{k}={v}\n" for k, v in d.items()))
    (tmp_path / "cmd").write_text(cmd)
    p = subprocess.run([sys.executable, os.path.join(DEV, "notes.py"), files["m"], files["c"],
                        files["x"], tmp_path / "cmd", tmp_path / "launch.json",
                        os.path.join(DEV, "models.tsv")], capture_output=True, text=True, check=True)
    notes = dict(line.split("=", 1) for line in p.stdout.splitlines())
    return notes, p.stderr, json.loads((tmp_path / "launch.json").read_text())


def test_llama_launch_and_env(tmp_path):
    notes, err, launch = run(tmp_path)
    assert notes["launch"] == "b 4096 · ub 1024 · fa on · ctk q8_0 · ctv q8_0 · ngl 99 · np 4 · no-mmap"
    assert notes["env_flags"] == "GGML_SYCL_DISABLE_OPT=0 · ZES_ENABLE_SYSMAN=1"  # offline switch left out
    assert launch["env"]["HF_HUB_OFFLINE"] == "1"                                 # but kept in meta
    assert "warning" not in err


def test_llama_defaults_shown(tmp_path):
    notes, _, _ = run(tmp_path, cmd="llama-server -m /m.gguf -ngl 99 -fa", m={"kv": "f16"})
    assert notes["launch"] == "b default · ub default · fa on · ctk default · ctv default · ngl 99"


def test_vllm_launch_env_and_kv_warning(tmp_path):
    notes, err, _ = run(tmp_path, cmd=VLLM_CMD, c={"engine": "vllm"})
    assert "max-num-batched-tokens 4224" in notes["launch"]
    assert "no-enable-prefix-caching" in notes["launch"]
    assert 'compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}' in notes["launch"]
    assert notes["env_flags"] == "K8V4_PREFILL=onednn · HF_TOKEN=<redacted>"
    assert "--kv q8_0 but the launch command sets int8_k_int4_v" in err


def test_no_command(tmp_path):
    notes, _, _ = run(tmp_path, cmd="")
    assert notes["launch"] == "unknown (pass --server-cmd-file)"
    assert notes["env_flags"] == "unknown"


def test_standard_model(tmp_path):
    notes, err, _ = run(tmp_path)
    assert notes["model_file"].endswith("standard: Qwen3.8-27B UD-Q4_K_M")
    assert "standard models" not in err


def test_standard_model_wrong_quant(tmp_path):
    _, err, _ = run(tmp_path, m={"quant": "Q4_K_M"})
    assert "declare --quant UD-Q4_K_M" in err


def test_other_model(tmp_path):
    notes, err, _ = run(tmp_path, c={"model_sha256": "ab" * 32})
    assert "not a standard model (sha256 abababababab)" in notes["model_file"]
    assert "won't count" in err


def test_model_not_hashed(tmp_path):
    notes, _, _ = run(tmp_path, c={"model_sha256": ""})
    assert notes["model_file"].endswith("hash not checked")


def test_thinking_checked_against_replies(tmp_path):
    notes, err, _ = run(tmp_path, m={"thinking": "off"})
    assert notes["thinking"] == "off · reasoning seen"
    assert "--thinking off, but the greedy replies contain reasoning" in err
    notes, err, _ = run(tmp_path, m={"thinking": "high"}, x={"greedy_reasoning": "no"})
    assert "--thinking high, but the greedy replies contain no reasoning" in err
    _, err, _ = run(tmp_path, m={"thinking": "default"}, x={"greedy_reasoning": "no"})
    assert "thinking" not in err


def test_run_note_betterbench_build(tmp_path):
    notes, _, _ = run(tmp_path)
    assert notes["run"] == "full · concurrency on · bench-standard 0.4.1 · betterbench 0.6.1"
    notes, _, _ = run(tmp_path, m={"betterbench_build": "v0.6.1-3-gabc1234-dirty"})
    assert notes["run"].endswith("betterbench 0.6.1 @ v0.6.1-3-gabc1234-dirty")


def test_gpu_note_clocks(tmp_path):
    notes, _, _ = run(tmp_path, c={"gpu_vram_gb": "32", "gpu_max_core_mhz": "2800",
                                   "gpu_max_mem_mhz": "19000"})
    assert notes["gpu"] == "1x Intel Arc Pro B65 · 32 GB · max 2800 MHz core · 19000 MHz mem"


def test_max_ctx_vllm_from_engine_log(tmp_path):
    notes, _, _ = run(tmp_path, cmd=VLLM_CMD, m={"kv": "int8_k_int4_v"},
                      c={"engine": "vllm", "ctx": "262144", "kv_cache_tokens": "412416"})
    assert notes["max_ctx"] == "412k tokens (engine KV cache) · 262k set"


def test_max_ctx_llama_estimate(tmp_path):
    # 4 GiB free; q8_0 K and V, 16384 elements each per token = 34816 bytes
    c = {"ctx": "32768", "kv_elems_per_token": "16384 16384 8192 8192",
         "gpu_vram_total_mib": "32768", "gpu_vram_used_mib": "28672"}
    notes, _, _ = run(tmp_path, c=c)
    est = 32768 + 4 * 2**30 // 34816
    assert notes["max_ctx"] == f"≈ {est / 1000:.0f}k tokens est. · 33k set · 4.3 GB free · 34 KiB/token"
    # --swa-full keeps the sliding-window layers at full size too
    notes, _, _ = run(tmp_path, cmd=LLAMA_CMD + " --swa-full", c=c)
    assert notes["max_ctx"].endswith("51 KiB/token")


def test_max_ctx_left_out_without_data(tmp_path):
    notes, _, _ = run(tmp_path)
    assert "max_ctx" not in notes
