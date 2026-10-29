"""postrun.py: energy per phase and the bandwidth figure."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import postrun  # noqa: E402


def test_joules_between_interpolates():
    ts, cum = [0.0, 10.0, 20.0], [0.0, 1000.0, 3000.0]
    assert postrun.joules_between(ts, cum, 5, 15) == 1500.0
    assert postrun.joules_between(ts, cum, -5, 30) == 3000.0


def test_energy_curve_counter_or_integrated():
    rows = [(0, 100, 50.0, 0), (10, 100, 1050.0, 0)]
    assert postrun.energy_curve(rows)[1] == [0.0, 1000.0]
    rows = [(0, 100, None, 0), (10, 300, None, 0)]
    ts, cum, src = postrun.energy_curve(rows)
    assert cum == [0.0, 2000.0] and src == "integrated power"


def test_phase_energy_with_warmup_and_levels():
    # a steady 100 W, so energy = 100 J per second of each piece
    rows = [(float(t), 100.0, 100.0 * t, 2000.0) for t in range(0, 101)]
    timeline = [(0.0, "[single] chat: warmup 1 + 4 runs"),
                (10.0, "[prefill] depth ~2000 tok: warmup 2 + 8"),
                (20.0, "[prefill] skip depth ~64000: needs more"),
                (30.0, "[concurrency] level 2: 48 requests")]
    res = {"single_stream": {"chat": [{"ok": True, "completion_tokens": 100}] * 4},
           "prefill": [{"target_depth": 2000, "prompt_tokens": [1000] * 8}],
           "concurrency": [{"level": 2, "aggregate_tps": 10.0, "wall_s": 50.0}]}
    ph = postrun.phase_energy(res, rows, timeline, 80.0)
    # decode: 10 s = 1000 J; 400 measured tokens, x 5/4 for the warm-up pass
    assert ph["decode"] == {"joules": 1000, "avg_w": 100.0, "tokens_incl_warmup_est": 500,
                            "j_per_token": 2.0}
    # prefill runs until the concurrency line (the skip line starts nothing): 20 s
    assert ph["prefill"]["joules"] == 2000
    assert ph["prefill"]["j_per_1k_tokens"] == round(2000 / (8000 * 10 / 8) * 1000, 3)
    # level 2: 50 s = 5000 J for 500 tokens
    assert ph["concurrency"]["2"]["j_per_token"] == 10.0


def test_roofline_prefers_loaded_weights_and_scales_with_tp():
    single = [{"decode_tps": 100.0}]
    c = {"gpu_mem_bw_gbs": "500", "model_bytes": "4000000000", "model_loaded_gib": "1.0"}
    r = postrun.roofline(single, c, {"tp": "2"})
    assert r["weight_source"] == "loaded (engine log)"
    assert r["weight_bytes"] == 2 * 2**30
    assert r["mem_bw_gbs"] == 1000.0
    assert r["pct_of_mem_bw"] == round(100 * 100 * 2 * 2**30 / 1e9 / 1000, 1)


def test_roofline_falls_back_to_stored_and_declared_bw():
    r = postrun.roofline([{"decode_tps": 50.0}], {"model_bytes": "4000000000"},
                         {"tp": "1", "mem_bw": "400"})
    assert r["weight_source"] == "stored files" and r["pct_of_mem_bw"] == 50.0
    assert postrun.roofline([{"decode_tps": 50.0}], {}, {"tp": "1", "mem_bw": "400"}) is None
