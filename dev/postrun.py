"""postrun.py - figures that need the finished run, and the summary.

usage: T0=epoch T1=epoch postrun.py RUN_DIR

Reads results.json, meta.json, power.csv (epoch_s,watts,joules,mhz) and
timeline.tsv (epoch<TAB>BetterBench progress line) from RUN_DIR. Writes
energy (whole run and per phase) and the bandwidth figure to meta.json under
"post_run", and prints the summary to stderr.
"""
import bisect
import json
import os
import re
import statistics
import sys


def read_power(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            p = line.strip().split(",")
            try:
                rows.append((float(p[0]), float(p[1]), float(p[2]) if p[2] else None,
                             float(p[3]) if len(p) > 3 and p[3] else 0.0))
            except (ValueError, IndexError):
                pass
    return rows


def energy_curve(rows):
    """(times, cumulative joules): from the energy counter when every sample
    has one, else integrated from the power samples."""
    ts = [r[0] for r in rows]
    if all(r[2] is not None for r in rows) and rows[-1][2] >= rows[0][2]:
        return ts, [r[2] - rows[0][2] for r in rows], "energy counter"
    cum = [0.0]
    for a, b in zip(rows, rows[1:]):
        cum.append(cum[-1] + (b[0] - a[0]) * (a[1] + b[1]) / 2)
    return ts, cum, "integrated power"


def joules_between(ts, cum, t0, t1):
    """Energy between two moments, interpolating between samples."""
    def at(t):
        if t <= ts[0]:
            return cum[0]
        if t >= ts[-1]:
            return cum[-1]
        i = bisect.bisect_right(ts, t)
        f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        return cum[i - 1] + f * (cum[i] - cum[i - 1])
    return max(0.0, at(t1) - at(t0))


SEGMENT = re.compile(r"^\[(single|prefill|concurrency)\]\s+(.*)$")
WARMUP = re.compile(r"warmup (\d+) \+ (\d+)")


def segments(timeline, t_end):
    """BetterBench prints a progress line when each piece of work starts; a
    piece lasts until the next line. Yields (phase, key, warm-up factor, t0, t1)."""
    marks = []
    for t, line in timeline:
        m = SEGMENT.match(line)
        if not m or "skip" in m.group(2):
            continue
        phase, rest = m.groups()
        key = rest.split(":")[0].strip()
        if phase == "concurrency":
            key = re.sub(r"\D", "", key)
        w = WARMUP.search(rest)
        factor = (int(w.group(1)) + int(w.group(2))) / int(w.group(2)) if w and int(w.group(2)) else 1.0
        marks.append((t, phase, key, factor))
    for i, (t, phase, key, factor) in enumerate(marks):
        yield phase, key, factor, t, marks[i + 1][0] if i + 1 < len(marks) else t_end


def phase_energy(res, rows, timeline, t_end):
    """Joules and joules per token for decode, prefill and each concurrency
    level. Warm-up passes draw power but aren't in results.json, so their
    tokens are estimated from the measured passes of the same piece."""
    ts, cum, _ = energy_curve(rows)
    single = res.get("single_stream") or {}
    prefill = {str(p.get("target_depth")): p for p in res.get("prefill") or []}
    conc = {str(x.get("level")): x for x in res.get("concurrency") or []}
    out = {"decode": {"joules": 0.0, "seconds": 0.0, "tokens": 0.0},
           "prefill": {"joules": 0.0, "seconds": 0.0, "tokens": 0.0}, "concurrency": {}}
    for phase, key, factor, t0, t1 in segments(timeline, t_end):
        j = joules_between(ts, cum, t0, t1)
        if phase == "single":
            tok = sum(r.get("completion_tokens") or 0 for r in single.get(key, []) if r.get("ok", True))
            d = out["decode"]
        elif phase == "prefill":
            depth = re.sub(r"\D", "", key)
            tok = sum(prefill.get(depth, {}).get("prompt_tokens") or [])
            d = out["prefill"]
        else:
            x = conc.get(key, {})
            tok, factor = x.get("aggregate_tps", 0) * x.get("wall_s", 0), 1.0
            d = out["concurrency"].setdefault(key, {"joules": 0.0, "seconds": 0.0, "tokens": 0.0})
        d["joules"] += j
        d["seconds"] += t1 - t0
        d["tokens"] += tok * factor
    def finish(d, per=1):
        return {"joules": round(d["joules"]), "avg_w": round(d["joules"] / d["seconds"], 1) if d["seconds"] else None,
                "tokens_incl_warmup_est": round(d["tokens"]),
                "j_per_token" if per == 1 else "j_per_1k_tokens":
                    round(d["joules"] / d["tokens"] * per, 3) if d["tokens"] else None}
    result = {}
    if out["decode"]["seconds"]:
        result["decode"] = finish(out["decode"])
    if out["prefill"]["seconds"]:
        result["prefill"] = finish(out["prefill"], per=1000)
    if out["concurrency"]:
        result["concurrency"] = {k: finish(v) for k, v in out["concurrency"].items()}
    return result


def roofline(single, c, m):
    """Batch-1 decode reads all weights once per token: decode t/s x weight
    bytes is the bandwidth it reaches. The weights are what the engine loaded
    (vLLM logs it per worker) or else what is stored; tensor parallelism
    spreads them over tp GPUs, each with its own bandwidth."""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    tps = [r["decode_tps"] for r in single if r.get("decode_tps")]
    bw = num(c.get("gpu_mem_bw_gbs")) or num(m.get("mem_bw"))
    tp = int(m.get("tp") or 1)
    loaded, stored = num(c.get("model_loaded_gib")), num(c.get("model_bytes"))
    if loaded:
        nbytes, src = loaded * 2**30 * tp, "loaded (engine log)"
    elif stored:
        nbytes, src = stored, "stored files"
    else:
        return None
    if not tps or not bw:
        return None
    med = statistics.median(tps)
    eff = med * nbytes / 1e9
    return {"decode_tps_median": round(med, 2), "weight_bytes": int(nbytes), "weight_source": src,
            "mem_bw_gbs": bw * tp, "gpus": tp, "effective_gbs": round(eff, 1),
            "pct_of_mem_bw": round(100 * eff / (bw * tp), 1)}


def main(d):
    res = json.load(open(f"{d}/results.json", encoding="utf-8"))
    meta = json.load(open(f"{d}/meta.json", encoding="utf-8"))
    c, m = meta["collected"], meta["mandatory"]
    t0, t1 = float(os.environ["T0"]), float(os.environ["T1"])
    post = {"wall_s": round(t1 - t0, 1)}

    single = [r for recs in (res.get("single_stream") or {}).values() for r in recs
              if r.get("ok", True)]
    tok = sum(r.get("completion_tokens") or 0 for r in single)
    tok += sum(round(x.get("aggregate_tps", 0) * x.get("wall_s", 0))
               for x in res.get("concurrency") or [])

    rows = read_power(f"{d}/power.csv")
    timeline = []
    if os.path.exists(f"{d}/timeline.tsv"):
        for line in open(f"{d}/timeline.tsv", encoding="utf-8"):
            t, _, text = line.rstrip("\n").partition("\t")
            try:
                timeline.append((float(t), text))
            except ValueError:
                pass
    if len(rows) >= 2:
        ts, cum, src = energy_curve(rows)
        span = ts[-1] - ts[0]
        joules = cum[-1]
        e = {"joules": round(joules), "source": src, "sampled_s": round(span, 1),
             "avg_w": round(joules / span, 1) if span else None,
             "peak_w": max(r[1] for r in rows), "output_tokens": tok,
             "j_per_output_token": round(joules / tok, 3) if tok else None}
        # A clock-gated idle GPU reads 0 MHz; average the samples where it runs.
        clk = [r[3] for r in rows if r[3] > 0]
        if clk:
            e.update(avg_mhz=round(statistics.mean(clk)), min_mhz=round(min(clk)),
                     max_mhz=round(max(clk)), clock_samples=f"{len(clk)}/{len(rows)}")
        if timeline:
            e["phases"] = phase_energy(res, rows, timeline, t1)
        post["energy"] = e

    r = roofline(single, c, m)
    if r:
        post["roofline"] = r

    # The summary: headline numbers plus the setup that makes them comparable.
    def med(xs):
        xs = [x for x in xs if x]
        return statistics.median(xs) if xs else None
    def f(x, unit, dec=0):
        return "—" if x is None else f"{x:,.{dec}f} {unit}"
    notes = meta["notes"]
    tps = [x["decode_tps"] for x in single if x.get("decode_tps")]
    L = [f"  {meta['model']} · {notes.get('engine', '?')}",
         f"  {notes.get('gpu', '?')} · {notes.get('power_limit', '?')} limit",
         f"  {notes.get('run', '?')} · greedy {notes.get('greedy_check', '?')}",
         f"  decode (batch 1)  {f(med(tps), 't/s', 1)}    TTFT p50  "
         f"{f(med(x.get('ttft_ms') for x in single), 'ms')}"]
    conc = res.get("concurrency") or []
    if conc:
        L.append("  concurrency       " + "   ".join(
            f"{x['level']}: {x.get('aggregate_tps', 0):,.0f} t/s" for x in conc))
    live = [p for p in res.get("prefill") or [] if not p.get("skipped")]
    if live:
        L.append("  prefill           " + "   ".join(
            f"{p['target_depth'] // 1000}k: {f(med(p.get('pp_tps') or []), 't/s')}" for p in live))
    if r:
        L.append(f"  bandwidth         {r['effective_gbs']:.0f} GB/s effective = "
                 f"{r['pct_of_mem_bw']:.0f}% of {r['mem_bw_gbs']:.0f} GB/s "
                 f"(weights {r['weight_bytes'] / 1e9:.2f} GB, {r['weight_source']})")
    if "energy" in post:
        e = post["energy"]
        clock = (f" · clock avg {e['avg_mhz']} MHz ({e['min_mhz']}-{e['max_mhz']})"
                 if "avg_mhz" in e else "")
        L.append(f"  power             avg {e['avg_w']} W, peak {e['peak_w']:.0f} W{clock}")
        ph = e.get("phases") or {}
        parts = []
        if ph.get("decode", {}).get("j_per_token") is not None:
            parts.append(f"decode {ph['decode']['j_per_token']} J/tok at {ph['decode']['avg_w']} W")
        if ph.get("prefill", {}).get("j_per_1k_tokens") is not None:
            parts.append(f"prefill {ph['prefill']['j_per_1k_tokens']} J/1k tok")
        if parts:
            L.append("  energy            " + " · ".join(parts))
        if ph.get("concurrency"):
            L.append("  energy by conc.   " + "   ".join(
                f"{k}: {v['j_per_token']} J/tok" for k, v in ph["concurrency"].items()
                if v.get("j_per_token") is not None))
        L.append(f"  energy (total)    {e['joules'] / 1000:.1f} kJ, "
                 f"{e['j_per_output_token']} J per output token over the whole run")
    print("\n" + "\n".join(L), file=sys.stderr)

    meta["post_run"] = post
    with open(f"{d}/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
