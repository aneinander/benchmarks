#!/usr/bin/env python3
"""Tidy the header of BetterBench's results.html, in place, after a run.

  report-header.py RUN_DIR

Fewer badges, nothing measured changes: the model badge takes over the model_file details;
PCIe, ReBAR and the power limit join the gpu badge; corpus, passes, temperature and prefix
cache join the run badge; the endpoint badge goes; launch and env flags become a dropdown
holding the real launch command (server-cmd.txt). A warning badge that is not one of the
usual values stays on its own. Everything outside the header is left byte for byte, and
results.json and meta.json keep every value, including the model hash.

BENCH_MODEL_LIBRARY (default /mnt/model-library, laid out as ORG/MODEL) is where the org in
the model badge comes from when the model is a local folder.
"""
import glob
import html as htmllib
import os
import re
import sys

DOT = " · "
FLAG_BADGES = {"launch": ("Launch flags", " "), "env_flags": ("Environment flags", "=")}
CHIP_END = "</span>"
RUN_BADGES = r"corpus v\S+|\d+ passes/cat|temp \S+|greedy|(cold|warm) prefix cache( \(nonce\))?"
CSS = """
  .flags { background:var(--surface); border:1px solid var(--border); border-radius:10px; font-size:13px; min-width:0; }
  .flags summary { cursor:pointer; padding:9px 14px; color:var(--ink-2); font-family:var(--mono); font-size:12px; }
  .flags-body { border-top:1px solid var(--border); padding:2px 14px 13px; }
  .flags pre { margin:12px 0 0; font-family:var(--mono); font-size:12px; line-height:1.6; color:var(--ink); overflow-x:auto; }
  .flags h2 { all:unset; display:block; font-family:var(--mono); font-size:10.5px; letter-spacing:.1em;
              text-transform:uppercase; color:var(--muted); margin:12px 0 6px; }
  .flags dl { display:grid; grid-template-columns:max-content minmax(0,1fr); gap:3px 14px; margin:0;
              font-family:var(--mono); font-size:12px; }
  .flags dt { color:var(--muted); }
  .flags dd { margin:0; color:var(--ink); overflow-wrap:anywhere; }
  @media (max-width:560px) { .flags dl { grid-template-columns:minmax(0,1fr); gap:0; } .flags dd { margin-bottom:6px; } }
"""


def tokens(cmd):
    """Shell words with their quotes kept, so the printed command can be pasted as is."""
    return re.findall(r"(?:'[^']*'|\"(?:[^\"\\]|\\.)*\"|[^\s'\"])+", cmd)


def show_command(cmd):
    """One option per line. Anything that is not one plain command (a script, a compose file) is shown as written."""
    one = re.sub(r"\\\r?\n", " ", cmd).strip()
    squeeze = lambda s: re.sub(r"\s", "", s)
    if "\n" in one or squeeze("".join(tokens(one))) != squeeze(one):
        return cmd.strip()
    lines = []
    for tok in tokens(one):
        flag = tok.lstrip("'\"").startswith("-")
        if not lines or flag:
            lines.append([tok])
        elif not lines[-1][0].lstrip("'\"").startswith("-"):  # the leading words (docker run), then image and path
            lines[-1].append(tok) if len(lines) == 1 else lines.append([tok])
        elif len(lines[-1]) == 1 and "=" not in lines[-1][0]:  # the value of a "--flag value" pair
            lines[-1].append(tok)
        else:
            lines.append([tok])
    return " \\\n".join(("  " if i else "") + " ".join(line) for i, line in enumerate(lines))


def flag_section(heading, value, sep):
    rows = ""
    for piece in value.split(DOT):
        key, _, val = piece.partition(sep)
        rows += f"<dt>{key}</dt><dd>{val or 'on'}</dd>"
    return f"<h2>{heading}</h2><dl>{rows}</dl>"


def model_org(name):
    """'Org/' from the model library. A local parent folder is not an org (it can be any folder)."""
    library = os.environ.get("BENCH_MODEL_LIBRARY", "/mnt/model-library")
    hits = glob.glob(os.path.join(glob.escape(library), "*", glob.escape(name))) if name else []
    return os.path.basename(os.path.dirname(hits[0])) + "/" if len(hits) == 1 else ""


def model_badge(model_file, cmd, fallback):
    """'Model: Org/name · size'. The location goes; a path inside a container is resolved through its mount."""
    pieces = model_file.split(DOT)
    where = "" if re.fullmatch(r"[\d.]+ GB|sha256 .*", pieces[0]) else pieces.pop(0)
    size = [p for p in pieces if p.endswith(" GB")]
    if where.startswith(("/", ".", "~")):
        mount = re.search(r"(?:-v|--volume)[\s=]+([^\s:]+):" + re.escape(where) + r"(?::\w+)?(?:\s|$)", cmd)
        name = htmllib.escape(os.path.basename((mount.group(1) if mount else where).rstrip("/")), quote=False)
        label = model_org(name) + name
    else:  # a Hugging Face repo id already says org/name
        label = where
    return "Model: " + DOT.join([label or fallback] + size)


def build(html, cmd):
    m = re.search(r'<div class="runbar">(.*?)</div>(?=\s*</header>)', html, re.S)
    if not m:
        sys.exit("badge row not found")
    chips = re.findall(r'<span class="chip">.*?</span>(?=<span class="chip">|$)', m.group(1), re.S)
    if len(chips) != m.group(1).count('class="chip"'):
        sys.exit("could not parse every badge")

    keep, flags, gpu_extra, run_extra, model_chip = [], {}, {}, [], None
    text_of = lambda chip: re.sub(r"<[^>]+>", "", chip).strip()
    for chip in chips:
        text = text_of(chip)
        key, sep, value = text.partition(": ")
        if key in FLAG_BADGES and sep:
            flags[key] = value
        elif key in ("pcie", "power_limit") and sep:
            gpu_extra[key] = value
        elif key == "model_file" and sep:
            model_chip = chip
        elif "<code>http" in chip:  # the endpoint: an address, of no use to a reader
            continue
        elif re.fullmatch(RUN_BADGES, text):
            run_extra.append(text)
        else:
            keep.append(chip)

    def add_to(prefix, extra):
        hits = [i for i, c in enumerate(keep) if text_of(c).startswith(prefix)]
        if len(hits) != 1:
            sys.exit(f"no single '{prefix}' badge to add {extra} to")
        keep[hits[0]] = keep[hits[0]][:-len(CHIP_END)] + DOT + DOT.join(extra) + CHIP_END

    if run_extra:
        add_to("run: ", run_extra)
    extra = ([f"PCIe {gpu_extra['pcie']}"] if "pcie" in gpu_extra else []) + \
            ([f"{gpu_extra['power_limit']} limit"] if "power_limit" in gpu_extra else [])
    if extra:
        add_to("gpu: ", extra)

    # the first badge repeats the page title; show the model file details there instead
    if model_chip:
        title = re.search(r"<h1>(.*?)</h1>", html, re.S)
        title = title.group(1) if title else None
        if keep and text_of(keep[0]) == title:
            badge = model_badge(text_of(model_chip).partition(": ")[2], cmd, title)
            keep[0] = keep[0][:-len(title + CHIP_END)] + badge + CHIP_END
        else:
            keep.insert(0, model_chip)

    new = f'<div class="runbar">{"".join(keep)}</div>'
    if cmd.strip():
        new += ('\n    <details class="flags"><summary>Deploy command</summary><div class="flags-body">'
                f'<pre>{htmllib.escape(show_command(cmd), quote=False)}</pre></div></details>')
    elif flags:
        body = "".join(flag_section(FLAG_BADGES[k][0], flags[k], FLAG_BADGES[k][1]) for k in FLAG_BADGES if k in flags)
        new += ('\n    <details class="flags"><summary>Launch and environment flags</summary>'
                f'<div class="flags-body">{body}</div></details>')
    out = html[:m.start()] + new + html[m.end():]
    return out.replace("</style>", CSS + "</style>", 1)


if __name__ == "__main__":
    run_dir = sys.argv[1]
    report = os.path.join(run_dir, "results.html")
    cmd_path = os.path.join(run_dir, "server-cmd.txt")
    with open(report, encoding="utf-8") as f:
        original = f.read()
    cmd = ""
    if os.path.exists(cmd_path):
        with open(cmd_path, encoding="utf-8", errors="replace") as f:
            cmd = f.read()
    result = build(original, cmd)
    with open(report + ".tmp", "w", encoding="utf-8") as f:
        f.write(result)
    os.replace(report + ".tmp", report)
