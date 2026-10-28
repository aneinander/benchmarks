# Submitting a result

Send the whole run directory `results/<timestamp>-<submitter>-<model>-<engine>-<gpu>/`:

| file | what |
|---|---|
| `results.json` | BetterBench results, all notes under `env.notes` |
| `results.html` | the standalone report |
| `meta.json` | detected + declared setup, the exact BetterBench command |
| `server-cmd.txt` | the server launch command (from `--server-cmd-file`, or from llama-swap) |

Only full runs count (`bench.sh --full`): the `run` note starts with `full`. Runs
without `--full` are quick smoke tests.

If `server-cmd.txt` is missing, pass `--server-cmd-file` next time, or add the launch
command by hand. Values of variables or flags named like `*TOKEN*`, `*KEY*`, `*SECRET*`,
`*PASSWORD*` are redacted automatically; check the file for anything else private
before sending.
