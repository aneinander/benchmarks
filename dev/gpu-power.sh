#!/bin/sh
# gpu-power.sh - run ON THE INFERENCE SERVER. Samples the power draw and the
# energy counter and the core clock of all GPUs until killed, one CSV line
# per sample:
#   epoch_s,watts,joules,mhz
# watts is the sum over all GPUs; joules is the sum of the cards' cumulative
# energy counters (empty where the card has none, then integrate the watts);
# mhz is the mean graphics clock over the GPUs.
# Prints nothing at all when no supported tool is found (nvidia-smi, xpu-smi).
#
# usage: gpu-power.sh [INTERVAL_S]   (default 2)

set -u
LC_ALL=C; export LC_ALL
interval=${1:-2}

if command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi --query-gpu=power.draw --format=csv,noheader >/dev/null 2>&1; then
  # total_energy_consumption is in mJ since the driver loaded
  sample() {
    nvidia-smi --query-gpu=power.draw,total_energy_consumption,clocks.gr \
               --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' '{w += $1; if ($2 + 0 > 0) j += $2 / 1000; else nj = 1; c += $3}
                   END {if (NR) printf "%.2f,%s,%.0f\n", w, nj ? "" : sprintf("%.3f", j), c / NR}'
  }
elif command -v xpu-smi >/dev/null 2>&1; then
  # one call takes about a second
  sample() {
    xpu-smi dump --device -1 --metrics power.draw,energy.consumed,clocks.current.graphics \
                 --number 1 2>/dev/null |
      awk -F', *' 'NR > 1 && $3 + 0 > 0 {w += $3; n++; if ($4 + 0 > 0) j += $4; else nj = 1; c += $5}
                   END {if (n) printf "%.2f,%s,%.0f\n", w, nj ? "" : sprintf("%.3f", j), c / n}'
  }
else
  exit 0
fi

while :; do
  t=$(date +%s.%N)
  s=$(sample)
  [ -n "$s" ] && printf '%s,%s\n' "$t" "$s"
  sleep "$interval"
done
