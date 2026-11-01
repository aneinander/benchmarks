#!/bin/sh
# gpu-power.sh - run ON THE INFERENCE SERVER. Samples the power draw and the
# energy counter and the core and memory clocks of all GPUs until killed, one
# CSV line per sample:
#   epoch_s,watts,joules,mhz,mem_mhz
# watts is the sum over all GPUs; joules is the sum of the cards' cumulative
# energy counters (empty where the card has none, then integrate the watts);
# mhz and mem_mhz are the mean graphics and memory clocks over the GPUs
# (mem_mhz empty where the tool doesn't report it).
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
    nvidia-smi --query-gpu=power.draw,total_energy_consumption,clocks.gr,clocks.mem \
               --format=csv,noheader,nounits 2>/dev/null |
      awk -F', *' '{w += $1; if ($2 + 0 > 0) j += $2 / 1000; else nj = 1; c += $3; m += $4}
                   END {if (NR) printf "%.2f,%s,%.0f,%.0f\n", w, nj ? "" : sprintf("%.3f", j), c / NR, m / NR}'
  }
elif command -v xpu-smi >/dev/null 2>&1; then
  # One call takes about a second. Not every xpu-smi has the memory clock,
  # and an unknown metric fails the whole call: try it once, else go without.
  metrics=power.draw,energy.consumed,clocks.current.graphics
  xpu-smi dump --device -1 --metrics "$metrics,clocks.current.memory" --number 1 2>/dev/null |
    awk -F', *' 'NR > 1 && $6 + 0 > 0 {ok = 1} END {exit !ok}' && metrics="$metrics,clocks.current.memory"
  sample() {
    xpu-smi dump --device -1 --metrics "$metrics" --number 1 2>/dev/null |
      awk -F', *' 'NR > 1 && $3 + 0 > 0 {w += $3; n++; if ($4 + 0 > 0) j += $4; else nj = 1; c += $5
                                         if ($6 + 0 > 0) m += $6; else nm = 1}
                   END {if (n) printf "%.2f,%s,%.0f,%s\n", w, nj ? "" : sprintf("%.3f", j), c / n,
                                      nm ? "" : sprintf("%.0f", m / n)}'
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
