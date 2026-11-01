#!/bin/sh
# gpu-power.sh - run ON THE INFERENCE SERVER. Samples the power draw and the
# energy counter and the core clock of all GPUs until killed, one CSV line
# per sample:
#   epoch_s,watts,joules,mhz
# watts is the sum over all GPUs; joules is the sum of the cards' cumulative
# energy counters (empty where the card has none, then integrate the watts);
# mhz is the mean graphics clock over the GPUs.
# Prints nothing at all when no supported tool is found (nvidia-smi, xpu-smi,
# or amdgpu's hwmon files in sysfs, which have no energy counter).
#
# usage: gpu-power.sh [INTERVAL_S]   (default 2)
#        GPU_POWER_SYSFS=DIR: the sysfs root for the amdgpu lookup (tests)

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
  # amdgpu: power1_average (older kernels) or power1_input, in microwatts;
  # freq1_input is the graphics clock in Hz.
  amd=""
  for h in "${GPU_POWER_SYSFS:-/sys}"/bus/pci/drivers/amdgpu/*:*/hwmon/hwmon*; do
    { [ -r "$h/power1_average" ] || [ -r "$h/power1_input" ]; } && amd="$amd $h"
  done
  [ -n "$amd" ] || exit 0
  sample() {
    for h in $amd; do
      p=$(cat "$h/power1_average" 2>/dev/null || cat "$h/power1_input" 2>/dev/null)
      f=$(cat "$h/freq1_input" 2>/dev/null)
      echo "${p:-0} ${f:-0}"
    done | awk '$1 > 0 {w += $1 / 1e6; c += $2 / 1e6; n++}
                END {if (n) printf "%.2f,,%.0f\n", w, c / n}'
  }
fi

while :; do
  t=$(date +%s.%N)
  s=$(sample)
  [ -n "$s" ] && printf '%s,%s\n' "$t" "$s"
  sleep "$interval"
done
