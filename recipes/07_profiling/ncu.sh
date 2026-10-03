#!/usr/bin/env bash
# Kernel-level profile with Nsight Compute: why is *this* kernel slow?
#
#   recipes/07_profiling/ncu.sh gemv_warp python recipes/03_int4_gemv/run.py --shape 14336x4096 --rows 1 --skip-naive
#   recipes/07_profiling/ncu.sh gemv_naive python recipes/03_int4_gemv/run.py --shape 4096x4096 --rows 1
#
# First arg is a regex on the kernel name. Profiles 3 launches after skipping 10 warm-up ones.
# Read, in order:
#   GPU Speed Of Light   -> memory % vs compute %: which wall is this kernel against?
#   Memory Workload      -> DRAM throughput vs recipe 01's number; L1/L2 hit rates; "sectors per request"
#                           (4 = coalesced for 4-byte loads; 32 = every lane hit its own line)
#   Occupancy / Launch   -> registers per thread, achieved occupancy, waves per SM
#   Warp State Stats     -> top stall reason (long scoreboard = waiting on memory; barrier; math pipe)
#   Source page          -> per-line stalls (kernels here build with -lineinfo)
# Needs GPU performance counters: run as root or with NVreg_RestrictProfilingToAdminUsers=0, and in Docker
# add --cap-add=SYS_ADMIN.
set -euo pipefail
[[ $# -gt 1 ]] || { sed -n '2,17p' "$0"; exit 1; }
kernel=$1; shift
out="results/profiles/ncu_${kernel//[^A-Za-z0-9_]/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p results/profiles
ncu --kernel-name "regex:$kernel" --launch-skip 10 --launch-count 3 --set full --import-source yes \
    --force-overwrite -o "$out" "$@"
ncu --import "$out.ncu-rep" --page details --section SpeedOfLight --section MemoryWorkloadAnalysis | head -80
echo "wrote $out.ncu-rep (open in the Nsight Compute GUI for the source view)"
