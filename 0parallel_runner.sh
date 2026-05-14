#!/usr/bin/env bash
set -euo pipefail

cd /home/christhian/iwcmc_oran

#SCENARIO="scratch/scenario-hierarchical-xangai-UAV"
SCENARIO="scenario-simple-test"
OUTDIR="parallel-results"
JOBS=2
RUNS=4

#./ns3 build "$SCENARIO"

mkdir -p "$OUTDIR"

seq 1 "$RUNS" | xargs -P "$JOBS" -I{} bash -c '
  run={}
  rundir="'"$OUTDIR"'/run-$run"
  mkdir -p "$rundir"

  ./ns3 run "'"$SCENARIO"' \
    --simTime=3.0 \
    --RngSeed=1 \
    --RngRun=$run \
    --cwd="$rundir" \
    > "$rundir/stdout.log" 2> "$rundir/stderr.log"
'