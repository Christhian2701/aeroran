#!/bin/bash

set -euo pipefail

SCENARIO="scenario-simple-test"
RESULTS_DIR="${RESULTS_DIR:-results_simple_test}"
SIM_TIME="${SIM_TIME:-10.0}"
USE_SEMAPHORES="${USE_SEMAPHORES:-0}"
SCHEDULE_CONTROL_MESSAGES="${SCHEDULE_CONTROL_MESSAGES:-0}"
ENABLE_SIMPLE_MOBILITY="${ENABLE_SIMPLE_MOBILITY:-0}"
CONTROL_PAYLOAD_FILE="${CONTROL_PAYLOAD_FILE:-}"
CONTROL_FILENAME="${CONTROL_FILENAME:-hierarchical_actions.csv}"
EXTRA_ARGS="${*:-}"

echo "================================================="
echo "  Running Lightweight O-RAN Metric Test"
echo "================================================="

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

STAGED_CONTROL_FILE=""
if [ -n "$CONTROL_PAYLOAD_FILE" ]; then
    if [ ! -f "$CONTROL_PAYLOAD_FILE" ]; then
        echo "Error: control payload file '$CONTROL_PAYLOAD_FILE' not found."
        exit 1
    fi

    STAGED_CONTROL_FILE="$CONTROL_FILENAME"
    cp "$CONTROL_PAYLOAD_FILE" "$RESULTS_DIR/control_payload_source.csv"
    cp "$CONTROL_PAYLOAD_FILE" "$STAGED_CONTROL_FILE"
fi

SCENARIO_ARGS="scratch/$SCENARIO \
--simTime=$SIM_TIME \
--useSemaphores=$USE_SEMAPHORES \
--scheduleControlMessages=$SCHEDULE_CONTROL_MESSAGES \
--enableSimpleMobility=$ENABLE_SIMPLE_MOBILITY"

if [ -n "$STAGED_CONTROL_FILE" ]; then
    SCENARIO_ARGS="$SCENARIO_ARGS --controlFileName=$STAGED_CONTROL_FILE"
fi

if [ -n "$EXTRA_ARGS" ]; then
    SCENARIO_ARGS="$SCENARIO_ARGS $EXTRA_ARGS"
fi

# Determine the build system (ns3 vs waf)
if [ -f "./ns3" ]; then
   RUN_CMD=(./ns3 run "$SCENARIO_ARGS")
elif [ -f "./waf" ]; then
    RUN_CMD=(./waf --run "$SCENARIO_ARGS")
else
    echo "Error: Neither ./ns3 nor ./waf found in current directory."
    exit 1
fi

# Run the simulation and capture standard output
printf 'Executing:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'
printf '%s\n' "$SCENARIO_ARGS" > "$RESULTS_DIR/run_command.txt"
"${RUN_CMD[@]}" > "$RESULTS_DIR/terminal_output.log" 2>&1

# Extract the RLF metrics from the terminal log into a clean CSV
echo "Extracting RLF metrics..."
echo "Timestamp,CellID,BadUEs" > "$RESULTS_DIR/rlf_metrics.csv"
grep '^RLF_DUMP,' "$RESULTS_DIR/terminal_output.log" | sed 's/RLF_DUMP,//g' >> "$RESULTS_DIR/rlf_metrics.csv" || true

# Move all the generated E2 KPMs and ns-3 traces to the results folder
echo "Gathering output files..."
shopt -s nullglob
OUTPUT_PATTERNS=(
    bsState.txt
    hierarchical_actions.csv
    cu-up*.txt
    cu-cp*.txt
    du-*.txt
    DlPdcpStats.txt
    UlPdcpStats.txt
    DlRlcStats.txt
    UlRlcStats.txt
    DlPhyTransmissionTrace.txt
    EnbSchedAllocTraces.txt
    *TxPhyStats.txt
    *RxPhyStats.txt
    *MacStats.txt
    LteDlRsrpSinrStats.txt
    LteUlSinrStats.txt
    LteUlInterferenceStats.txt
    UeFailures.txt
    mobility-trace.txt
)

for pattern in "${OUTPUT_PATTERNS[@]}"; do
    for file in $pattern; do
        [ -e "$file" ] || continue
        mv "$file" "$RESULTS_DIR"/
    done
done
shopt -u nullglob

find "$RESULTS_DIR" -maxdepth 1 -type f -printf '%f\n' | sort > "$RESULTS_DIR/file_list.txt"

echo "================================================="
echo "  Test Complete! Check the '$RESULTS_DIR' folder."
echo "================================================="
