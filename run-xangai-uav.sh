#!/bin/bash

set -euo pipefail

SCENARIO="scenario-hierarchical-xangai-UAV"
RESULTS_DIR="${RESULTS_DIR:-base_ppo2}"
SIM_TIME="${SIM_TIME:-10.0}"
UAV_MOBILITY_MODE="${UAV_MOBILITY_MODE:-1}"
UAV_FLIGHT_PATTERN="${UAV_FLIGHT_PATTERN:-4}"
POSITION_ALLOCATOR="${POSITION_ALLOCATOR:-2}"
ENABLE_TRACES="${ENABLE_TRACES:-true}"
PATH_GYM_OK_METRICS="${PATH_GYM_OK_METRICS:-true}"
USE_SEMAPHORES="${USE_SEMAPHORES:-0}"
SCHEDULE_CONTROL_MESSAGES="${SCHEDULE_CONTROL_MESSAGES:-0}"
CONTROL_PAYLOAD_FILE="${CONTROL_PAYLOAD_FILE:-}"
CONTROL_FILENAME="${CONTROL_FILENAME:-hierarchical_actions.csv}"
EXTRA_ARGS="${*:-}"

echo "================================================="
echo "  Running Hierarchical Xangai UAV Scenario"
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

SCENARIO_ARGS="$SCENARIO \
--simTime=$SIM_TIME \
--uavMobilityMode=$UAV_MOBILITY_MODE \
--uavFlightPattern=$UAV_FLIGHT_PATTERN \
--positionAllocator=$POSITION_ALLOCATOR \
--enableTraces=$ENABLE_TRACES \
--pathGymOkMetrics=$PATH_GYM_OK_METRICS \
--useSemaphores=$USE_SEMAPHORES \
--scheduleControlMessages=$SCHEDULE_CONTROL_MESSAGES"

if [ -n "$STAGED_CONTROL_FILE" ]; then
    SCENARIO_ARGS="$SCENARIO_ARGS --controlFileName=$STAGED_CONTROL_FILE"
fi

if [ -n "$EXTRA_ARGS" ]; then
    SCENARIO_ARGS="$SCENARIO_ARGS $EXTRA_ARGS"
fi

if [ -f "./ns3" ]; then
    RUN_CMD=(./ns3 run "$SCENARIO_ARGS")
elif [ -f "./waf" ]; then
    RUN_CMD=(./waf --run "$SCENARIO_ARGS")
else
    echo "Error: Neither ./ns3 nor ./waf found in current directory."
    exit 1
fi

printf 'Executing:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'
printf '%s\n' "$SCENARIO_ARGS" > "$RESULTS_DIR/run_command.txt"

"${RUN_CMD[@]}" > "$RESULTS_DIR/terminal_output.log" 2>&1

echo "Extracting RLF metrics..."
echo "Timestamp,CellID,BadUEs" > "$RESULTS_DIR/rlf_metrics.csv"
grep '^RLF_DUMP,' "$RESULTS_DIR/terminal_output.log" | sed 's/RLF_DUMP,//' >> "$RESULTS_DIR/rlf_metrics.csv" || true

echo "Gathering output files..."
shopt -s nullglob
OUTPUT_PATTERNS=(
    bsState.txt
    path-gym-ok-*.txt
    cu-cp-cell-*.txt
    cu-up-cell-*.txt
    du-cell-*.txt
    *PdcpStats*.txt
    *RlcStats*.txt
    DlRlcRetx.txt
    DlPhyTransmissionTrace.txt
    EnbSchedAllocTraces.txt
    RxPacketTrace.txt
    *TxPhyStats.txt
    *RxPhyStats.txt
    *MacStats.txt
    mobility-trace.txt
    MmWaveSinrTime.txt
    LteDlRsrpSinrStats.txt
    LteUlSinrStats.txt
    LteUlInterferenceStats.txt
    UeFailures.txt
    *Handover*.txt
    CellIdStats*.txt
    *SwitchStats.txt
    RlcAmBufferSize.txt
    X2Stats.txt
    enbs.txt
    ues.txt
)

for pattern in "${OUTPUT_PATTERNS[@]}"; do
    for file in $pattern; do
        [ -e "$file" ] || continue
        mv "$file" "$RESULTS_DIR"/
    done
done
shopt -u nullglob

if [ -n "$STAGED_CONTROL_FILE" ] && [ -e "$STAGED_CONTROL_FILE" ]; then
    mv "$STAGED_CONTROL_FILE" "$RESULTS_DIR"/
fi

find "$RESULTS_DIR" -maxdepth 1 -type f -printf '%f\n' | sort > "$RESULTS_DIR/file_list.txt"

echo "================================================="
echo "  Run Complete! Check the '$RESULTS_DIR' folder."
echo "================================================="
