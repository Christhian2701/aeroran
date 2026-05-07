# Simple Test Scenario Summary

This document explains what `scratch/scenario-simple-test.cc` does, which metrics it generates, and what the resulting files look like.

## 1. Purpose of the Scenario

`scenario-simple-test.cc` is a compact ns-3 / mmWave / LTE multi-connectivity scenario meant to validate a `100 ms` metric extraction pipeline for offline O-RAN / DRL experiments.

The scenario is intentionally small:

- `1` LTE anchor cell
- `2` mmWave gNB cells
- `2` UEs
- `1` remote host generating downlink UDP traffic

It is not meant to be a realistic city-scale benchmark. Its main goal is to verify that:

- E2-style KPM files are generated every `100 ms`
- standard ns-3 PHY / RLC / PDCP traces are generated
- base-station state logging works
- a simple mobility pattern can trigger RLF-style low-SINR events

## 2. Topology and Traffic

### Radio topology

- LTE eNB is placed at `(50, 50, 3)`
- mmWave gNB 1 is colocated with the LTE anchor at `(50, 50, 3)`
- mmWave gNB 2 is placed at `(150, 50, 3)`

### UE motion

The UEs use `ConstantVelocityMobilityModel`:

- UE 0 starts at `(60, 50, 1.5)` and moves with velocity `(260, 0, 0)`
- UE 1 starts at `(140, 50, 1.5)` and moves with velocity `(220, 35, 0)`

This is a simple deterministic motion pattern that pushes the UEs away from the initial coverage region and helps create low-SINR conditions.

### Traffic

- Downlink UDP traffic is generated from the remote host to each UE
- packet size is `1024` bytes
- rate is `2 Mbps` per UE
- traffic starts at `0.2 s`

## 3. Control and Reporting Configuration

The scenario forces several important defaults before device installation:

- E2 indication periodicity: `100 ms`
- bearer stats epoch duration: `100 ms`
- mmWave ideal RRC: enabled
- HARQ: enabled
- `MmWavePhyMacCommon::NumHarqProcess = 100`
- `ThreeGppChannelModel::UpdatePeriod = 100 ms`
- `ThreeGppChannelConditionModel::UpdatePeriod = 100 ms`

These settings are important because earlier, more minimal settings caused the simulation to spend excessive wall-clock time very early in `Simulator::Run()`.

## 4. Custom Metrics Added by the Scenario

The scenario adds two custom metric mechanisms on top of standard ns-3 traces.

### 4.1 RLF tracker

The code maintains a map of UEs with bad SINR per cell. Every `100 ms`, it prints:

```text
RLF_DUMP,<time>,<cellId>,<badUeCount>
```

Two trace sources feed this counter:

- LTE UE PHY current-cell SINR trace
- mmWave SINR notification trace from `LteEnbRrc::NotifyMmWaveSinr`

The threshold is:

- count the UE if `SINR < -5 dB`

### 4.2 BS state tracker

The scenario periodically writes the state of secondary cells to `bsState.txt` every `100 ms`.

Columns:

- `Timestamp`: simulation time in seconds
- `UNIX`: absolute timestamp in milliseconds
- `Id`: cell ID
- `State`: `1` means active / allowed, `0` means inactive / not allowed

## 5. Standard Output Files Generated

The scenario generates a mix of custom files, E2-style KPM files, and standard ns-3 trace files.

### Custom / wrapper-oriented files

- `bsState.txt`
- stdout lines containing `RLF_DUMP,...`
- `results_simple_test/rlf_metrics.csv` when `run-simple-test.sh` is used

### E2-style KPM files

- `cu-cp-cell-1.txt`
- `cu-cp-cell-2.txt`
- `cu-cp-cell-3.txt`
- `cu-up-cell-1.txt`
- `cu-up-cell-2.txt`
- `cu-up-cell-3.txt`
- `du-cell-2.txt`
- `du-cell-3.txt`

### Standard ns-3 trace/stat files

- `DlPdcpStats.txt`
- `DlRlcStats.txt`
- `DlTxPhyStats.txt`
- `UlTxPhyStats.txt`
- `DlPhyTransmissionTrace.txt`
- `EnbSchedAllocTraces.txt`
- `UeFailures.txt`

## 6. What Each File Represents

### `bsState.txt`

Tracks whether the mmWave secondary cells are active.

Example:

```text
Timestamp UNIX Id State
0.1 1778085709041 2 1
0.1 1778085709041 3 1
0.2 1778085709141 2 1
0.2 1778085709141 3 1
```

Interpretation:

- at `0.1 s`, cells `2` and `3` were active
- the file is sampled every `100 ms`

### `rlf_metrics.csv`

Produced by `run-simple-test.sh` by filtering stdout `RLF_DUMP` lines.

Example:

```text
Timestamp,CellID,BadUEs
0.5,2,2
0.6,2,1
0.6,3,1
0.7,2,1
```

Interpretation:

- at `0.5 s`, cell `2` had `2` UEs below the outage threshold
- at `0.6 s`, cell `2` had `1` bad UE and cell `3` had `1` bad UE

### `cu-cp-cell-1.txt`

CU-CP style control-plane KPMs for the LTE anchor cell.

Example:

```text
timestamp,ueImsiComplete,numActiveUes,RRC.ConnMean,DRB.EstabSucc.5QI.UEID (numDrb),DRB.RelActNbr.5QI.UEID (0),eNB id,sameCellSinr,sameCellSinr 3gpp encoded
1778085709041,00001,2,1,1,0,1,52.339191,127.000000
1778085709041,00002,2,1,1,0,1,42.483397,127.000000
1778085709141,00001,2,2,1,0,1,47.622399,127.000000
```

Useful fields:

- UE IMSI
- number of active UEs
- LTE same-cell SINR
- encoded SINR field

### `cu-cp-cell-2.txt`

CU-CP style KPMs for a mmWave cell, including serving and neighbor SINR values.

Example excerpt:

```text
1778084831753,00002,1,1,0,0,2,2,-43.518278,0.000000,3,-22.702917,1.000000
1778084831853,00002,1,1,0,0,2,2,-35.785445,0.000000,3,-37.542338,0.000000
1778084831953,00002,1,1,0,0,2,2,-35.050548,0.000000,3,-44.663228,0.000000
```

Interpretation:

- these rows show the mobility-driven degradation on the mmWave side
- this is the main evidence that the simple motion can drive the UE into outage-like conditions

### `cu-up-cell-1.txt`

CU-UP style PDCP / throughput KPMs.

Example:

```text
timestamp,ueImsiComplete,DRB.PdcpSduDelayDl(cellAverageLatency),m_pDCPBytesUL(0),m_pDCPBytesDL(cellDlTxVolume),DRB.PdcpSduVolumeDl_Filter.UEID(txBytes),Tot.PdcpSduNbrDl.UEID(txDlPackets),DRB.PdcpSduBitRateDl.UEID(pdcpThroughput),DRB.PdcpSduDelayDl.UEID(pdcpLatency),txPdcpPduLteRlc,txPdcpPduBytesLteRlc,QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc),DRB.PdcpPduNbrDl.Qos.UEID(txPdcpPduNrRlc)
1778085709241,00001,23.837264,0,354,177,21,1770.000000,29.738231,8,67,110.000000,13
1778085709341,00001,22.957941,0,422,211,25,2110.000000,21.408065,11,93,118.000000,14
```

Useful fields:

- PDCP DL volume
- per-UE transmitted bytes
- PDCP throughput
- PDCP latency

### `du-cell-2.txt`

DU style radio / scheduling KPMs for mmWave.

Example:

```text
1778085709341,00002,111,2,139,139,1,0,0,35,15,0,0,35,0,20,74410,...
1778085709941,00002,111,2,139,139,1,0,0,27,11,27,0,0,0,16,28818,...
1778085710741,00001,111,2,139,139,1,2,0,14,6,14,0,0,3,8,16308,...
```

Useful fields:

- available DL / UL PRBs
- PRB usage
- number of transport blocks
- modulation buckets (`Qpsk`, `16Qam`, `64Qam`)
- PDU volume
- SINR histogram bins

This file is the main source for:

- PRB usage / resource allocation
- MAC PDU / transport-block counts
- 64-QAM usage counters

### `DlTxPhyStats.txt`

Standard LTE PHY transmission stats.

Example:

```text
% time	cellId	IMSI	RNTI	layer	mcs	size	rv	ndi	ccId
222	1	0	1	0	28	9422	0	1	0
234	1	0	2	0	28	9422	0	1	0
```

Useful fields:

- transmission time
- cell ID
- UE / RNTI
- MCS
- transport size in bytes

This is useful for PHY-layer transmission volume analysis on the LTE side.

## 7. Metrics You Can Directly Use

The scenario now produces enough data for the following practical metrics:

- **RLF / outage count**
  - from `RLF_DUMP` or `rlf_metrics.csv`
- **Active cells**
  - from `bsState.txt`
- **Activation timeline**
  - also from `bsState.txt`
- **PDCP throughput / PDCP bytes**
  - from `cu-up-cell-*.txt`
  - also from `DlPdcpStats.txt`
- **RLC stats**
  - from `DlRlcStats.txt`
- **PRB usage / scheduling**
  - from `du-cell-*.txt`
- **Transport-block counts and modulation buckets**
  - from `du-cell-*.txt`
- **PHY transmission sizes**
  - from `DlTxPhyStats.txt` and `UlTxPhyStats.txt`
- **SINR evolution**
  - from `cu-cp-cell-*.txt`

## 8. Practical Interpretation

In its current form, the scenario is useful as a small regression / smoke test for:

- validating that `100 ms` exports are working
- validating that custom RLF counting is working
- validating that a deliberately simple mobility pattern can create low-SINR conditions
- validating that E2-style KPM files and standard ns-3 traces are all generated together

It is not yet a realistic large-scale training scenario. It is better understood as:

- a compact instrumentation test
- a sanity check for file formats and sampling cadence
- a controlled way to verify that the RLF path is actually exercised

## 9. Main Takeaway

The current `scenario-simple-test.cc` is now a working, explanatory, `100 ms` metric-generation scenario with:

- deterministic UE motion
- stable execution
- E2-style KPM logging
- standard ns-3 bearer / PHY tracing
- RLF event generation through simple mobility-induced SINR degradation
