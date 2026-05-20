# AeroRAN: UAV-Enabled O-RAN Simulation and RL Control on ns-3

## Overview

This repository extends the `ns3-mmwave` / `ns-O-RAN` simulation stack with support for:

- UAV-mounted gNB mobility;
- realistic UE mobility based on Shanghai-style urban trajectories;
- hierarchical and energy-saving control workflows for O-RAN experimentation;
- online Reinforcement Learning (RL) control integrated with live ns-3 simulations;
- offline trace aggregation for post-processing and dataset generation.

## Research Context

The central research scenario implemented in this repository is:

- `scratch/scenario-hierarchical-xangai-UAV.cc`

This scenario combines:

- LTE anchor and mmWave/NR secondary cells;
- UE mobility in urban settings;
- optional UAV mobility for NR base stations;
- hierarchical control interfaces for traffic steering and energy saving;
- KPI generation at periodic control intervals, typically `100 ms`.

The repository also includes a static counterpart:

- `scratch/scenario-hierarchical-xangai.cc`

and a lightweight validation scenario:

- `scratch/scenario-simple-test.cc`


### UAV-Enabled gNB Mobility

The UAV scenario extends conventional gNB deployment by allowing NR base stations to operate as aerial nodes. Supported mobility patterns include:

- hovering,
- circular motion,
- patrol trajectories,
- figure-eight motion,
- grid/lawnmower coverage,
- altitude-varying motion.

These behaviors are controlled through scenario parameters such as:

- `uavMobilityMode`,
- `uavFlightPattern`,
- `uavBaseAltitude`,
- `uavMaxSpeed`,
- `uavOrbitRadius`,
- `uavPatrolLength`,
- `uavGridSize`.

### Realistic UE Mobility

The project supports multiple UE placement and movement modes, including:

- uniform/random allocation,
- allocation around selected BSs,
- Shanghai-inspired urban mobility traces.

This enables comparison between static, semi-structured, and more realistic urban access dynamics.

### O-RAN-Oriented KPI Export

The scenarios generate traces and E2-style measurement files that can be used for:

- throughput analysis,
- latency analysis,
- RLF analysis,
- PRB utilization analysis,
- modulation statistics,
- cell state tracking,
- offline dataset generation.

Typical output files include:

- `bsState.txt`
- `cu-cp-cell-*.txt`
- `cu-up-cell-*.txt`
- `du-cell-*.txt`
- `DlPdcpStats.txt`
- `DlRlcStats.txt`
- `DlPhyTransmissionTrace.txt`
- `EnbSchedAllocTraces.txt`

### RL Integration

The repository contains both SAC- and PPO-based control paths.

- SAC support is centered around `sac1/` and `run_hierarchical_sac.py`.
- PPO support is centered around:
  - `ppo_env.py`
  - `ppo_trainer.py`
  - `ppo_runner.py`

The PPO path currently focuses on the **Energy Saving (ES)** use case, where the controller decides the ON/OFF state of NR cells online while ns-3 is running.

## Repository Structure

Key directories and files:

- `scratch/`
  - simulation scenarios, including UAV and static hierarchical cases
- `src/`
  - ns-3 / mmWave / LTE model code used by the scenarios
- `ns-o-ran-gym/`
  - Gym-compatible environment and ns-3 control bridge
- `sac1/`
  - SAC environments, training logic, and hierarchical agent code
- `1_offline_train/`
  - control payload generation utilities and offline action CSVs
- `aggregator_ppo/`
  - PPO-oriented post-processing pipeline
- `run-xangai-uav.sh`
  - runner for the UAV hierarchical scenario
- `run-xangai-static.sh`
  - runner for the static hierarchical scenario
- `run-simple-test.sh`
  - lightweight validation runner
- `sem_campaign_runner.py`
  - multi-simulation execution with `sem`
- `export_simulation_to_csv.py`
  - standalone raw-trace-to-CSV export utility

## External Dependencies

This repository depends on components from the ns-O-RAN ecosystem.

### E2SIM

The E2SIM implementation is required, especially for ASN.1 and E2-related headers used by the project.

Recommended installation flow:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake libsctp-dev autoconf automake libtool bison flex libboost-all-dev
sudo apt-get install -y g++ python3

cd
git clone https://github.com/wineslab/ns-o-ran-e2-sim oran-e2sim
cd oran-e2sim/e2sim
mkdir build
./build_e2sim.sh 3
```

### ns-o-ran-gym

The RL environments are built on top of:

- `https://github.com/wineslab/ns-o-ran-gym`

If not already on the repository, clone it into the root:

```bash
git clone https://github.com/wineslab/ns-o-ran-gym
```

### O-RAN ns-3 Interface

The `oran-interface` module is expected under `contrib/`:

```bash
cd contrib
git clone -b master https://github.com/o-ran-sc/sim-ns3-o-ran-e2 oran-interface
cd ..
```

## Build

This project uses the `ns3` wrapper and can be configured with Ninja:

```bash
sudo apt-get install -y ninja-build make
./ns3 configure -G Ninja --build-profile=optimized
./ns3 build
```

## Simple testing

To test that the build and runtime pipeline are working, use the simple test scenario:

```bash
SIM_TIME=5 RESULTS_DIR=results_simple_test ./run-simple-test.sh
```

Optionally, the simple test can be driven by an external control payload:

```bash
CONTROL_PAYLOAD_FILE=/path/to/hierarchical_actions.csv \
SCHEDULE_CONTROL_MESSAGES=1 \
./run-simple-test.sh
```

## Running the Main Scenarios

### Static Hierarchical Scenario

This scenario keeps the base stations fixed while allowing realistic UE mobility:

```bash
SIM_TIME=5 RESULTS_DIR=baseline_static_5s POSITION_ALLOCATOR=2 ./run-xangai-static.sh
```

With an external control file:

```bash
CONTROL_PAYLOAD_FILE=/path/to/hierarchical_actions.csv \
USE_SEMAPHORES=1 \
SIM_TIME=5 \
RESULTS_DIR=baseline_static_5s \
POSITION_ALLOCATOR=2 \
./run-xangai-static.sh
```

### UAV Hierarchical Scenario

This is the main aerial scenario. It combines UE mobility with mobile UAV gNBs:

```bash
SIM_TIME=30 \
RESULTS_DIR=uav_res \
UAV_MOBILITY_MODE=1 \
UAV_FLIGHT_PATTERN=4 \
POSITION_ALLOCATOR=2 \
CONTROL_PAYLOAD_FILE=/path/to/hierarchical_actions.csv \
SCHEDULE_CONTROL_MESSAGES=1 \
./run-xangai-uav.sh
```

### Direct ns-3 run command

```bash
./ns3 run "scenario-hierarchical-xangai-UAV --simTime=10.0 --uavMobilityMode=1 --uavFlightPattern=1"
```

## Control Payloads and Offline Actions

The hierarchical scenarios can be controlled through CSV payloads. These files typically contain rows of the form:

```text
timestamp,action_type,param1,param2
```

For Energy Saving control:

```text
timestamp,0,cellId,state
```

Utilities in `1_offline_train/` help generate such files. For example:

```bash
cd 1_offline_train
python3 generate_actions_batch.py 8 --seed 78
cd ..
```

## Parallel Campaigns with `sem`

For repeated or parallel execution, the repository includes a `sem`-based runner:

```bash
python3 sem_campaign_runner.py
```

This is useful for generating multiple independent simulation outputs in parallel, especially when each run uses a separate control file.

## Online RL Training

### SAC

The SAC path uses online interaction between the RL environment and a live ns-3 simulation:

```bash
python3 sac1/train_sac.py \
  --ns3_path . \
  --config sac1/hierarchical_use_case.json \
  --output_folder output_hierarchical_sac \
  --total_timesteps 300000 \
  --save_path sac_hierarchical_model
```

### PPO

The PPO path is designed for the ES-only control problem, where the agent controls ON/OFF decisions for NR cells:

```bash
python3 ppo_trainer.py \
  --ns3_path . \
  --output_folder output_ppo_es_uav \
  --total_timesteps 10000 \
  --save_path ppo_es_uav_model \
  --optimized
```

Important characteristics of the PPO path:

- the model interacts online with a live ns-3 simulation;
- the environment exposes only the ES observation vector;
- the action is a binary ON/OFF decision per NR cell;
- the reward is ES-oriented and includes radio/QoS penalties.

## PPO Inference

Once a PPO model has been trained and saved as a `.zip`, it can be run online as the controller:

```bash
python3 ppo_runner.py \
  --model_path ppo_es_uav_model.zip \
  --ns3_path . \
  --output_folder output_ppo_inference \
  --num_episodes 5 \
  --deterministic \
  --optimized
```

This will:

- load the trained PPO model,
- launch ns-3 through the PPO environment,
- apply the model’s ON/OFF actions online,
- generate raw trace files in per-run output folders.


