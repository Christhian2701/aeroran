# ns-3 O-RAN Extension: UAV Mobility & Hierarchical RL (ISCC Submission)

This repository is a fork of **ns-O-RAN** (based on the **ns3-mmwave** module), extended to support advanced scenarios involving Unmanned Aerial Vehicles (UAVs) as base stations, realistic urban mobility (Shanghai traces), and hierarchical Reinforcement Learning (RL) control for Traffic Steering (TS) and Energy Saving (ES).

## 📄 ISCC Paper Contribution

**This repository contains the source code and simulation scenarios developed for a paper submitted to the IEEE Symposium on Computers and Communications (ISCC).**

### New Scenario: `scenario-hierarchical-xangai-UAV.cc`

The core contribution is the implementation of a new simulation scenario located in `src/mmwave/examples/scenario-hierarchical-xangai-UAV.cc`. This scenario integrates several advanced features designed to evaluate hierarchical control in dynamic 5G O-RAN environments:

* **UAV Base Station Mobility:**
  * Transforms standard gNBs into UAV-mounted base stations (UAV-BS).
  * Supports multiple 3D flight patterns configurable via global variables: *Hovering (with GPS drift), Circular Orbit, Linear Patrol, Figure-Eight, Grid Coverage (Lawnmower),* and *Altitude Variation*.
  * Configurable parameters for altitude, speed, orbit radius, and patrol length.

* **Shanghai Urban Mobility:**
  * Implements realistic User Equipment (UE) movement based on Shanghai urban traces.
  * Includes distinct movement patterns: *Highway, Urban Turning, Intersection Crossing, Roundabout, Stop-and-Go (Traffic Light),* and *Diagonal Movement*.

* **Hierarchical RL Integration:**
  * Designed to work with a hierarchical RL agent controlling both **Traffic Steering (TS)** (via forced handovers based on SINR) and **Energy Saving (ES)** (via cell ON/OFF switching).
  * Generates comprehensive KPIs for agent training, including per-UE SINR/throughput and aggregated energy metrics.
  * Includes `BsStateTrace` for logging cell states (ON/OFF) essential for the ES agent's observation space.

### Running the ISCC Scenario

To run the specific scenario developed for the paper:

```bash
./ns3 --run "scenario-hierarchical-xangai-UAV --simTime=10.0 --uavMobilityMode=1 --uavFlightPattern=1"
