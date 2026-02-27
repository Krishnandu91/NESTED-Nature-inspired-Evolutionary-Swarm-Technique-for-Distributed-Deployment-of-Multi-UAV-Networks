#  NESTED FRAMEWORK

A **Discrete-Event Simulation (DES)** framework for studying distributed UAV (Unmanned Aerial Vehicle) swarm coordination, coverage optimization, energy management, and fault tolerance in dynamic wireless networks.

---

##  Project Structure

```
#1_NESTED/
├── config/
│   └── Configuration.py          # All simulation parameters
├── modules/
│   ├── Building_link.py           # Access & backhaul link 
│   ├── Calculation.py             # Path loss, SNR, network 
│   ├── CoverageCalculator.py      # Coverage metrics & tracking
│   ├── DistributedAlgorithm.py    # Core distributed decision 
│   ├── energy.py                  # UAV energy model (hover, fly, comms)
│   ├── EnergyLogger.py            # Energy & message logging to CSV
│   ├── GridManager.py             # Grid-based field management
│   ├── ResultsLogger.py           # Simulation result logging
│   ├── UAVAgent.py                # Individual UAV agent class
│   └── Visualization.py           # Frame-by-frame animation & plots
├── Dataset/
│   ├── Coverage_recover_and_degradation_analysis_100.csv
│   ├── Coverage_recover_and_degradation_analysis_200.csv
│   ├── Coverage_recover_and_degradation_analysis_300.csv
│   ├── Coverage_recover_and_degradation_analysis_20km.csv
│   ├── Coverage_recover_and_degradation_analysis_40km.csv
│   └── Original_Dataset.csv
├── Baseline_Approach/             # Baseline algorithm implementation
├── Baseline_vs_NESTED/            # Comparative evaluation scripts
├── results/                       # Auto-generated output CSVs & summaries
├── visualizations/                # Saved frames & animations (GIF/MP4)
├── Gravity.ipynb                  # Gravity-based UAV placement notebook
├── Sensivity_Analysis.ipynb       # Parameter sensitivity analysis notebook
└── Main.py                        # Simulation entry point
```



## Overview

This project simulates a **multi-UAV wireless network** deployed over a 2D field to provide communication coverage for mobile ground users. UAVs operate as **fully distributed agents** — each makes independent decisions based only on local observations and neighbor message-passing.

### Key Features

- **Distributed Decision-Making** — No central controller; each UAV computes its own movement scores using local coverage maps and neighbor exchanges
- **SimPy Discrete-Event Simulation** — UAV agents run as concurrent SimPy processes synchronized via an AllOf barrier
- **Realistic Energy Model** — Hovering power, kinetic (flight) power, and radio communication power modeled from published parameters.
- **Stochastic & Energy Failures** — UAVs can fail randomly (system malfunction) or die from battery depletion
- **Dynamic UAV Addition** — Checkpoint process detects coverage drops or network fragmentation and deploys replacement UAVs using a gravity-based grid search
- **User Mobility** — Ground users follow real mobility traces loaded from CSV datasets
- **Full Logging** — Energy consumption, message passing, death events, stochastic failures, and coverage all logged per timestep

---

## Configuration
```
All parameters are set in `config/Configuration.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| FIELD_SIZE | 30,000 m | Simulation field (30 × 30 km²) |
| NO_UAV | 5 | Initial number of UAVs |
| UAV_HEIGHT | 1,500 m | Fixed flight altitude |
| NO_PEOPLE | 200 | Number of ground users |
| SIM_DURATION | 75 | Simulation time steps |
| NUM_SIMULATION_RUNS | 50 | Monte Carlo iterations |
| ACCESS_LINK_RANGE | 3,300 m | UAV-to-user link range |
| BACKHAUL_LINK_RANGE` | 8,200 m | UAV-to-UAV link range |
| MAX_BATTERY_ENERGY | 125,000 J | Full battery capacity (≈ 34.7 Wh) |
| STOCHASTIC_FAILURE_THRESHOLD | 0.001 | Probability of random failure per check |
| COVERAGE_THRESHOLD | 80% | Minimum coverage before adding a UAV |
| WE_COVERAGE | 0.70 | Weight for coverage score |
| WE_HOTSPOT | 0.30 | Weight for hotspot score |
| BETA | 0.3 | Neighbor influence weight |
```
### Selecting a Dataset

Edit `USER_MOBILITY_FILE` in `Configuration.py` to choose a scenario:

```python
# 100 users, 30×30 km²
USER_MOBILITY_FILE = r"path/to/Coverage_recover_and_degradation_analysis_100.csv"

# 200 users, 30×30 km² (default)
USER_MOBILITY_FILE = r"path/to/Coverage_recover_and_degradation_analysis_200.csv"

# 300 users, 30×30 km²
USER_MOBILITY_FILE = r"path/to/Coverage_recover_and_degradation_analysis_300.csv"

# 20×20 km² field
USER_MOBILITY_FILE = r"path/to/Coverage_recover_and_degradation_analysis_20km.csv"

# 40×40 km² field
USER_MOBILITY_FILE = r"path/to/Coverage_recover_and_degradation_analysis_40km.csv"

# Original dataset
USER_MOBILITY_FILE = r"path/to/Original_Dataset.csv"
```




### Requirements

```
pip install simpy numpy networkx matplotlib pandas
```

### Running the Simulation

```
python Main.py
```

You will be prompted to choose which iterations to animate:

```
Enter 0        : No visualization
Enter 1        : Only the first iteration
Enter N        : First N iterations
Enter 1,3,5    : Specific iterations (comma-separated)
```

---

## Output Files
```
All outputs are saved to the `results/` directory:

| File | Description |
|---|---|
| simulation_results_<timestamp>.csv | Per-timestep: UAVs, failures, coverage, network efficiency, components |
| energy_consumption_<timestamp>.csv | Per-UAV per-timestep energy breakdown |
| message_passing_<timestamp>.csv | Detailed log of every inter-UAV message |
| message_statistics_<timestamp>.csv | Aggregated message counts per timestep |
| death_events_<timestamp>.csv| UAV death log (cause, energy, neighbors notified) |
| stochastic_failures_<timestamp>.csv | Random failure draw log |
| energy_summary_iter<N>.txt | Per-UAV energy summary for each iteration |
| simulation_summary_iter<N>.txt | Simulation configuration & average metrics |

Animations are saved to `visualizations/`:
- simulation_animation.gif

```
