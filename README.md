# Lane Change MPC - Final Project Section 3

**Author:** bjin457@connect.hkust-gz.edu.cn

## Overview

This project implements a Model Predictive Controller (MPC) for autonomous vehicle lane change maneuvers with dynamic obstacle avoidance. The system demonstrates advanced autonomous driving capabilities including safety constraints, passenger comfort optimization, and real-time performance.

## Features

- **Lane Change Controller**: Smooth and safe lane change maneuvers
- **Dynamic Obstacle Avoidance**: Real-time collision avoidance with moving vehicles
- **Comfort Optimization**: Minimizes jerk and steering rate for passenger comfort
- **Multi-objective Optimization**: Balances safety, comfort, and performance
- **Comprehensive Analysis**: Performance metrics, visualizations, and benchmarking
- **Configurable Parameters**: Centralized configuration system for easy tuning

## Project Structure

```
Final_Project_MPC_LD/
├── config.py                   # Configuration parameters (NEW)
├── lane_change_mpc.py          # Main MPC implementation
├── visualization.py            # Visualization tools
├── performance_analysis.py     # Performance analysis and benchmarking
├── main_simulation.py          # Main simulation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── results/                    # Generated results (relative path)
    ├── trajectory_overview.png
    ├── state_evolution.png
    ├── control_inputs.png
    ├── performance_metrics.png
    ├── lane_change_animation.gif
    ├── performance_metrics.csv
    └── FINAL_REPORT.md
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the main simulation:
```bash
python main_simulation.py
```

## Configuration System

All simulation parameters are now centralized in `config.py` for easy modification:

### SimulationConfig
- Simulation time and time step
- MPC prediction and control horizons
- Lane change timing parameters
- Output directory (relative path)

### VehicleConfig
- Vehicle dimensions and dynamics limits
- Initial and target conditions
- Physical constraints

### RoadConfig
- Lane geometry and road parameters
- Safety margins

### ObstacleConfig
- Obstacle vehicle configurations
- Prediction parameters

### MPCWeights
- Cost function weight tuning
- Comfort and performance balance

### BenchmarkConfig
- Predefined parameter variations
- Scenario configurations

### VisualizationConfig
- Plot settings and colors
- Animation parameters

## Technical Details

### Vehicle Model
- **Type**: Bicycle model with 6 states [x, y, v, θ, δ, a]
- **Control Inputs**: Steering rate (δ̇) and jerk (j)
- **Constraints**: Speed limits, acceleration limits, steering limits

### MPC Configuration (Configurable in config.py)
- **Prediction Horizon**: 20 steps (2.0 seconds) - adjustable
- **Control Horizon**: 10 steps (1.0 seconds) - adjustable
- **Sample Time**: 0.1 seconds - adjustable
- **Solver**: Sequential Quadratic Programming (SQP)

### Cost Function Components
1. **Tracking Cost**: Lateral position and velocity tracking
2. **Control Effort**: Steering and acceleration penalties
3. **Comfort Cost**: Jerk and steering rate minimization
4. **Safety Cost**: Obstacle avoidance penalties
5. **Constraint Cost**: Lane boundary violations

### Safety Features
- **Collision Avoidance**: Soft constraints for dynamic obstacles
- **Lane Boundaries**: Hard constraints preventing lane departure
- **Comfort Limits**: Jerk and steering rate constraints
- **Speed Limits**: Velocity range constraints

## Usage Examples

### Basic Simulation
```python
from lane_change_mpc import LaneChangeScenario

# Create and run scenario
scenario = LaneChangeScenario()
results = scenario.run_simulation()
```

### Modifying Configuration
```python
from config import SimulationConfig, MPCWeights

# Modify simulation time
SimulationConfig.SIMULATION_TIME = 20.0

# Adjust MPC weights for more comfort-focused behavior
MPCWeights.R_DELTA_DOT = 200.0  # Higher steering rate penalty
MPCWeights.R_JERK = 100.0       # Higher jerk penalty
```

### Performance Analysis
```python
from performance_analysis import PerformanceAnalyzer

# Analyze results
analyzer = PerformanceAnalyzer(results, scenario)
report = analyzer.generate_report()
print(report)
```

### Visualization
```python
from visualization import LaneChangeVisualizer

# Create visualizations
visualizer = LaneChangeVisualizer(results, scenario)
visualizer.plot_trajectory_overview()
visualizer.create_animation()
```

### Benchmark Testing
```python
from performance_analysis import BenchmarkSuite

# Run benchmark suite
benchmark = BenchmarkSuite()
benchmark.run_benchmark()
comparison = benchmark.generate_comparison_report()
```

## Results

The MPC controller successfully demonstrates:

- **Lane Change Completion**: < 12 seconds
- **Safety Clearance**: > 3 meters from obstacles
- **Passenger Comfort**: Low jerk and steering rate
- **Tracking Accuracy**: RMS lateral error < 0.5 meters

## Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Completion Time | < 12s | ~8-10s |
| Min Obstacle Distance | > 3m | > 4m |
| RMS Lateral Error | < 0.5m | < 0.3m |
| Max Jerk | < 2 m/s³ | < 1.5 m/s³ |
| Comfort Score | < 0.5 | < 0.4 |

## Easy Parameter Tuning

### For Comfort-Focused Behavior:
```python
# In config.py
MPCWeights.R_DELTA_DOT = 200.0  # Higher steering rate penalty
MPCWeights.R_JERK = 100.0       # Higher jerk penalty
MPCWeights.Q_Y = 5.0            # Lower lateral tracking urgency
```

### For Performance-Focused Behavior:
```python
# In config.py
MPCWeights.R_DELTA_DOT = 50.0   # Lower steering rate penalty
MPCWeights.R_JERK = 25.0        # Lower jerk penalty
MPCWeights.Q_Y = 20.0           # Higher lateral tracking urgency
```

### For Different Scenarios:
```python
# Dense traffic
ObstacleConfig.OBSTACLES = [
    {'initial_position': [80.0, 5.4], 'speed': 15.0},
    {'initial_position': [120.0, 1.8], 'speed': 18.0},
    {'initial_position': [160.0, 5.4], 'speed': 16.0}
]

# Fast traffic
ObstacleConfig.OBSTACLES = [
    {'initial_position': [100.0, 5.4], 'speed': 25.0},
    {'initial_position': [150.0, 1.8], 'speed': 28.0}
]
```

## Configuration Benefits

1. **Easy Tuning**: Modify parameters without touching core code
2. **Portable**: Uses relative paths - works anywhere
3. **Organized**: Parameters grouped by functionality
4. **Documented**: Clear parameter descriptions and units
5. **Flexible**: Pre-defined configurations for different scenarios

## File Organization

- `config.py` - All configurable parameters
- `main_simulation.py` - Orchestrates the complete simulation
- Core modules import configuration as needed
- Results saved to `./results/` relative to project directory

## Author

**bjin457@connect.hkust-gz.edu.cn**  
ROAS 5700 - Model Predictive Control  
Final Project

## License

This project is for educational purposes as part of the ROAS 5700 course requirements.
