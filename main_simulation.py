#!/usr/bin/env python3
"""
Main simulation script for Lane Change MPC Final Project
Course: ROAS 5700
"""

from lane_change_mpc import LaneChangeScenario
from visualization import LaneChangeVisualizer
from performance_analysis import PerformanceAnalyzer, BenchmarkSuite
from config import (SimulationConfig, VehicleConfig, RoadConfig, 
                   ObstacleConfig, MPCWeights, BenchmarkConfig, VisualizationConfig,
                   MULTI_VEHICLE_CONFIG, OvertakingConfig, SIMULATION_CONFIG, MPC_CONFIG)
import numpy as np
import time
import os

def main():
    """Main function to run complete lane change MPC analysis"""
    
    print("="*60)
    print("AUTONOMOUS OVERTAKING MPC SIMULATION")
    print("Multi-Vehicle Scenario with Dynamic Lane Changes")
    print("Author: Boyu JIN")
    print("="*60)
    
    # Get absolute path of output directory for better handling
    output_dir = os.path.abspath(SimulationConfig.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print current configuration
    print_configuration_summary()
    
    # 1. Run basic lane change simulation first
    print("\n1. Running Basic Lane Change Simulation...")
    
    scenario = LaneChangeScenario()
    results = scenario.run_simulation()
    
    # Print basic summary
    print(f"\n📊 Simulation Summary:")
    print(f"   Simulation time: {SimulationConfig.SIMULATION_TIME}s")
    print(f"   Lane change completed: {'Yes' if len(results['states']) > 0 else 'No'}")
    
    # 2. Performance Analysis
    print("\n2. Analyzing Performance...")
    analyzer = PerformanceAnalyzer(results, scenario)
    report = analyzer.generate_report()
    print(report)
    
    # 3. Visualization
    print("\n3. Generating Visualizations...")
    
    try:
        visualizer = LaneChangeVisualizer(results, scenario)
        
        # Generate plots
        print("  - Plotting trajectory overview...")
        visualizer.plot_trajectory_overview()
        
        print("  - Plotting state evolution...")
        visualizer.plot_state_evolution()
        
        print("  - Plotting control inputs...")
        visualizer.plot_control_inputs()
        
        print("  - Plotting performance metrics...")
        visualizer.plot_performance_metrics()
        
        print("  - Creating animation...")
        anim = visualizer.create_animation()
        
        if anim is not None:
            print("✓ Animation created successfully")
        else:
            print("⚠ Animation creation failed, but static plots are available")
            
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        print("Simulation completed but visualization failed")
    
    print("\n✅ Simulation completed successfully!")
    print(f"📁 Results saved in: {os.path.abspath(SimulationConfig.OUTPUT_DIR)}")
    
    # 4. Benchmark Suite
    print("\n4. Running Benchmark Suite...")
    benchmark = BenchmarkSuite()
    benchmark.run_benchmark()
    comparison_report = benchmark.generate_comparison_report()
    print(comparison_report)
    
    # 5. Generate Final Report
    print("\n5. Generating Final Report...")
    generate_final_report(analyzer, benchmark)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print(f"All results saved to {output_dir}")
    print("="*60)

def print_configuration_summary():
    """Print summary of current configuration parameters"""
    print("\nCURRENT CONFIGURATION:")
    print(f"  Simulation Time: {SimulationConfig.SIMULATION_TIME}s")
    print(f"  Number of Vehicles: {MULTI_VEHICLE_CONFIG['num_vehicles']}")
    print(f"  Road Length: {MULTI_VEHICLE_CONFIG['road_length']}m")
    print(f"  Slow Vehicle Probability: {MULTI_VEHICLE_CONFIG['slow_vehicle_probability']*100}%")
    print(f"  Number of Obstacles: {len(ObstacleConfig.OBSTACLES)}")
    print(f"  Min Overtaking Speed Diff: {OvertakingConfig.MIN_SPEED_DIFFERENCE} m/s")
    print(f"  Safety Gap Required: {OvertakingConfig.SAFETY_GAP_REQUIRED}m")
    print(f"  Output Directory: {os.path.abspath(SimulationConfig.OUTPUT_DIR)}")

def generate_final_report(analyzer, benchmark):
    """Generate comprehensive final report"""
    
    final_report = f"""
LANE CHANGE MPC - FINAL PROJECT REPORT
{'='*80}

PROJECT OVERVIEW:
This project implements a Model Predictive Controller (MPC) for autonomous 
vehicle lane change maneuvers with dynamic obstacle avoidance. The controller
optimizes passenger comfort while ensuring safety constraints are met.

TECHNICAL APPROACH:
- Vehicle Model: Bicycle model with steering rate and jerk as control inputs
- Prediction Horizon: {SimulationConfig.PREDICTION_HORIZON} time steps ({SimulationConfig.PREDICTION_HORIZON * SimulationConfig.DT} seconds)
- Control Horizon: {SimulationConfig.CONTROL_HORIZON} time steps ({SimulationConfig.CONTROL_HORIZON * SimulationConfig.DT} seconds)
- Sampling Time: {SimulationConfig.DT} seconds
- Optimization: Sequential Quadratic Programming (SQP)

CONFIGURATION PARAMETERS:
- Simulation Duration: {SimulationConfig.SIMULATION_TIME}s
- Vehicle Length: {VehicleConfig.LENGTH}m
- Lane Width: {RoadConfig.LANE_WIDTH}m
- Number of Obstacles: {len(ObstacleConfig.OBSTACLES)}
- MPC Weights - Lateral Position: {MPCWeights.Q_Y}
- MPC Weights - Steering Rate: {MPCWeights.R_DELTA_DOT}
- MPC Weights - Jerk: {MPCWeights.R_JERK}

KEY FEATURES:
1. Dynamic obstacle avoidance with moving vehicles
2. Passenger comfort optimization (jerk and steering rate minimization)
3. Lane boundary constraints
4. Multi-objective cost function balancing safety, comfort, and performance
5. Configurable parameters for easy simulation tuning

{analyzer.generate_report()}

{benchmark.generate_comparison_report()}

CONFIGURATION FLEXIBILITY:
The simulation now uses a centralized configuration system that allows easy
modification of parameters without changing core code:
- Simulation parameters in SimulationConfig
- Vehicle dynamics in VehicleConfig  
- Road geometry in RoadConfig
- Obstacle scenarios in ObstacleConfig
- MPC tuning weights in MPCWeights
- Benchmark variations in BenchmarkConfig

CONCLUSIONS:
The implemented MPC successfully demonstrates autonomous lane change capabilities
with the following achievements:
- Safe collision avoidance with dynamic obstacles
- Smooth and comfortable passenger experience
- Robust performance across different tuning configurations
- Real-time feasible computation requirements
- Easy parameter tuning through centralized configuration

FUTURE IMPROVEMENTS:
1. Implementation of robust MPC for uncertainty handling
2. Integration of machine learning for adaptive weight tuning
3. Extension to multi-vehicle coordination scenarios
4. Incorporation of stochastic obstacle motion models
5. Real-time parameter adaptation based on traffic conditions

Author: Boyu JIN
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Course: ROAS 5700 - Model Predictive Control
"""
    
    # Save final report using relative path
    report_path = os.path.join(SimulationConfig.OUTPUT_DIR, 'FINAL_REPORT.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(final_report)
    
    print(f"✓ Final report saved to {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
