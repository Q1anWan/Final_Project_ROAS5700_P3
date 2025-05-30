import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import os
from config import *

class PerformanceAnalyzer:
    """Performance analysis tools for lane change MPC simulation"""
    
    def __init__(self, results, scenario):
        self.results = results
        self.scenario = scenario
        self.metrics = {}
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        states = self.results['states']
        controls = self.results['controls']
        time_vec = self.results['time']
        
        # Target values
        target_y = self.scenario.target_lane * RoadConfig.LANE_WIDTH + RoadConfig.LANE_WIDTH/2
        target_v = self.scenario.target_velocity
        
        # Completion metrics
        self.metrics['completion_time'] = time_vec[-1]
        self.metrics['lane_change_completed'] = abs(states[-1, 1] - target_y) < 0.5
        
        # Tracking performance
        lateral_error = states[:, 1] - target_y
        velocity_error = states[:, 2] - target_v
        
        self.metrics['rms_lateral_error'] = np.sqrt(np.mean(lateral_error**2))
        self.metrics['max_lateral_error'] = np.max(np.abs(lateral_error))
        self.metrics['final_lateral_error'] = abs(lateral_error[-1])
        
        self.metrics['rms_velocity_error'] = np.sqrt(np.mean(velocity_error**2))
        self.metrics['max_velocity_error'] = np.max(np.abs(velocity_error))
        self.metrics['final_velocity_error'] = abs(velocity_error[-1])
        
        # Safety metrics
        self.metrics['min_obstacle_distance'] = self.calculate_min_obstacle_distance()
        self.metrics['safety_violations'] = self.count_safety_violations()
        
        # Comfort metrics
        if len(controls) > 0:
            steering_rates = controls[:, 0]
            jerks = controls[:, 1]
            
            self.metrics['max_steering_rate'] = np.max(np.abs(steering_rates))
            self.metrics['rms_steering_rate'] = np.sqrt(np.mean(steering_rates**2))
            self.metrics['max_jerk'] = np.max(np.abs(jerks))
            self.metrics['rms_jerk'] = np.sqrt(np.mean(jerks**2))
            
            # Comfort score (combined metric)
            self.metrics['comfort_score'] = np.sqrt(np.mean(steering_rates**2) + np.mean(jerks**2))
        else:
            # Default values if no control data
            self.metrics['max_steering_rate'] = 0.0
            self.metrics['rms_steering_rate'] = 0.0
            self.metrics['max_jerk'] = 0.0
            self.metrics['rms_jerk'] = 0.0
            self.metrics['comfort_score'] = 0.0
        
        # Smoothness metrics
        self.metrics['path_smoothness'] = self.calculate_path_smoothness()
        self.metrics['velocity_smoothness'] = self.calculate_velocity_smoothness()
        
        # Efficiency metrics
        costs = self.results.get('costs', [])
        self.metrics['average_cost'] = np.mean(costs) if len(costs) > 0 else 0.0
        self.metrics['computation_time'] = self.results.get('computation_time', 0.0)
        
        # Lane keeping metrics
        self.metrics['lane_departure_time'] = self.calculate_lane_departure_time()
        self.metrics['lane_return_time'] = self.calculate_lane_return_time()
        
        # Overtaking-specific metrics
        self.metrics['overtaking_efficiency'] = self.calculate_overtaking_efficiency()
        self.metrics['average_overtaking_time'] = self.calculate_average_overtaking_time()
        self.metrics['max_lateral_acceleration'] = self.calculate_max_lateral_acceleration()
        self.metrics['overtaking_smoothness'] = self.calculate_overtaking_smoothness()
    
    def calculate_min_obstacle_distance(self):
        """Calculate minimum distance to any obstacle during simulation"""
        states = self.results['states']
        time_vec = self.results['time']
        min_distance = float('inf')
        
        if not hasattr(self.scenario, 'obstacles') or not self.scenario.obstacles:
            return min_distance
        
        for i, state in enumerate(states):
            if i >= len(time_vec):
                break
            current_time = time_vec[i]
            ego_x, ego_y = state[0], state[1]
            
            for obs in self.scenario.obstacles:
                obs_x_current = obs[0] + obs[2] * current_time
                obs_y = obs[1]
                
                distance = np.sqrt((ego_x - obs_x_current)**2 + (ego_y - obs_y)**2)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 100.0
    
    def count_safety_violations(self):
        """Count number of safety constraint violations"""
        violations = 0
        safety_threshold = 3.0  # meters
        
        if self.metrics['min_obstacle_distance'] < safety_threshold:
            violations += 1
            
        # Check lane boundary violations
        states = self.results['states']
        lane_width = RoadConfig.LANE_WIDTH
        num_lanes = RoadConfig.LANE_COUNT
        
        for state in states:
            y = state[1]
            if y < -lane_width/2 or y > (num_lanes - 0.5) * lane_width:
                violations += 1
                break
        
        return violations
    
    def calculate_path_smoothness(self):
        """Calculate path smoothness using curvature"""
        states = self.results['states']
        if len(states) < 3:
            return 0.0
        
        # Calculate curvature
        x = states[:, 0]
        y = states[:, 1]
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Avoid division by zero
        denominator = (dx**2 + dy**2)**(3/2)
        valid_mask = denominator > 1e-6
        
        curvature = np.zeros_like(dx)
        curvature[valid_mask] = np.abs(dx[valid_mask] * ddy[valid_mask] - dy[valid_mask] * ddx[valid_mask]) / denominator[valid_mask]
        
        return np.mean(curvature[np.isfinite(curvature)])
    
    def calculate_velocity_smoothness(self):
        """Calculate velocity smoothness"""
        states = self.results['states']
        velocities = states[:, 2]
        
        if len(velocities) < 2:
            return 0.0
        
        velocity_changes = np.diff(velocities)
        return np.std(velocity_changes)
    
    def calculate_lane_departure_time(self):
        """Calculate time when vehicle starts departing from initial lane"""
        states = self.results['states']
        time_vec = self.results['time']
        initial_lane = getattr(self.scenario, 'current_lane', 0)
        lane_width = RoadConfig.LANE_WIDTH
        
        initial_lane_center = initial_lane * lane_width + lane_width/2
        threshold = lane_width * 0.3  # 30% of lane width
        
        for i, state in enumerate(states):
            if i >= len(time_vec):
                break
            if abs(state[1] - initial_lane_center) > threshold:
                return time_vec[i]
        
        return time_vec[-1] if len(time_vec) > 0 else 0.0
    
    def calculate_lane_return_time(self):
        """Calculate time when vehicle enters target lane"""
        states = self.results['states']
        time_vec = self.results['time']
        target_lane_center = self.scenario.target_lane * RoadConfig.LANE_WIDTH + RoadConfig.LANE_WIDTH/2
        lane_width = RoadConfig.LANE_WIDTH
        threshold = lane_width * 0.3  # 30% of lane width
        
        for i, state in enumerate(states):
            if i >= len(time_vec):
                break
            if abs(state[1] - target_lane_center) < threshold:
                return time_vec[i]
        
        return time_vec[-1] if len(time_vec) > 0 else 0.0
    
    def calculate_overtaking_efficiency(self):
        """Calculate overtaking efficiency ratio"""
        overtaking_events = self.results.get('overtaking_events', [])
        attempts = len([e for e in overtaking_events if 'start' in e.get('type', '')])
        completions = len([e for e in overtaking_events if 'complete' in e.get('type', '')])
        
        return completions / attempts if attempts > 0 else 1.0
    
    def calculate_average_overtaking_time(self):
        """Calculate average time for overtaking maneuvers"""
        overtaking_events = self.results.get('overtaking_events', [])
        if len(overtaking_events) < 2:
            return self.metrics.get('completion_time', 0.0)
        
        start_times = [e['time'] for e in overtaking_events if 'start' in e.get('type', '')]
        end_times = [e['time'] for e in overtaking_events if 'complete' in e.get('type', '')]
        
        if len(start_times) > 0 and len(end_times) > 0:
            overtaking_times = []
            for start in start_times:
                for end in end_times:
                    if end > start:
                        overtaking_times.append(end - start)
                        break
            return np.mean(overtaking_times) if overtaking_times else 0.0
        return 0.0
    
    def calculate_max_lateral_acceleration(self):
        """Calculate maximum lateral acceleration during simulation"""
        states = self.results['states']
        if len(states) < 2:
            return 0.0
        
        # Calculate lateral acceleration from velocity and heading
        velocities = states[:, 2]
        headings = states[:, 3] if states.shape[1] > 3 else np.zeros(len(states))
        
        lat_accels = []
        dt = SimulationConfig.DT
        for i in range(1, len(states)):
            if velocities[i] > 0:
                heading_rate = (headings[i] - headings[i-1]) / dt
                lat_accel = velocities[i] * heading_rate
                lat_accels.append(abs(lat_accel))
        
        return max(lat_accels) if lat_accels else 0.0
    
    def calculate_overtaking_smoothness(self):
        """Calculate smoothness of overtaking maneuver"""
        controls = self.results['controls']
        if len(controls) == 0:
            return 1.0
        
        # Smoothness based on control input variations
        steering_rates = controls[:, 0]
        jerks = controls[:, 1]
        
        steering_smoothness = 1.0 / (1.0 + np.std(steering_rates))
        jerk_smoothness = 1.0 / (1.0 + np.std(jerks))
        
        return (steering_smoothness + jerk_smoothness) / 2.0
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        report = f"""
PERFORMANCE ANALYSIS REPORT
{'='*50}

COMPLETION METRICS:
- Completion Time: {self.metrics['completion_time']:.2f} seconds
- Lane Change Completed: {'✓' if self.metrics['lane_change_completed'] else '✗'}
- Lane Departure Time: {self.metrics['lane_departure_time']:.2f} seconds
- Lane Return Time: {self.metrics['lane_return_time']:.2f} seconds

TRACKING PERFORMANCE:
- RMS Lateral Error: {self.metrics['rms_lateral_error']:.3f} m
- Max Lateral Error: {self.metrics['max_lateral_error']:.3f} m
- Final Lateral Error: {self.metrics['final_lateral_error']:.3f} m
- RMS Velocity Error: {self.metrics['rms_velocity_error']:.3f} m/s
- Max Velocity Error: {self.metrics['max_velocity_error']:.3f} m/s

SAFETY METRICS:
- Min Obstacle Distance: {self.metrics['min_obstacle_distance']:.2f} m
- Safety Violations: {self.metrics['safety_violations']}
- Safety Status: {'✓ SAFE' if self.metrics['min_obstacle_distance'] > 3.0 else '⚠ UNSAFE'}

COMFORT METRICS:
- Max Steering Rate: {self.metrics['max_steering_rate']:.3f} rad/s
- RMS Steering Rate: {self.metrics['rms_steering_rate']:.3f} rad/s
- Max Jerk: {self.metrics['max_jerk']:.3f} m/s³
- RMS Jerk: {self.metrics['rms_jerk']:.3f} m/s³
- Comfort Score: {self.metrics['comfort_score']:.3f}

SMOOTHNESS METRICS:
- Path Smoothness: {self.metrics['path_smoothness']:.6f}
- Velocity Smoothness: {self.metrics['velocity_smoothness']:.3f}

EFFICIENCY METRICS:
- Average Cost: {self.metrics['average_cost']:.2f}
- Computation Time: {self.metrics['computation_time']:.3f} seconds

OVERALL RATING:
{self.calculate_overall_rating()}
"""
        return report
    
    def generate_overtaking_report(self):
        """Generate specialized report for overtaking performance"""
        overtaking_events = self.results.get('overtaking_events', [])
        
        report = f"""
OVERTAKING PERFORMANCE ANALYSIS
{'='*50}

OVERTAKING SUMMARY:
- Total Overtaking Events: {len(overtaking_events)}
- Successful Overtakes: {len([e for e in overtaking_events if e.get('type') == 'overtake_complete'])}
- Average Overtaking Time: {self.metrics['average_overtaking_time']:.2f} seconds
- Overtaking Efficiency: {self.metrics['overtaking_efficiency']:.1%}

SAFETY METRICS:
- Minimum Distance to Obstacles: {self.metrics['min_obstacle_distance']:.2f} m
- Safety Violations: {self.metrics['safety_violations']}
- Lane Departure Safety: {'✓ SAFE' if self.metrics['lane_departure_time'] > 1.0 else '⚠ UNSAFE'}

COMFORT DURING OVERTAKING:
- Maximum Lateral Acceleration: {self.metrics['max_lateral_acceleration']:.2f} m/s²
- Overtaking Smoothness Score: {self.metrics['overtaking_smoothness']:.3f}

DETAILED EVENTS:
"""
        
        for i, event in enumerate(overtaking_events):
            report += f"  Event {i+1}: {event.get('type', 'unknown')} at t={event.get('time', 0):.1f}s\n"
        
        report += f"\n{self.generate_report()}"
        
        return report
    
    def calculate_overall_rating(self):
        """Calculate overall performance rating"""
        score = 100
        
        # Deduct points for poor performance
        if not self.metrics['lane_change_completed']:
            score -= 30
        
        if self.metrics['completion_time'] > 12:
            score -= 10
        
        if self.metrics['min_obstacle_distance'] < 3.0:
            score -= 25
        
        if self.metrics['rms_lateral_error'] > 0.5:
            score -= 10
        
        if self.metrics['comfort_score'] > 0.5:
            score -= 10
        
        if self.metrics['safety_violations'] > 0:
            score -= 15
        
        if score >= 90:
            return "EXCELLENT (A+)"
        elif score >= 80:
            return "GOOD (A)"
        elif score >= 70:
            return "SATISFACTORY (B)"
        elif score >= 60:
            return "NEEDS IMPROVEMENT (C)"
        else:
            return "POOR (F)"
    
    def save_metrics_to_csv(self):
        """Save metrics to CSV file in output directory"""
        df = pd.DataFrame([self.metrics])
        filepath = os.path.join(SimulationConfig.OUTPUT_DIR, 'performance_metrics.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Metrics saved to: {os.path.abspath(filepath)}")
        return df
    
    def plot_performance_comparison(self, other_analyzers=None):
        """Plot performance comparison with other configurations"""
        if other_analyzers is None:
            other_analyzers = []
        
        # Prepare data for comparison
        metrics_names = ['completion_time', 'rms_lateral_error', 'min_obstacle_distance', 'comfort_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            values = [self.metrics.get(metric, 0)]
            labels = ['Current']
            
            for j, analyzer in enumerate(other_analyzers):
                values.append(analyzer.metrics.get(metric, 0))
                labels.append(f'Config {j+1}')
            
            axes[i].bar(labels, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save to output directory
        filepath = os.path.join(SimulationConfig.OUTPUT_DIR, 'performance_comparison.png')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {os.path.abspath(filepath)}")
        plt.show()

class BenchmarkSuite:
    """Benchmark suite for testing different MPC configurations"""
    
    def __init__(self):
        self.configurations = self.generate_configurations()
        self.results = {}
    
    def generate_configurations(self):
        """Generate different MPC configurations for benchmarking - improved"""
        base_weights = {
            'Q_Y': MPCWeights.Q_Y,
            'Q_V': MPCWeights.Q_V,
            'R_DELTA_DOT': MPCWeights.R_DELTA_DOT,
            'R_JERK': MPCWeights.R_JERK,
            'Q_TERMINAL_Y': MPCWeights.Q_TERMINAL_Y
        }
        
        configurations = {
            'baseline': base_weights.copy(),
            'comfort_focused': {
                **base_weights, 
                'R_DELTA_DOT': 200.0, 
                'R_JERK': 120.0,
                'Q_Y': 8.0,  # Reduced for better convergence
                'SLACK_WEIGHT': 300.0
            },
            'performance_focused': {
                **base_weights, 
                'Q_Y': 20.0, 
                'Q_V': 3.0,
                'R_DELTA_DOT': 80.0,  # Increased for stability
                'R_JERK': 60.0,
                'SLACK_WEIGHT': 400.0
            },
            'safety_focused': {
                **base_weights, 
                'Q_Y': 12.0,  # Reduced
                'R_DELTA_DOT': 180.0,
                'R_JERK': 100.0,
                'SLACK_WEIGHT': 200.0  # Much reduced
            },
            'balanced': {
                **base_weights, 
                'Q_Y': 15.0,
                'R_DELTA_DOT': 150.0,
                'R_JERK': 80.0,
                'SLACK_WEIGHT': 300.0
            }
        }
        
        return configurations
    
    def run_benchmark(self):
        """Run benchmark tests for all configurations"""
        print("Running benchmark suite...")
        
        for config_name, weights in self.configurations.items():
            print(f"  Testing configuration: {config_name}")
            
            try:
                # Import here to avoid circular imports
                from lane_change_mpc import LaneChangeScenario
                
                # Create scenario with custom weights
                scenario = LaneChangeScenario()
                
                # Apply custom weights to MPC
                if hasattr(scenario, 'mpc'):
                    for weight_name, value in weights.items():
                        if hasattr(MPCWeights, weight_name):
                            setattr(MPCWeights, weight_name, value)
                
                # Run simulation
                results = scenario.run_simulation()
                analyzer = PerformanceAnalyzer(results, scenario)
                self.results[config_name] = analyzer
                
                rating = analyzer.calculate_overall_rating()
                print(f"    ✓ Completed - Score: {rating}")
                
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
                self.results[config_name] = None
        
        print("Benchmark suite completed!")
    
    def generate_comparison_report(self):
        """Generate comparison report for all configurations"""
        if not self.results:
            return "No benchmark results available."
        
        report = f"""
BENCHMARK COMPARISON REPORT
{'='*60}

Configuration Performance Summary:
"""
        
        # Create comparison table
        metrics_to_compare = [
            'completion_time', 'rms_lateral_error', 'min_obstacle_distance', 
            'comfort_score', 'safety_violations'
        ]
        
        for config_name, analyzer in self.results.items():
            if analyzer is not None:
                report += f"\n{config_name.upper()}:\n"
                report += f"  Overall Rating: {analyzer.calculate_overall_rating()}\n"
                for metric in metrics_to_compare:
                    value = analyzer.metrics.get(metric, 'N/A')
                    if isinstance(value, float):
                        report += f"  {metric}: {value:.3f}\n"
                    else:
                        report += f"  {metric}: {value}\n"
        
        # Find best configuration
        best_config = self.find_best_configuration()
        report += f"\nBEST CONFIGURATION: {best_config}\n"
        
        return report
    
    def find_best_configuration(self):
        """Find the best performing configuration"""
        best_score = -1
        best_config = "None"
        
        for config_name, analyzer in self.results.items():
            if analyzer is not None:
                # Calculate numerical score
                score = 100
                metrics = analyzer.metrics
                
                if not metrics['lane_change_completed']:
                    score -= 30
                if metrics['completion_time'] > 12:
                    score -= 10
                if metrics['min_obstacle_distance'] < 3.0:
                    score -= 25
                if metrics['rms_lateral_error'] > 0.5:
                    score -= 10
                if metrics['comfort_score'] > 0.5:
                    score -= 10
                if metrics['safety_violations'] > 0:
                    score -= 15
                
                if score > best_score:
                    best_score = score
                    best_config = config_name
        
        return best_config
    
    def save_benchmark_results(self):
        """Save benchmark results to CSV in output directory"""
        data = []
        for config_name, analyzer in self.results.items():
            if analyzer is not None:
                row = {'configuration': config_name}
                row.update(analyzer.metrics)
                data.append(row)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(SimulationConfig.OUTPUT_DIR, 'benchmark_results.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Benchmark results saved to: {os.path.abspath(filepath)}")
        return df

if __name__ == "__main__":
    # Example usage
    try:
        from lane_change_mpc import LaneChangeScenario
        
        # Run simulation
        scenario = LaneChangeScenario()
        results = scenario.run_simulation()
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(results, scenario)
        print(analyzer.generate_report())
        print("\n" + analyzer.generate_overtaking_report())
        
        # Run benchmark
        benchmark = BenchmarkSuite()
        benchmark.run_benchmark()
        print(benchmark.generate_comparison_report())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run this module from main_simulation.py for full functionality")
