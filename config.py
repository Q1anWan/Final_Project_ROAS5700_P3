"""
Configuration file for Lane Change MPC simulation parameters
Course: ROAS 5700
"""

import numpy as np

class SimulationConfig:
    """Main simulation configuration parameters"""
    
    # Time parameters
    SIMULATION_TIME = 35.0  # Reduced for better convergence
    DT = 0.1               # Time step (seconds)
    
    # MPC parameters - adjusted for better convergence
    PREDICTION_HORIZON = 15  # Reduced from 20 for faster solving
    CONTROL_HORIZON = 8      # Reduced from 10
    
    # Scenario parameters
    INITIAL_LANE = 0        # Starting lane (0 or 1)
    TARGET_LANE = 1         # Default target lane
    
    # Progress reporting
    PROGRESS_PRINT_INTERVAL = 5.0  # Print progress every N seconds
    
    # Output directory
    OUTPUT_DIR = './results'

class VehicleConfig:
    """Vehicle model parameters - adjusted for better performance"""
    
    # Vehicle dimensions
    LENGTH = 4.5    # Vehicle length (m)
    WIDTH = 1.8     # Vehicle width (m)
    WHEELBASE = 2.7 # Wheelbase (m)
    
    # Vehicle dynamics limits - relaxed for better convergence
    MAX_SPEED = 25.0        # Reduced maximum speed (m/s)
    MIN_SPEED = 8.0         # Increased minimum speed (m/s)
    MAX_ACCELERATION = 2.5   # Reduced for smoother control
    MAX_DECELERATION = -3.0  # Less aggressive deceleration (m/s²)
    MAX_STEERING_ANGLE = np.pi/8  # Reduced for gentler steering (rad)
    MAX_STEERING_RATE = np.pi/6   # Reduced steering rate (rad/s)
    MAX_JERK = 2.0          # Reduced jerk limit (m/s³)
    
    # Initial conditions
    INITIAL_POSITION = [0.0, 1.8]  # [x, y] initial position (m)
    INITIAL_SPEED = 18.0           # Slightly reduced initial speed (m/s)
    INITIAL_HEADING = 0.0          # Initial heading (rad)
    
    # Target conditions
    TARGET_SPEED = 18.0    # Reduced target speed (m/s)

class RoadConfig:
    """Road and lane configuration"""
    
    # Lane parameters
    LANE_WIDTH = 3.6       # Standard lane width (m)
    LANE_COUNT = 2         # Number of lanes
    ROAD_LENGTH = 500.0    # Total road length (m)
    
    # Lane positions (center of each lane)
    LANE_CENTERS = [1.8, 5.4]  # Y-coordinates of lane centers (m)
    
    # Safety margins
    SAFETY_MARGIN_LONGITUDINAL = 5.0  # Longitudinal safety distance (m)
    SAFETY_MARGIN_LATERAL = 0.5       # Lateral safety margin (m)

# Simplified obstacle configuration for better convergence
class ObstacleConfig:
    """Obstacle configuration parameters - simplified for better performance"""
    
    # Fewer, well-spaced obstacles
    OBSTACLES = [
        {
            'id': 'slow_car_1',
            'initial_position': [20.0, 1.8],   # Lane 0 (bottom)
            'speed': 12.0,                     # Slow speed
            'length': 4.5,
            'width': 1.8,
            'type': 'slow_vehicle'
        },
        {
            'id': 'slow_car_2', 
            'initial_position': [70.0, 5.4],  # Lane 0
            'speed': 10.0,                     # Very slow
            'length': 4.5,
            'width': 1.8,
            'type': 'slow_vehicle'
        },
        {
            'id': 'slow_car_3', 
            'initial_position': [120.0, 5.4],  # Lane 0
            'speed': 10.0,                     # Very slow
            'length': 4.5,
            'width': 1.8,
            'type': 'slow_vehicle'
        },
        {
            'id': 'slow_car_4', 
            'initial_position': [120.0, 5.4],  # Lane 0
            'speed': 12.0,                     # Very slow
            'length': 4.5,
            'width': 1.8,
            'type': 'slow_vehicle'
        },
        {
            'id': 'slow_car_5', 
            'initial_position': [200.0, 1.8],  # Lane 0
            'speed': 10.0,                     # Very slow
            'length': 4.5,
            'width': 1.8,
            'type': 'slow_vehicle'
        },
        {
            'id': 'slow_car_6', 
            'initial_position': [170.0, 5.4],  # Lane 0
            'speed': 12.0,                     # Very slow
            'length': 4.5,
            'width': 1.8,
            'type': 'slow_vehicle'
        }
    ]
    
    # Enhanced prediction parameters
    OBSTACLE_PREDICTION_HORIZON = 20  # Reduced for better performance
    OBSTACLE_UNCERTAINTY = 0.1        # Slightly increased uncertainty
    SAFETY_DISTANCE = 4.0             # Increased safety buffer
    OVERTAKING_ACTIVATION_DISTANCE = 40.0  # Reduced activation distance

class MPCWeights:
    """MPC cost function weights - retuned for better convergence"""
    
    # State tracking weights - balanced for stability
    Q_X = 0.5          # Increased longitudinal position weight
    Q_Y = 15.0         # Moderate lateral position weight
    Q_V = 2.0          # Increased velocity weight
    Q_PSI = 8.0        # Increased heading weight
    
    # Control input weights - higher for smoother control
    R_DELTA_DOT = 150.0  # Increased steering rate weight
    R_JERK = 80.0        # Increased jerk weight
    
    # Terminal weights
    Q_TERMINAL_X = 2.0
    Q_TERMINAL_Y = 30.0  # Reduced terminal lateral weight
    Q_TERMINAL_V = 8.0
    Q_TERMINAL_PSI = 15.0
    
    # Slack variable weights - reduced to prevent infeasibility
    SLACK_WEIGHT = 500.0  # Reduced from 1000
    
    # Comfort weights
    COMFORT_WEIGHT_LATERAL_ACCEL = 25.0
    COMFORT_WEIGHT_JERK = 40.0

class BenchmarkConfig:
    """Benchmark suite configuration"""
    
    # Weight tuning variations for benchmark
    WEIGHT_VARIATIONS = [
        {
            'name': 'Comfort_Focused',
            'Q_Y': 5.0,
            'R_DELTA_DOT': 200.0,
            'R_JERK': 100.0,
            'description': 'Prioritizes passenger comfort'
        },
        {
            'name': 'Performance_Focused', 
            'Q_Y': 20.0,
            'R_DELTA_DOT': 50.0,
            'R_JERK': 25.0,
            'description': 'Prioritizes fast lane change'
        },
        {
            'name': 'Balanced',
            'Q_Y': 10.0,
            'R_DELTA_DOT': 100.0,
            'R_JERK': 50.0,
            'description': 'Balanced comfort and performance'
        },
        {
            'name': 'Safety_Focused',
            'Q_Y': 15.0,
            'R_DELTA_DOT': 150.0,
            'R_JERK': 75.0,
            'SLACK_WEIGHT': 2000.0,
            'description': 'Prioritizes safety margins'
        }
    ]
    
    # Scenario variations
    SCENARIO_VARIATIONS = [
        {
            'name': 'Dense_Traffic',
            'obstacles': [
                {'initial_position': [80.0, 5.4], 'speed': 15.0},
                {'initial_position': [120.0, 1.8], 'speed': 18.0},
                {'initial_position': [160.0, 5.4], 'speed': 16.0}
            ]
        },
        {
            'name': 'Fast_Traffic',
            'obstacles': [
                {'initial_position': [100.0, 5.4], 'speed': 25.0},
                {'initial_position': [150.0, 1.8], 'speed': 28.0}
            ]
        }
    ]

class VisualizationConfig:
    """Visualization parameters - improved for better readability"""
    
    # Plot settings - improved resolution and size
    FIGURE_SIZE = (20, 10)  # Much larger figure
    DPI = 200  # Increased DPI for better resolution
    FONT_SIZE = 14  # Larger font size
    
    # Animation settings
    ANIMATION_INTERVAL = 50  # Slightly slower for better visibility
    ANIMATION_FPS = 20  # Adjusted fps
    
    # Colors - enhanced for better distinction
    EGO_VEHICLE_COLOR = '#FF6B6B'  # Bright red for ego vehicle
    OBSTACLE_COLOR = '#4ECDC4'     # Teal for obstacles
    TRAJECTORY_COLOR = '#FFE66D'   # Yellow for trajectory
    LANE_COLOR = '#95E1D3'         # Light green for lanes
    
    # Plot limits - better aspect ratio
    PLOT_XLIM = [-20, 200]  # Longer range
    PLOT_YLIM = [-2, 10]    # Taller for better visibility

# Multi-vehicle simulation configuration for realistic overtaking
MULTI_VEHICLE_CONFIG = {
    'num_vehicles': 8,  # More vehicles for realistic traffic
    'road_length': 1000.0,  # Longer road
    'initial_spacing_range': [20, 40],  # Variable spacing
    'speed_variation': 0.3,  # 30% speed variation
    'slow_vehicle_probability': 0.4,  # 40% chance of slow vehicle
    'slow_speed_factor': 0.6,  # Slow vehicles at 60% of normal speed
}

# Overtaking behavior configuration - more conservative
class OvertakingConfig:
    """Configuration for autonomous overtaking behavior"""
    
    # Speed thresholds for overtaking decision
    MIN_SPEED_DIFFERENCE = 2.0     # Reduced threshold (m/s)
    CRITICAL_TIME_HEADWAY = 3.0    # Increased time headway (seconds)
    MIN_FOLLOWING_DISTANCE = 20.0  # Increased following distance (m)
    
    # Lane change parameters - more conservative
    LANE_CHANGE_COMFORT_ZONE = 60.0  # Reduced distance
    SAFETY_GAP_REQUIRED = 30.0       # Increased required gap (m)
    ABORT_DISTANCE = 15.0            # Increased abort distance
    
    # Overtaking maneuver timing
    MAX_OVERTAKING_TIME = 10.0     # Increased time limit (s)
    RETURN_LANE_DELAY = 4.0        # Longer delay before return
    
    # Comfort parameters
    MAX_LATERAL_ACCELERATION = 1.5  # Reduced lateral acceleration (m/s²)
    PREFERRED_OVERTAKING_SPEED = 20.0  # Reduced overtaking speed (m/s)

# Simulation configuration for dense overtaking behavior
SIMULATION_CONFIG = {
    'dt': 0.1,
    'sim_time': 35.0,  # Adequate time to show multiple overtaking maneuvers
    'num_vehicles': 6,
    'initial_spacing': 12.0,  # Closer initial spacing
    'lane_width': 3.5,
    'road_length': 200.0
}

# MPC Controller parameters (optimized to prevent overshooting)
MPC_CONFIG = {
    'prediction_horizon': 20,
    'control_horizon': 5,
    'max_steering': 0.25,  # Reduced for smoother lane changes
    'max_steering_rate': 0.12,  # Prevent abrupt steering
    'max_acceleration': 3.0,
    'lateral_weight': 60.0,  # High weight to prevent overshooting
    'steering_weight': 12.0,  # Penalize excessive steering
    'smoothness_weight': 8.0  # Encourage smooth control
}

# Vehicle behavior parameters (more aggressive overtaking)
VEHICLE_CONFIG = {
    'min_following_distance': 7.0,
    'overtaking_threshold': 10.0,
    'safety_margin': 6.0,
    'max_speed': 25.0,
    'min_speed': 10.0,
    'lane_change_duration': 2.5  # Faster but controlled lane changes
}

# Animation parameters (faster playback for dense overtaking)
ANIMATION_CONFIG = {
    'fps': 25,  # Higher frame rate for smooth playback
    'interval': 40,  # 40ms intervals for 2x speed
    'figure_size': (16, 8),  # Wider for better road view
    'dpi': 120,  # Good balance of quality and performance
    'playback_speed': 2.0,  # 2x real-time speed
    'max_frames': 300,  # Limit frames for performance
    'view_ahead_distance': 30,  # How far ahead to center view
    'trajectory_history_length': 30  # Number of past positions to show
}

# Export all configurations
__all__ = [
    'SimulationConfig',
    'VehicleConfig', 
    'RoadConfig',
    'ObstacleConfig',
    'MPCWeights',
    'BenchmarkConfig',
    'VisualizationConfig',
    'OvertakingConfig',  # ADD THIS
    'MULTI_VEHICLE_CONFIG',
    'SIMULATION_CONFIG',
    'MPC_CONFIG'
]
