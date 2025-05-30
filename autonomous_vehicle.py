import numpy as np
from config import *

class AutonomousVehicle:
    """Autonomous vehicle with overtaking capabilities"""
    
    def __init__(self, vehicle_id, x, y, v, current_lane, target_speed, mpc_controller):
        self.id = vehicle_id
        
        # State variables [x, y, v, psi, delta, a]
        self.x = x              # Longitudinal position
        self.y = y              # Lateral position  
        self.v = v              # Velocity
        self.psi = 0.0          # Heading angle
        self.delta = 0.0        # Steering angle
        self.a = 0.0            # Acceleration
        
        # Lane tracking
        self.current_lane = current_lane
        self.original_lane = current_lane  # Remember original lane
        self.target_speed = target_speed
        
        # Vehicle parameters
        self.length = VehicleConfig.LENGTH
        self.width = VehicleConfig.WIDTH
        self.wheelbase = VehicleConfig.WHEELBASE
        
        # Control and decision making
        self.mpc_controller = mpc_controller
        self.overtaking_state = 'normal'  # 'normal', 'overtaking', 'returning'
        self.overtaking_start_time = None
        self.last_lane_change_time = 0.0
        
        # Performance tracking
        self.total_lane_changes = 0
        self.successful_overtakes = 0
        self.comfort_score = 0.0
        
        # Safety parameters
        self.safety_margin = OvertakingConfig.SAFETY_GAP_REQUIRED
        self.max_lateral_accel = OvertakingConfig.MAX_LATERAL_ACCELERATION
        
    def get_state(self):
        """Get current vehicle state vector"""
        return np.array([self.x, self.y, self.v, self.psi, self.delta, self.a])
    
    def update(self, control_input, dt):
        """Update vehicle state using bicycle model"""
        delta_dot, jerk = control_input
        
        # Update steering angle
        self.delta += delta_dot * dt
        self.delta = np.clip(self.delta, -VehicleConfig.MAX_STEERING_ANGLE, 
                           VehicleConfig.MAX_STEERING_ANGLE)
        
        # Update acceleration
        self.a += jerk * dt
        self.a = np.clip(self.a, VehicleConfig.MAX_DECELERATION, 
                        VehicleConfig.MAX_ACCELERATION)
        
        # Update velocity
        self.v += self.a * dt
        self.v = np.clip(self.v, VehicleConfig.MIN_SPEED, VehicleConfig.MAX_SPEED)
        
        # Bicycle model dynamics
        beta = np.arctan(0.5 * np.tan(self.delta))  # Slip angle
        
        # Update position and heading
        self.x += self.v * np.cos(self.psi + beta) * dt
        self.y += self.v * np.sin(self.psi + beta) * dt
        self.psi += (self.v / self.wheelbase) * np.sin(beta) * dt
        
        # Normalize heading
        self.psi = np.arctan2(np.sin(self.psi), np.cos(self.psi))
        
        # Update current lane based on position
        self.update_current_lane()
        
        # Update comfort score
        self.update_comfort_score(control_input, dt)
    
    def update_current_lane(self):
        """Update current lane based on lateral position"""
        lane_width = RoadConfig.LANE_WIDTH
        
        # Determine which lane the vehicle is in
        if self.y < lane_width:
            new_lane = 0
        else:
            new_lane = 1
            
        # Detect lane change
        if new_lane != self.current_lane:
            self.total_lane_changes += 1
            self.current_lane = new_lane
            
            # Update overtaking state
            if self.overtaking_state == 'normal' and new_lane != self.original_lane:
                self.overtaking_state = 'overtaking'
                self.overtaking_start_time = self.last_lane_change_time
            elif self.overtaking_state == 'overtaking' and new_lane == self.original_lane:
                self.overtaking_state = 'returning'
                self.successful_overtakes += 1
            elif self.overtaking_state == 'returning' and new_lane == self.original_lane:
                self.overtaking_state = 'normal'
    
    def update_comfort_score(self, control_input, dt):
        """Update passenger comfort score based on control inputs"""
        delta_dot, jerk = control_input
        
        # Lateral acceleration from steering
        lateral_accel = (self.v ** 2 * np.tan(self.delta)) / self.wheelbase
        
        # Comfort penalty (lower is better)
        comfort_penalty = (
            (lateral_accel / self.max_lateral_accel) ** 2 +
            (jerk / VehicleConfig.MAX_JERK) ** 2 +
            (delta_dot / VehicleConfig.MAX_STEERING_RATE) ** 2
        )
        
        # Update running average
        self.comfort_score = 0.95 * self.comfort_score + 0.05 * comfort_penalty
    
    def can_overtake_safely(self, other_vehicles, obstacles):
        """Check if overtaking is safe given current traffic"""
        target_lane = 1 - self.current_lane
        
        # Check minimum gap requirements
        required_gap = self.safety_margin
        vehicle_ahead = None
        vehicle_behind = None
        min_distance_ahead = float('inf')
        min_distance_behind = float('inf')
        
        # Check other vehicles in target lane
        for other in other_vehicles:
            if other.id == self.id or abs(other.current_lane - target_lane) > 0.1:
                continue
                
            distance = other.x - self.x
            
            if distance > 0 and distance < min_distance_ahead:
                min_distance_ahead = distance
                vehicle_ahead = other
            elif distance < 0 and abs(distance) < min_distance_behind:
                min_distance_behind = abs(distance)
                vehicle_behind = other
        
        # Check obstacles in target lane
        target_lane_y = target_lane * RoadConfig.LANE_WIDTH + RoadConfig.LANE_WIDTH/2
        for obs in obstacles:
            if abs(obs['y'] - target_lane_y) < RoadConfig.LANE_WIDTH/2:
                distance = obs['x'] - self.x
                if distance > 0 and distance < min_distance_ahead:
                    min_distance_ahead = distance
                elif distance < 0 and abs(distance) < min_distance_behind:
                    min_distance_behind = abs(distance)
        
        # Safety check
        safe_ahead = min_distance_ahead > required_gap
        safe_behind = min_distance_behind > required_gap
        
        return safe_ahead and safe_behind
    
    def get_performance_metrics(self):
        """Get vehicle performance metrics"""
        return {
            'total_lane_changes': self.total_lane_changes,
            'successful_overtakes': self.successful_overtakes,
            'comfort_score': self.comfort_score,
            'current_speed': self.v,
            'target_speed': self.target_speed,
            'speed_efficiency': self.v / self.target_speed if self.target_speed > 0 else 0,
            'position': [self.x, self.y],
            'current_lane': self.current_lane,
            'overtaking_state': self.overtaking_state
        }
    
    def __repr__(self):
        return f"Vehicle({self.id}: x={self.x:.1f}, y={self.y:.1f}, v={self.v:.1f}, lane={self.current_lane})"
