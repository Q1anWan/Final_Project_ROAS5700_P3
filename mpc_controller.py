import numpy as np

class MPCController:
    def __init__(self):
        # Optimized parameters to prevent overshooting
        self.N = 20  # Increased prediction horizon
        self.dt = 0.1
        
        # Weight matrices (tuned to reduce overshooting)
        self.Q = np.diag([10.0, 50.0, 1.0])  # Increased lateral position weight
        self.R = np.diag([1.0, 10.0])  # Increased steering weight
        self.Qf = np.diag([10.0, 100.0, 1.0])  # Higher terminal lateral weight
        
        # Constraints (tighter for smoother lane changes)
        self.max_acceleration = 3.0
        self.max_steering = 0.3  # Reduced from 0.5 for gentler steering
        self.max_steering_rate = 0.15  # Added steering rate constraint
        
        # Lane change parameters
        self.lane_change_duration = 3.0  # Increased duration for smoother changes
        self.lateral_damping = 0.8  # Added damping factor

    def solve(self, current_state, target_lane, obstacles):
        # ...existing code...
        
        # Modified cost function with damping
        for k in range(self.N):
            # Lateral position error with damping
            lateral_error = y[k] - target_y
            if abs(lateral_error) < 0.5:  # Near target lane
                lateral_cost += self.lateral_damping * lateral_error**2
            else:
                lateral_cost += lateral_error**2
            
            # Steering smoothness constraint
            if k > 0:
                steering_rate = (delta[k] - delta[k-1]) / self.dt
                cost += 5.0 * steering_rate**2  # Penalize rapid steering changes
        
        # ...existing code...
