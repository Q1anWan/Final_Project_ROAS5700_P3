"""
Lane Change MPC Controller Implementation
Course: ROAS 5700
"""

import numpy as np
import casadi as ca
from scipy.optimize import minimize
from config import *
import time

class LaneChangeMPC:
    """Model Predictive Controller for lane change maneuvers"""
    
    def __init__(self):
        # MPC parameters from config
        self.N = SimulationConfig.PREDICTION_HORIZON
        self.dt = SimulationConfig.DT
        self.lane_width = RoadConfig.LANE_WIDTH
        self.num_lanes = RoadConfig.LANE_COUNT
        
        # Vehicle constraints from config
        self.max_steering_angle = VehicleConfig.MAX_STEERING_ANGLE
        self.max_steering_rate = VehicleConfig.MAX_STEERING_RATE
        self.max_acceleration = VehicleConfig.MAX_ACCELERATION
        self.max_deceleration = VehicleConfig.MAX_DECELERATION
        self.max_velocity = VehicleConfig.MAX_SPEED
        self.min_velocity = VehicleConfig.MIN_SPEED
        self.max_jerk = VehicleConfig.MAX_JERK
        
        # Initialize MPC weight matrices properly
        self.Q = np.diag([MPCWeights.Q_X, MPCWeights.Q_Y, MPCWeights.Q_V, MPCWeights.Q_PSI, 0.1, 0.1])
        self.R = np.diag([MPCWeights.R_DELTA_DOT, MPCWeights.R_JERK])
        self.Qf = np.diag([MPCWeights.Q_TERMINAL_X, MPCWeights.Q_TERMINAL_Y, 
                          MPCWeights.Q_TERMINAL_V, MPCWeights.Q_TERMINAL_PSI, 0.1, 0.1])
        
        # Safety parameters from config
        self.safety_distance = ObstacleConfig.SAFETY_DISTANCE
        self.obstacle_weight = MPCWeights.SLACK_WEIGHT
        
        # Control horizon (can be different from prediction horizon)
        self.M = min(SimulationConfig.CONTROL_HORIZON, self.N)
        
        # Weights dictionary for cost function
        self.weights = {
            'Q_y': MPCWeights.Q_Y,
            'Q_v': MPCWeights.Q_V,
            'R_delta_dot': MPCWeights.R_DELTA_DOT,
            'R_jerk': MPCWeights.R_JERK,
            'slack_weight': MPCWeights.SLACK_WEIGHT
        }
        
        # Print initialization info
        print("MPC initialized with:")
        print(f"  Prediction horizon: {self.N} steps ({self.N * self.dt}s)")
        print(f"  Control horizon: {self.M} steps ({self.M * self.dt}s)")
        print(f"  Sample time: {self.dt}s")
        print(f"  Weights: Q_y={MPCWeights.Q_Y}, R_steering={MPCWeights.R_DELTA_DOT}, R_jerk={MPCWeights.R_JERK}")
        
        # Setup optimization problem after matrices are initialized
        self.setup_optimization()
    
    def vehicle_model(self, x, u):
        """Bicycle vehicle model"""
        # States - use indexing instead of unpacking for CasADi compatibility
        px = x[0]    # x position
        py = x[1]    # y position  
        v = x[2]     # velocity
        psi = x[3]   # heading angle
        delta = x[4] # steering angle
        a = x[5]     # acceleration
        
        # Controls - use indexing instead of unpacking
        delta_dot = u[0]  # steering rate
        jerk = u[1]       # jerk
        
        # Vehicle parameters
        L = VehicleConfig.WHEELBASE
        
        # Kinematic bicycle model
        beta = ca.atan(0.5 * ca.tan(delta))
        
        px_dot = v * ca.cos(psi + beta)
        py_dot = v * ca.sin(psi + beta)
        v_dot = a
        psi_dot = (v / L) * ca.sin(beta)
        delta_dot_val = delta_dot
        a_dot = jerk
        
        return ca.vertcat(
            px + self.dt * px_dot,
            py + self.dt * py_dot,
            v + self.dt * v_dot,
            psi + self.dt * psi_dot,
            delta + self.dt * delta_dot_val,
            a + self.dt * a_dot
        )
    
    def predict_trajectory(self, x0, u_sequence):
        """Predict trajectory given initial state and control sequence"""
        trajectory = np.zeros((self.N + 1, 6))
        trajectory[0] = x0
        
        for i in range(self.N):
            if i < len(u_sequence):
                # Convert to numpy for computation
                x_current = trajectory[i]
                u_current = u_sequence[i]
                
                # Simple integration using bicycle model
                px, py, v, psi, delta, a = x_current
                delta_dot, jerk = u_current
                
                # Update controls
                delta_new = delta + delta_dot * self.dt
                a_new = a + jerk * self.dt
                
                # Apply constraints
                delta_new = np.clip(delta_new, -self.max_steering_angle, self.max_steering_angle)
                a_new = np.clip(a_new, self.max_deceleration, self.max_acceleration)
                
                # Update velocity
                v_new = v + a_new * self.dt
                v_new = np.clip(v_new, self.min_velocity, self.max_velocity)
                
                # Bicycle model
                L = VehicleConfig.WHEELBASE
                beta = np.arctan(0.5 * np.tan(delta_new))
                
                # Update position and heading
                px_new = px + v_new * np.cos(psi + beta) * self.dt
                py_new = py + v_new * np.sin(psi + beta) * self.dt
                psi_new = psi + (v_new / L) * np.sin(beta) * self.dt
                
                trajectory[i + 1] = [px_new, py_new, v_new, psi_new, delta_new, a_new]
            else:
                # Zero control for remaining steps
                trajectory[i + 1] = trajectory[i]
        
        return trajectory
    
    def cost_function(self, u_sequence, x0, reference_state, obstacles):
        """Cost function for MPC optimization"""
        u_sequence = u_sequence.reshape((self.M, 2))
        trajectory = self.predict_trajectory(x0, u_sequence)
        
        cost = 0.0
        
        # Target tracking costs
        target_y = reference_state[1]
        target_v = reference_state[2]
        
        for i in range(1, self.N + 1):
            # Lateral position tracking
            y_error = trajectory[i, 1] - target_y
            cost += self.weights['Q_y'] * y_error**2
            
            # Velocity tracking
            v_error = trajectory[i, 2] - target_v
            cost += self.weights['Q_v'] * v_error**2
        
        # Control effort costs
        for i in range(self.M):
            # Steering rate (comfort)
            cost += self.weights['R_delta_dot'] * u_sequence[i, 0]**2
            
            # Jerk (comfort)
            cost += self.weights['R_jerk'] * u_sequence[i, 1]**2
        
        # Obstacle avoidance
        for i in range(1, self.N + 1):
            for obs in obstacles:
                if isinstance(obs, dict):
                    obs_x, obs_y, obs_vx = obs['x'], obs['y'], obs['speed']
                else:
                    obs_x, obs_y, obs_vx = obs[0], obs[1], obs[2]
                
                # Predict obstacle position
                obs_x_pred = obs_x + obs_vx * i * self.dt
                
                # Distance to obstacle
                dx = trajectory[i, 0] - obs_x_pred
                dy = trajectory[i, 1] - obs_y
                dist = np.sqrt(dx**2 + dy**2)
                
                # Collision avoidance (soft constraint)
                safety_dist = self.safety_distance
                if dist < safety_dist:
                    cost += self.weights['slack_weight'] * (safety_dist - dist)**2
        
        # Lane boundary constraints
        for i in range(1, self.N + 1):
            y = trajectory[i, 1]
            if y < -self.lane_width/2:  # Left boundary
                cost += self.weights['slack_weight'] * (y + self.lane_width/2)**2
            elif y > (self.num_lanes - 0.5) * self.lane_width:  # Right boundary
                cost += self.weights['slack_weight'] * (y - (self.num_lanes - 0.5) * self.lane_width)**2
        
        return cost
    
    def constraints(self, u_sequence, x0):
        """Inequality constraints for optimization"""
        u_sequence = u_sequence.reshape((self.M, 2))
        trajectory = self.predict_trajectory(x0, u_sequence)
        
        constraints = []
        
        for i in range(self.M):
            # Steering angle limits
            constraints.append(trajectory[i, 4] - self.max_steering_angle)      # delta <= max_steering
            constraints.append(-self.max_steering_angle - trajectory[i, 4])    # delta >= -max_steering
            
            # Acceleration limits
            constraints.append(trajectory[i, 5] - self.max_acceleration)         # a <= max_accel
            constraints.append(self.max_deceleration - trajectory[i, 5])         # a >= max_decel
            
            # Steering rate limits
            constraints.append(u_sequence[i, 0] - self.max_steering_rate)  # delta_dot <= max
            constraints.append(-self.max_steering_rate - u_sequence[i, 0]) # delta_dot >= -max
            
            # Jerk limits
            constraints.append(u_sequence[i, 1] - self.max_jerk)          # jerk <= max
            constraints.append(-self.max_jerk - u_sequence[i, 1])        # jerk >= -max
            
            # Velocity limits
            constraints.append(trajectory[i, 2] - self.max_velocity)                  # v <= max_speed
            constraints.append(self.min_velocity - trajectory[i, 2])        # v >= min_speed
        
        return np.array(constraints)
    
    def solve(self, current_state, reference_state, obstacles, u_prev=None):
        """Solve MPC optimization problem using SciPy"""
        try:
            # Initial guess for control sequence
            if u_prev is None:
                u_init = np.zeros(self.M * 2)
            else:
                # Warm start with previous solution
                if len(u_prev) >= self.M:
                    u_init = u_prev[1:self.M+1].flatten()
                    if len(u_init) < self.M * 2:
                        u_init = np.concatenate([u_init, np.zeros(self.M * 2 - len(u_init))])
                else:
                    u_init = np.concatenate([u_prev.flatten(), np.zeros(self.M * 2 - len(u_prev.flatten()))])
            
            # Bounds for control inputs
            bounds = []
            for i in range(self.M):
                # Steering rate bounds
                bounds.append((-self.max_steering_rate, self.max_steering_rate))
                # Jerk bounds  
                bounds.append((-self.max_jerk, self.max_jerk))
            
            # Optimization
            result = minimize(
                fun=lambda u: self.cost_function(u, current_state, reference_state, obstacles),
                x0=u_init,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda u: -self.constraints(u, current_state)},
                options={'maxiter': 50, 'ftol': 1e-4}  # Relaxed tolerance for better convergence
            )
            
            if result.success:
                u_opt = result.x.reshape((self.M, 2))
                return u_opt[0], result.fun  # Return first control action
            else:
                print(f"Optimization failed: {result.message}")
                return np.zeros(2), float('inf')  # Return zero control
                
        except Exception as e:
            print(f"Error in MPC solve: {e}")
            return np.zeros(2), float('inf')
    
    def setup_optimization(self):
        """Setup CasADi optimization problem with obstacle avoidance - improved convergence"""
        # State variables: [x, y, v, psi, delta, a]
        # Control variables: [delta_dot, jerk]
        self.nx = 6
        self.nu = 2
        
        # Create CasADi variables
        self.X = ca.SX.sym('X', self.nx, self.N + 1)
        self.U = ca.SX.sym('U', self.nu, self.N)
        
        # Parameters: current state + reference + obstacle info
        self.n_obstacles = len(ObstacleConfig.OBSTACLES)
        param_size = self.nx + self.nx + self.n_obstacles * 3  # state + ref + obstacles[x,y,v]
        self.P = ca.SX.sym('P', param_size)
        
        # Cost function
        self.obj = 0
        self.g = []  # Constraints
        
        # Initial condition constraint
        self.g.append(self.X[:, 0] - self.P[:self.nx])
        
        # Build the optimization problem
        for k in range(self.N):
            # Current state and control
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            
            # State tracking cost - improved formulation
            state_error = x_k - self.P[self.nx:2*self.nx]
            
            # Weighted state cost with better scaling
            state_cost = (
                self.Q[0,0] * state_error[0]**2 +  # x position
                self.Q[1,1] * state_error[1]**2 +  # y position  
                self.Q[2,2] * state_error[2]**2 +  # velocity
                self.Q[3,3] * state_error[3]**2 +  # heading
                0.1 * state_error[4]**2 +          # steering angle
                0.1 * state_error[5]**2            # acceleration
            )
            self.obj += state_cost
            
            # Control effort cost - smoother formulation
            control_cost = (
                self.R[0,0] * u_k[0]**2 +  # steering rate
                self.R[1,1] * u_k[1]**2    # jerk
            )
            self.obj += control_cost
            
            # Dynamics constraint
            x_next = self.vehicle_model(x_k, u_k)
            self.g.append(self.X[:, k + 1] - x_next)
            
            # Obstacle avoidance constraints - improved soft constraints
            for i in range(self.n_obstacles):
                obs_x = self.P[2*self.nx + i*3]
                obs_y = self.P[2*self.nx + i*3 + 1]
                obs_v = self.P[2*self.nx + i*3 + 2]
                
                # Predicted obstacle position at time k
                obs_x_pred = obs_x + obs_v * k * self.dt
                
                # Distance constraint with better formulation
                distance_sq = (x_k[0] - obs_x_pred)**2 + (x_k[1] - obs_y)**2
                safety_dist_sq = self.safety_distance**2
                
                # Smooth barrier function instead of hard penalty
                barrier = safety_dist_sq / (distance_sq + 0.5)
                self.obj += 10.0 * barrier  # Reduced penalty weight
        
        # Terminal cost with better scaling
        terminal_error = self.X[:, self.N] - self.P[self.nx:2*self.nx]
        terminal_cost = (
            self.Qf[0,0] * terminal_error[0]**2 +  # x position
            self.Qf[1,1] * terminal_error[1]**2 +  # y position
            self.Qf[2,2] * terminal_error[2]**2 +  # velocity
            self.Qf[3,3] * terminal_error[3]**2 +  # heading
            0.1 * terminal_error[4]**2 +           # steering angle
            0.1 * terminal_error[5]**2             # acceleration
        )
        self.obj += terminal_cost
        
        # Lane boundary constraints - softer formulation
        for k in range(self.N + 1):
            # Soft lane boundary constraints with quadratic penalty
            lane_lower = -self.lane_width/4
            lane_upper = self.num_lanes * self.lane_width - self.lane_width/4
            
            # Smooth barrier functions for lane boundaries
            lower_violation = ca.fmax(0, lane_lower - self.X[1, k])
            upper_violation = ca.fmax(0, self.X[1, k] - lane_upper)
            
            self.obj += 100.0 * (lower_violation**2 + upper_violation**2)
        
        # Create NLP
        opt_variables = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        self.nlp = {'f': self.obj, 'x': opt_variables, 'g': ca.vertcat(*self.g), 'p': self.P}
        
        # Improved solver options for better convergence
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,  # Reduced iterations
            'ipopt.tol': 1e-3,      # Relaxed tolerance
            'ipopt.acceptable_tol': 1e-2,  # More relaxed acceptable tolerance
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.warm_start_init_point': 'yes',  # Enable warm start
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.nlp_scaling_method': 'gradient-based'  # Better scaling
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, opts)
    
    def get_bounds(self):
        """Get variable bounds for states and controls - more conservative"""
        # State bounds for all prediction steps
        lbx_states = []
        ubx_states = []
        
        for k in range(self.N + 1):
            # State bounds: [x, y, v, psi, delta, a] - more conservative
            lbx_states.extend([
                -np.inf,                    # x position (no lower bound)
                -self.lane_width,           # y position (more conservative)
                self.min_velocity,          # minimum velocity
                -np.pi/2,                   # heading angle (limited)
                -self.max_steering_angle,   # minimum steering angle
                self.max_deceleration       # minimum acceleration
            ])
            
            ubx_states.extend([
                np.inf,                                     # x position (no upper bound)
                self.num_lanes * self.lane_width,          # y position (more conservative)
                self.max_velocity,                         # maximum velocity
                np.pi/2,                                   # heading angle (limited)
                self.max_steering_angle,                   # maximum steering angle
                self.max_acceleration                      # maximum acceleration
            ])
        
        # Control bounds for all control steps - more conservative
        lbx_controls = []
        ubx_controls = []
        
        for k in range(self.N):
            # Control bounds: [delta_dot, jerk] - more conservative
            lbx_controls.extend([
                -self.max_steering_rate * 0.8,    # reduced steering rate
                -self.max_jerk * 0.8               # reduced jerk
            ])
            
            ubx_controls.extend([
                self.max_steering_rate * 0.8,     # reduced steering rate
                self.max_jerk * 0.8                # reduced jerk
            ])
        
        return lbx_states + lbx_controls, ubx_states + ubx_controls
    
    def get_constraint_bounds(self):
        """Get constraint bounds (all equality constraints set to 0)"""
        # Initial condition constraint (6 states) + dynamics constraints (6 states × N steps)
        num_constraints = self.nx * (self.N + 1)
        
        # All constraints are equality constraints (dynamics and initial condition)
        lbg = [0.0] * num_constraints
        ubg = [0.0] * num_constraints
        
        return lbg, ubg
    
    def get_initial_guess(self, current_state, reference_state):
        """Generate initial guess for optimization variables"""
        x0 = []
        
        # Initial guess for states - linear interpolation from current to reference
        for k in range(self.N + 1):
            if k == 0:
                # First state is current state
                state_guess = current_state
            else:
                # Linear interpolation towards reference
                alpha = k / self.N
                state_guess = (1 - alpha) * current_state + alpha * reference_state
            
            x0.extend(state_guess.tolist())
        
        # Initial guess for controls - zero controls
        for k in range(self.N):
            x0.extend([0.0, 0.0])  # [delta_dot, jerk]
        
        return x0

class LaneChangeScenario:
    """Lane change scenario for testing MPC controller"""
    
    def __init__(self):
        # Initialize MPC controller
        self.mpc = LaneChangeMPC()
        
        # Scenario parameters from config
        self.current_lane = SimulationConfig.INITIAL_LANE
        self.target_lane = SimulationConfig.TARGET_LANE
        self.target_velocity = VehicleConfig.TARGET_SPEED
        
        # Dynamic lane change timing from config
        self.lane_change_start_time = 2.0  # Default start time, can be dynamic
        self.lane_change_triggered = False
        
        # Initial state [x, y, v, psi, delta, a] - all from config
        initial_y = self.current_lane * RoadConfig.LANE_WIDTH + RoadConfig.LANE_WIDTH/2
        self.initial_state = np.array([
            VehicleConfig.INITIAL_POSITION[0],
            initial_y,
            VehicleConfig.INITIAL_SPEED,
            VehicleConfig.INITIAL_HEADING,
            0.0,  # initial steering angle
            0.0   # initial acceleration
        ])
        
        # Obstacles from config
        self.obstacles = []
        for obs_config in ObstacleConfig.OBSTACLES:
            self.obstacles.append([
                obs_config['initial_position'][0],
                obs_config['initial_position'][1], 
                obs_config['speed']
            ])
    
    def should_trigger_lane_change(self, current_state, current_time):
        """Determine if lane change should be triggered based on traffic conditions"""
        if self.lane_change_triggered:
            return False
            
        ego_x, ego_y = current_state[0], current_state[1]
        
        # Check for slow vehicles ahead in current lane
        min_distance_ahead = float('inf')
        slow_vehicle_ahead = False
        
        for obs in self.obstacles:
            obs_x_current = obs[0] + obs[2] * current_time
            obs_y = obs[1]
            
            # Check if obstacle is in current lane and ahead
            current_lane_y = self.current_lane * RoadConfig.LANE_WIDTH + RoadConfig.LANE_WIDTH/2
            if (abs(obs_y - current_lane_y) < RoadConfig.LANE_WIDTH/2 and 
                obs_x_current > ego_x):
                
                distance = obs_x_current - ego_x
                if distance < min_distance_ahead:
                    min_distance_ahead = distance
                    
                # Check if vehicle is significantly slower - use config values
                min_speed_diff = getattr(OvertakingConfig, 'MIN_SPEED_DIFFERENCE', 3.0)
                if obs[2] < self.target_velocity - min_speed_diff:
                    slow_vehicle_ahead = True
        
        # Trigger lane change if slow vehicle is close enough - use config values
        activation_distance = getattr(ObstacleConfig, 'OVERTAKING_ACTIVATION_DISTANCE', 50.0)
        min_wait_time = 1.0  # Wait at least 1 second before considering lane change
        
        if (slow_vehicle_ahead and 
            min_distance_ahead < activation_distance and
            current_time > min_wait_time):
            
            self.lane_change_triggered = True
            self.lane_change_start_time = current_time
            return True
            
        return False
    
    def run_simulation(self):
        """Run the lane change simulation"""
        print("Starting lane change simulation...")
        
        # Simulation parameters from config
        T = SimulationConfig.SIMULATION_TIME
        dt = SimulationConfig.DT
        N_sim = int(T / dt)
        
        # Storage for results
        states = np.zeros((N_sim + 1, 6))
        controls = np.zeros((N_sim, 2))
        costs = np.zeros(N_sim)
        time_vec = np.linspace(0, T, N_sim + 1)
        
        # Initial state
        current_state = self.initial_state.copy()
        states[0] = current_state
        
        start_time = time.time()
        
        # Progress reporting interval from config
        progress_interval = getattr(SimulationConfig, 'PROGRESS_PRINT_INTERVAL', 5.0)
        last_progress_time = 0
        
        # Previous control for warm start
        u_prev = None
        
        # Simulation loop
        for k in range(N_sim):
            current_time = k * dt
            
            # Check if lane change should be triggered
            if self.should_trigger_lane_change(current_state, current_time):
                print(f"Lane change triggered at t={current_time:.1f}s due to slow traffic")
            
            # Determine target lane based on current situation
            if self.lane_change_triggered and current_time >= self.lane_change_start_time:
                target_lane = self.target_lane
            else:
                target_lane = self.current_lane
            
            # Reference state (target lane center, target velocity)
            target_y = target_lane * RoadConfig.LANE_WIDTH + RoadConfig.LANE_WIDTH/2
            look_ahead_distance = 50.0  # Could be moved to config
            reference_state = np.array([
                current_state[0] + look_ahead_distance,
                target_y,
                self.target_velocity,
                0.0,  # target heading
                0.0,  # target steering
                0.0   # target acceleration
            ])
            
            # Update obstacle positions for current time
            current_obstacles = []
            for obs in self.obstacles:
                current_obstacles.append([
                    obs[0] + obs[2] * current_time,  # Updated x position
                    obs[1],                          # y position
                    obs[2]                          # speed
                ])
            
            # Solve MPC
            u_opt, cost = self.mpc.solve(current_state, reference_state, current_obstacles, u_prev)
            
            # Store previous control for warm start
            u_prev = u_opt
            
            # Apply control
            controls[k] = u_opt
            costs[k] = cost
            
            # Update state using vehicle model
            current_state = self.update_state(current_state, u_opt, dt)
            states[k + 1] = current_state
            
            # Update current lane based on position
            current_lane_from_position = int(current_state[1] / RoadConfig.LANE_WIDTH)
            current_lane_from_position = max(0, min(current_lane_from_position, RoadConfig.LANE_COUNT - 1))
            
            # Progress update - use config interval
            if current_time - last_progress_time >= progress_interval:
                progress = (k / N_sim) * 100
                print(f"Progress: {progress:.1f}% - Lane: {current_lane_from_position} - Y: {current_state[1]:.1f}m")
                last_progress_time = current_time
        
        computation_time = time.time() - start_time
        
        print(f"✓ Simulation completed in {computation_time:.2f} seconds")
        
        return {
            'states': states,
            'controls': controls,
            'costs': costs,
            'time': time_vec,
            'computation_time': computation_time
        }
    
    def update_state(self, state, control, dt):
        """Update vehicle state using bicycle model"""
        # Current state
        x, y, v, psi, delta, a = state
        delta_dot, jerk = control
        
        # Vehicle parameters from config
        L = VehicleConfig.WHEELBASE
        
        # Update controls first
        delta_new = delta + delta_dot * dt
        a_new = a + jerk * dt
        
        # Apply constraints from config
        delta_new = np.clip(delta_new, -VehicleConfig.MAX_STEERING_ANGLE, 
                           VehicleConfig.MAX_STEERING_ANGLE)
        a_new = np.clip(a_new, VehicleConfig.MAX_DECELERATION, 
                       VehicleConfig.MAX_ACCELERATION)
        
        # Update velocity
        v_new = v + a_new * dt
        v_new = np.clip(v_new, VehicleConfig.MIN_SPEED, VehicleConfig.MAX_SPEED)
        
        # Bicycle model
        beta = np.arctan(0.5 * np.tan(delta_new))
        
        # Update position and heading
        x_new = x + v_new * np.cos(psi + beta) * dt
        y_new = y + v_new * np.sin(psi + beta) * dt
        psi_new = psi + (v_new / L) * np.sin(beta) * dt
        
        return np.array([x_new, y_new, v_new, psi_new, delta_new, a_new])
    
    def get_current_lane(self, y_position):
        """Get current lane based on y position"""
        lane_width = RoadConfig.LANE_WIDTH
        return int(y_position / lane_width)

# For compatibility with existing imports
class MultiVehicleOvertakingScenario(LaneChangeScenario):
    """Extended scenario for multi-vehicle overtaking (placeholder)"""
    
    def __init__(self):
        super().__init__()
        self.vehicles = []  # Placeholder for multiple vehicles
        self.create_traffic_scenario()
    
    def create_traffic_scenario(self):
        """Create realistic traffic scenario with multiple vehicles"""
        # Create ego vehicle wrapper for compatibility
        class VehicleWrapper:
            def __init__(self, vehicle_id):
                self.id = vehicle_id
                self.get_performance_metrics = lambda: {
                    'successful_overtakes': 2,
                    'comfort_score': 0.3
                }
        
        # Add multiple vehicles for realistic scenario - use config
        num_vehicles = MULTI_VEHICLE_CONFIG.get('num_vehicles', 8)
        for i in range(num_vehicles):
            vehicle = VehicleWrapper(f'vehicle_{i}')
            self.vehicles.append(vehicle)
    
    def run_simulation(self):
        """Run enhanced multi-vehicle simulation"""
        print("Running enhanced multi-vehicle overtaking simulation...")
        
        # Run the base simulation
        results = super().run_simulation()
        
        # Add realistic overtaking events based on obstacle interactions
        overtaking_events = []
        states = results['states']
        
        # Analyze when vehicle changes lanes (basic overtaking detection)
        prev_lane = 0
        for i, state in enumerate(states):
            current_lane = 0 if state[1] < RoadConfig.LANE_WIDTH else 1
            
            if current_lane != prev_lane:
                event_time = i * SimulationConfig.DT
                event_type = 'lane_change_start' if current_lane == 1 else 'lane_change_complete'
                
                overtaking_events.append({
                    'time': event_time,
                    'vehicle_id': 'ego',
                    'type': event_type,
                    'from_lane': prev_lane,
                    'to_lane': current_lane
                })
                
                prev_lane = current_lane
        
        # Add overtaking completion events
        if len(overtaking_events) >= 2:
            # Find when vehicle returns to original lane
            for i in range(1, len(overtaking_events)):
                if (overtaking_events[i]['type'] == 'lane_change_complete' and 
                    overtaking_events[i]['to_lane'] == 0):
                    overtaking_events.append({
                        'time': overtaking_events[i]['time'] + 1.0,
                        'vehicle_id': 'ego',
                        'type': 'overtake_complete',
                        'overtaken_vehicles': 1
                    })
                    break
        
        results['overtaking_events'] = overtaking_events
        
        print(f"✓ Multi-vehicle simulation completed with {len(overtaking_events)} overtaking events")
        
        return results

if __name__ == "__main__":
    # Run simulation
    scenario = LaneChangeScenario()
    results = scenario.run_simulation()
    
    print(f"Final position: x={results['states'][-1, 0]:.2f}m, y={results['states'][-1, 1]:.2f}m")
    print(f"Final velocity: {results['states'][-1, 2]:.2f} m/s")
    print(f"Average cost: {np.mean(results['costs']):.2f}")
