"""
Visualization tools for Lane Change MPC simulation
Course: ROAS 5700
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import os
from config import *

class LaneChangeVisualizer:
    """Enhanced visualization tools for lane change MPC simulation with improved readability"""
    
    def __init__(self, results, scenario):
        self.results = results
        self.scenario = scenario
        
        # Enhanced plot settings for better readability - IMPROVED SPACING
        plt.rcParams.update({
            'font.size': 12,  # Reduced from 14 for better fit
            'figure.dpi': VisualizationConfig.DPI,
            'savefig.dpi': VisualizationConfig.DPI,
            'font.family': 'DejaVu Sans',
            'axes.linewidth': 1.5,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.5,
            'figure.constrained_layout.use': False,  # Disable for explicit control
            'axes.titlepad': 18,  # More space for titles
            'axes.labelpad': 12   # More space for labels
        })
        
        # Enhanced color scheme for better distinction
        self.colors = {
            'ego_vehicle': '#FF4444',      # Bright red
            'ego_trajectory': '#FF8888',   # Light red
            'obstacles': '#2E86AB',        # Blue
            'lane_lines': '#A8DADC',       # Light blue-gray
            'lane_fill': '#F1FAEE',        # Very light green
            'target_lane': '#457B9D',      # Darker blue
            'safety_zone': '#E63946',      # Red for safety
            'road': '#264653'              # Dark green-gray
        }

        # Create output directory
        self.output_dir = os.path.abspath(SimulationConfig.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Visualizer initialized - Output directory: {self.output_dir}")
    
    def plot_trajectory_overview(self):
        """Plot comprehensive trajectory overview with improved readability and spacing"""
        states = self.results['states']
        time_vec = self.results['time']
        
        # Create figure with explicit spacing control - make it even larger
        fig = plt.figure(figsize=(28, 20))  # Increased from (24, 18)
        
        # Use explicit subplot adjustment with better proportions for spatial plot
        fig.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.06, hspace=0.25, wspace=0.20)
        
        # Create GridSpec with better height ratios - give more space to spatial plot
        gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1, 1])  # Spatial plot gets 2.5x height
        
        # Create subplots with better arrangement
        ax1 = fig.add_subplot(gs[0, :])  # Top subplot spans both columns (spatial trajectory)
        ax2 = fig.add_subplot(gs[1, 0])  # Middle left (lateral position)
        ax3 = fig.add_subplot(gs[1, 1])  # Middle right (velocity/control)
        
        # Use fig.text for title placement with much more vertical space
        fig.text(0.5, 0.93, 'Lane Change MPC - Trajectory Overview', 
                fontsize=24, fontweight='bold', ha='center', va='center')
        
        # 1. Spatial trajectory plot with enhanced visualization and more space - PASS FIG
        self._plot_spatial_trajectory(ax1, states, fig, enhanced=True)
        
        # 2. Lateral position vs time with lane boundaries
        self._plot_lateral_position_improved(ax2, states, time_vec)
        
        # 3. Velocity and control overview combined
        self._plot_velocity_and_control_combined(ax3, states, time_vec)
        
        # Save with better spacing
        filepath = os.path.join(self.output_dir, 'trajectory_overview.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', 
                   pad_inches=0.5)  # More padding
        print(f"✓ Trajectory overview saved to: {filepath}")
        plt.show()
    
    def _plot_spatial_trajectory(self, ax, states, fig, enhanced=True):
        """Enhanced spatial trajectory plot with better spacing and no overlapping text"""
        # Road setup with improved styling
        lane_width = RoadConfig.LANE_WIDTH
        num_lanes = RoadConfig.LANE_COUNT
        road_width = num_lanes * lane_width
        
        # Plot road background with more generous x-range
        x_range = [states[0, 0] - 30, states[-1, 0] + 80]  # More space on both sides
        ax.fill_between(x_range, [0, 0], [road_width, road_width], 
                       color=self.colors['lane_fill'], alpha=0.3, label='Road')
        
        # Enhanced lane lines
        for i in range(num_lanes + 1):
            y_pos = i * lane_width
            line_style = '-' if i in [0, num_lanes] else '--'
            line_width = 3 if i in [0, num_lanes] else 2
            ax.plot(x_range, [y_pos, y_pos], line_style, 
                   color=self.colors['lane_lines'], linewidth=line_width, alpha=0.8)
        
        # Lane labels positioned lower to avoid legend overlap
        for i in range(num_lanes):
            lane_center = i * lane_width + lane_width/2
            # Position labels lower and more to the left to avoid legend overlap
            ax.text(x_range[0] + 25, lane_center - 0.5, f'Lane {i}', 
                   fontsize=14, fontweight='bold', ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           alpha=0.95, edgecolor='gray', linewidth=1.5))
        
        # Plot obstacles with improved visibility and tracking
        self._plot_obstacles_enhanced(ax, states, x_range)
        
        # Enhanced ego vehicle trajectory
        ego_x, ego_y = states[:, 0], states[:, 1]
        
        # Trajectory line with gradient coloring
        scatter = ax.scatter(ego_x, ego_y, c=np.arange(len(ego_x)), 
                           cmap='viridis', s=60, alpha=0.8, 
                           edgecolors='black', linewidth=0.8, label='Ego Trajectory')
        
        # Position colorbar lower to avoid overlap with x-axis label
        cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', 
                           shrink=0.6, pad=0.12, aspect=40)  # Increased pad from 0.08 to 0.12
        cbar.set_label('Time Progression', fontsize=14, labelpad=15)  # Increased labelpad
        
        # Start and end markers with larger size
        ax.plot(ego_x[0], ego_y[0], 'go', markersize=18, markeredgewidth=3, 
               markeredgecolor='black', label='Start', zorder=10)
        ax.plot(ego_x[-1], ego_y[-1], 'rs', markersize=18, markeredgewidth=3, 
               markeredgecolor='black', label='End', zorder=10)
        
        # Enhanced styling with better spacing
        ax.set_xlim(x_range)
        ax.set_ylim(-1.5, road_width + 1.5)  # More vertical space
        ax.set_xlabel('Longitudinal Position (m)', fontsize=16, fontweight='bold', labelpad=20)  # Increased labelpad
        ax.set_ylabel('Lateral Position (m)', fontsize=16, fontweight='bold', labelpad=15)
        ax.set_title('Spatial Trajectory with Obstacles', fontsize=18, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Position legend at upper right to avoid overlap with lane labels
        legend = ax.legend(fontsize=12, loc='upper right', 
                         frameon=True, fancybox=True, shadow=True, ncol=2,
                         bbox_to_anchor=(0.99, 0.99))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        
        # Don't force equal aspect ratio to allow more horizontal space
        # ax.set_aspect('equal', adjustable='box')  # Remove this line
    
    def _plot_obstacles_enhanced(self, ax, states, x_range):
        """Plot obstacles with enhanced visibility and proper tracking"""
        if not hasattr(self.scenario, 'obstacles') or not self.scenario.obstacles:
            return
        
        time_vec = self.results['time']
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.scenario.obstacles)))
        
        for i, obs in enumerate(self.scenario.obstacles):
            obs_x_initial, obs_y, obs_speed = obs[0], obs[1], obs[2]
            
            # Calculate obstacle trajectory over entire simulation
            obs_x_trajectory = []
            obs_times = []
            
            for t in time_vec:
                obs_x_current = obs_x_initial + obs_speed * t
                # Only show obstacle if it's within reasonable range
                if x_range[0] - 50 <= obs_x_current <= x_range[1] + 50:
                    obs_x_trajectory.append(obs_x_current)
                    obs_times.append(t)
            
            if obs_x_trajectory:
                # Plot obstacle trajectory line
                ax.plot(obs_x_trajectory, [obs_y] * len(obs_x_trajectory), 
                       '--', color=colors[i], linewidth=3, alpha=0.7,
                       label=f'Obstacle {i+1} Path')
                
                # Plot obstacle at multiple time points for visibility
                sample_indices = np.linspace(0, len(obs_x_trajectory)-1, 5, dtype=int)
                for j, idx in enumerate(sample_indices):
                    alpha = 0.3 + 0.7 * (j / len(sample_indices))  # Fade effect
                    size = 10 + 6 * (j / len(sample_indices))      # Larger size increase
                    
                    ax.plot(obs_x_trajectory[idx], obs_y, 'o', 
                           color=colors[i], markersize=size, alpha=alpha,
                           markeredgewidth=1.5, markeredgecolor='black')
                
                # Add obstacle label with better positioning to avoid overlap
                mid_idx = len(obs_x_trajectory) // 2
                if mid_idx < len(obs_x_trajectory):
                    # Position labels higher to avoid overlap with trajectory
                    ax.annotate(f'Obs{i+1}\n{obs_speed:.1f}m/s', 
                              xy=(obs_x_trajectory[mid_idx], obs_y),
                              xytext=(15, 35), textcoords='offset points',  # More offset
                              fontsize=11, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.4', 
                                       facecolor=colors[i], alpha=0.8,
                                       edgecolor='black', linewidth=1),
                              arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    def _plot_lateral_position_improved(self, ax, states, time_vec):
        """Improved lateral position plot with better spacing"""
        ego_y = states[:, 1]
        lane_width = RoadConfig.LANE_WIDTH
        
        # Plot lateral position with thicker line
        ax.plot(time_vec, ego_y, color=self.colors['ego_vehicle'], 
               linewidth=4, label='Ego Vehicle', alpha=0.8)
        
        # Lane boundaries with better styling
        for i in range(RoadConfig.LANE_COUNT + 1):
            y_pos = i * lane_width
            line_style = '-' if i in [0, RoadConfig.LANE_COUNT] else '--'
            line_width = 3 if i in [0, RoadConfig.LANE_COUNT] else 2
            ax.axhline(y=y_pos, color=self.colors['lane_lines'], 
                      linestyle=line_style, linewidth=line_width, alpha=0.7)
        
        # Lane centers with better label positioning
        for i in range(RoadConfig.LANE_COUNT):
            lane_center = i * lane_width + lane_width/2
            ax.axhline(y=lane_center, color=self.colors['target_lane'], 
                      linestyle=':', linewidth=1.5, alpha=0.6)
            
            # Position labels to avoid overlap
            label_x = time_vec[-1] * 0.85  # Move labels further left
            ax.text(label_x, lane_center + 0.3, f'Lane {i}', 
                   fontsize=10, ha='right', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           alpha=0.8, edgecolor='gray'))
        
        # Better axis formatting
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Lateral Position (m)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Lateral Position vs Time', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=11, loc='best')
        
        # Set better y-limits to avoid cramping
        ax.set_ylim(-0.5, RoadConfig.LANE_COUNT * lane_width + 0.5)
    
    def _plot_velocity_and_control_combined(self, ax, states, time_vec):
        """Combined velocity and control plot with dual y-axis"""
        ego_v = states[:, 2]
        target_v = self.scenario.target_velocity
        controls = self.results['controls']
        
        # Primary y-axis for velocity
        color1 = 'tab:blue'
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Velocity (m/s)', color=color1, fontsize=12, fontweight='bold', labelpad=10)
        
        line1 = ax.plot(time_vec, ego_v, color=color1, linewidth=3, 
                       label='Actual Velocity', alpha=0.8)
        ax.axhline(y=target_v, color=color1, linestyle='--', linewidth=2, 
                  alpha=0.7, label='Target Velocity')
        
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_ylim(min(ego_v) - 2, max(ego_v) + 2)
        
        # Secondary y-axis for control inputs
        ax2 = ax.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Control Magnitude', color=color2, fontsize=12, fontweight='bold', labelpad=10)
        
        if len(controls) > 0:
            # Plot control magnitudes
            control_time = time_vec[:-1]
            steering_magnitude = np.abs(controls[:, 0])
            jerk_magnitude = np.abs(controls[:, 1])
            
            line2 = ax2.plot(control_time, steering_magnitude, color='red', 
                           linewidth=2, alpha=0.7, label='|Steering Rate|')
            line3 = ax2.plot(control_time, jerk_magnitude, color='orange', 
                           linewidth=2, alpha=0.7, label='|Jerk|')
            
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(0, max(np.max(steering_magnitude), np.max(jerk_magnitude)) * 1.1)
        
        # Combined legend with better positioning
        lines1, labels1 = ax.get_legend_handles_labels()
        if len(controls) > 0:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
        else:
            lines, labels = lines1, labels1
            
        ax.legend(lines, labels, loc='upper right', fontsize=10, 
                 bbox_to_anchor=(0.98, 0.98))
        
        ax.set_title('Velocity Profile & Control Inputs', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle=':')
    
    def plot_state_evolution(self):
        """Plot detailed state evolution with improved spacing"""
        states = self.results['states']
        time_vec = self.results['time']
        
        # Create figure with explicit spacing - more top margin
        fig = plt.figure(figsize=(20, 16))
        fig.subplots_adjust(left=0.08, right=0.95, top=0.87, bottom=0.08, hspace=0.45, wspace=0.35)
        gs = fig.add_gridspec(3, 2)
        
        # Place title with much more space
        fig.text(0.5, 0.94, 'Vehicle State Evolution', fontsize=20, fontweight='bold', ha='center')
        
        state_names = ['X Position', 'Y Position', 'Velocity', 'Heading', 'Steering Angle', 'Acceleration']
        state_units = ['m', 'm', 'm/s', 'rad', 'rad', 'm/s²']
        
        for i in range(6):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            
            if i < states.shape[1]:
                ax.plot(time_vec, states[:, i], linewidth=3, 
                       color=self.colors['ego_vehicle'], alpha=0.8)
                
                # Better axis formatting
                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=8)
                ax.set_ylabel(f'{state_names[i]} ({state_units[i]})', 
                             fontsize=12, fontweight='bold', labelpad=8)
                ax.set_title(state_names[i], fontsize=14, fontweight='bold', pad=12)
                ax.grid(True, alpha=0.3, linestyle=':')
                
                # Add limit lines where applicable with better labels
                if i == 2:  # Velocity
                    ax.axhline(y=VehicleConfig.MAX_SPEED, color='red', 
                              linestyle='--', alpha=0.7, linewidth=2, label='Max Speed')
                    ax.axhline(y=VehicleConfig.MIN_SPEED, color='orange', 
                              linestyle='--', alpha=0.7, linewidth=2, label='Min Speed')
                    ax.legend(fontsize=10, loc='best')
                elif i == 4:  # Steering Angle
                    ax.axhline(y=VehicleConfig.MAX_STEERING_ANGLE, color='red', 
                              linestyle='--', alpha=0.7, linewidth=2)
                    ax.axhline(y=-VehicleConfig.MAX_STEERING_ANGLE, color='red', 
                              linestyle='--', alpha=0.7, linewidth=2)
                elif i == 5:  # Acceleration
                    ax.axhline(y=VehicleConfig.MAX_ACCELERATION, color='red', 
                              linestyle='--', alpha=0.7, linewidth=2)
                    ax.axhline(y=VehicleConfig.MAX_DECELERATION, color='orange', 
                              linestyle='--', alpha=0.7, linewidth=2)
        
        filepath = os.path.join(self.output_dir, 'state_evolution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"✓ State evolution plot saved to: {filepath}")
        plt.show()
    
    def plot_control_inputs(self):
        """Plot detailed control inputs with improved spacing"""
        controls = self.results['controls']
        time_vec = self.results['time']
        
        if len(controls) == 0:
            print("No control data available for plotting")
            return
        
        # Create figure with better spacing - more top margin
        fig = plt.figure(figsize=(16, 10))
        fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.12, hspace=0.35)
        gs = fig.add_gridspec(2, 1)
        
        # Place title higher with more space
        fig.text(0.5, 0.95, 'MPC Control Inputs', fontsize=20, fontweight='bold', ha='center')
        
        # Steering rate subplot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_vec[:-1], controls[:, 0], linewidth=3, 
                color='blue', label='Steering Rate', alpha=0.8)
        ax1.axhline(y=VehicleConfig.MAX_STEERING_RATE, color='red', 
                   linestyle='--', alpha=0.7, linewidth=2, label='Max Limit')
        ax1.axhline(y=-VehicleConfig.MAX_STEERING_RATE, color='red', 
                   linestyle='--', alpha=0.7, linewidth=2, label='Min Limit')
        ax1.set_ylabel('Steering Rate (rad/s)', fontsize=14, fontweight='bold', labelpad=12)
        ax1.set_title('Steering Rate Control', fontsize=16, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.legend(fontsize=12, loc='best')
        
        # Jerk subplot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_vec[:-1], controls[:, 1], linewidth=3, 
                color='red', label='Jerk', alpha=0.8)
        ax2.axhline(y=VehicleConfig.MAX_JERK, color='red', 
                   linestyle='--', alpha=0.7, linewidth=2, label='Max Limit')
        ax2.axhline(y=-VehicleConfig.MAX_JERK, color='red', 
                   linestyle='--', alpha=0.7, linewidth=2, label='Min Limit')
        ax2.set_xlabel('Time (s)', fontsize=14, fontweight='bold', labelpad=12)
        ax2.set_ylabel('Jerk (m/s³)', fontsize=14, fontweight='bold', labelpad=12)
        ax2.set_title('Jerk Control', fontsize=16, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend(fontsize=12, loc='best')
        
        filepath = os.path.join(self.output_dir, 'control_inputs.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"✓ Control inputs plot saved to: {filepath}")
        plt.show()
    
    def plot_performance_metrics(self):
        """Plot performance metrics visualization with better spacing"""
        try:
            from performance_analysis import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(self.results, self.scenario)
            
            # Create figure with much better spacing - increased top margin
            fig = plt.figure(figsize=(18, 12))
            fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.08, hspace=0.45, wspace=0.35)
            gs = fig.add_gridspec(2, 2)
            
            # Use fig.text with more space for title
            fig.text(0.5, 0.93, 'Performance Metrics Summary', fontsize=20, fontweight='bold', ha='center')
            
            # Completion metrics
            ax1 = fig.add_subplot(gs[0, 0])
            completion_data = {
                'Completion\nTime': analyzer.metrics['completion_time'],
                'Lane Change\nTime': analyzer.metrics.get('lane_return_time', 0) - 
                                  analyzer.metrics.get('lane_departure_time', 0),
                'Final Lateral\nError': analyzer.metrics['final_lateral_error'],
                'Min Obstacle\nDistance': analyzer.metrics['min_obstacle_distance']
            }
            
            bars1 = ax1.bar(completion_data.keys(), completion_data.values(), 
                           color=['skyblue', 'lightgreen', 'orange', 'red'], alpha=0.8)
            ax1.set_title('Completion Metrics', fontsize=14, fontweight='bold', pad=15)
            ax1.tick_params(axis='x', rotation=0, labelsize=10)  # No rotation, smaller text
            
            # Add value labels on bars
            for bar, value in zip(bars1, completion_data.values()):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Comfort metrics
            ax2 = fig.add_subplot(gs[0, 1])
            comfort_data = {
                'Max Steering\nRate': analyzer.metrics['max_steering_rate'],
                'RMS Steering\nRate': analyzer.metrics['rms_steering_rate'],
                'Max Jerk': analyzer.metrics['max_jerk'],
                'RMS Jerk': analyzer.metrics['rms_jerk']
            }
            
            bars2 = ax2.bar(comfort_data.keys(), comfort_data.values(), 
                           color=['purple', 'violet', 'brown', 'pink'], alpha=0.8)
            ax2.set_title('Comfort Metrics', fontsize=14, fontweight='bold', pad=15)
            ax2.tick_params(axis='x', rotation=0, labelsize=10)
            
            # Add value labels
            for bar, value in zip(bars2, comfort_data.values()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Tracking performance
            ax3 = fig.add_subplot(gs[1, 0])
            tracking_data = {
                'RMS Lateral\nError': analyzer.metrics['rms_lateral_error'],
                'Max Lateral\nError': analyzer.metrics['max_lateral_error'],
                'RMS Velocity\nError': analyzer.metrics['rms_velocity_error'],
                'Max Velocity\nError': analyzer.metrics['max_velocity_error']
            }
            
            bars3 = ax3.bar(tracking_data.keys(), tracking_data.values(), 
                           color=['cyan', 'darkturquoise', 'gold', 'orange'], alpha=0.8)
            ax3.set_title('Tracking Performance', fontsize=14, fontweight='bold', pad=15)
            ax3.tick_params(axis='x', rotation=0, labelsize=10)
            
            # Add value labels
            for bar, value in zip(bars3, tracking_data.values()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Overall score pie chart with better spacing
            ax4 = fig.add_subplot(gs[1, 1])
            score_categories = {
                'Completion': 25 if analyzer.metrics['lane_change_completed'] else 0,
                'Safety': 25 if analyzer.metrics['min_obstacle_distance'] > 3.0 else 0,
                'Comfort': 25 if analyzer.metrics['comfort_score'] < 0.5 else 0,
                'Tracking': 25 if analyzer.metrics['rms_lateral_error'] < 0.5 else 0
            }
            
            colors_pie = ['lightgreen', 'lightcoral', 'lightblue', 'lightyellow']
            wedges, texts, autotexts = ax4.pie(score_categories.values(), 
                                              labels=score_categories.keys(), 
                                              autopct='%1.0f%%', 
                                              colors=colors_pie, 
                                              startangle=90,
                                              textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax4.set_title('Overall Score Breakdown', fontsize=14, fontweight='bold', pad=15)
            
            filepath = os.path.join(self.output_dir, 'performance_metrics.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
            print(f"✓ Performance metrics plot saved to: {filepath}")
            plt.show()
            
        except ImportError:
            print("Performance analysis module not available")
    
    def create_animation(self):
        """Create enhanced animation with improved obstacle tracking"""
        states = self.results['states']
        time_vec = self.results['time']
        
        if len(states) == 0:
            print("No trajectory data available for animation")
            return None
        
        # Enhanced figure setup with explicit spacing - more top margin
        fig, ax = plt.subplots(figsize=(18, 10))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10)
        
        # Road setup
        lane_width = RoadConfig.LANE_WIDTH
        num_lanes = RoadConfig.LANE_COUNT
        road_width = num_lanes * lane_width
        
        # Dynamic view window
        view_range = 100  # meters ahead and behind
        
        # Create fixed axes for info panel with better positioning
        info_panel_ax = fig.add_axes([0.72, 0.72, 0.25, 0.15])
        info_panel_ax.axis('off')  # Hide axes
        
        # Pre-create text object for info panel that will be updated
        info_text_obj = info_panel_ax.text(0.0, 1.0, "", fontsize=12, fontweight='bold',
                                         va='top', ha='left',
                                         bbox=dict(boxstyle='round,pad=0.5', 
                                                 facecolor='white', alpha=0.92, 
                                                 edgecolor='gray', linewidth=1))
        
        def init():
            ax.clear()
            return []
        
        def animate(frame):
            ax.clear()
            
            if frame >= len(states):
                return []
            
            current_state = states[frame]
            current_time = time_vec[frame] if frame < len(time_vec) else time_vec[-1]
            ego_x, ego_y = current_state[0], current_state[1]
            
            # Dynamic view centered on ego vehicle
            x_min = ego_x - view_range
            x_max = ego_x + view_range
            
            # Road background
            ax.fill_between([x_min, x_max], [0, 0], [road_width, road_width], 
                         color=self.colors['lane_fill'], alpha=0.3)
            
            # Lane lines
            for i in range(num_lanes + 1):
                y_pos = i * lane_width
                line_style = '-' if i in [0, num_lanes] else '--'
                line_width = 3 if i in [0, num_lanes] else 2
                ax.plot([x_min, x_max], [y_pos, y_pos], line_style, 
                     color=self.colors['lane_lines'], linewidth=line_width, alpha=0.8)
            
            # Fixed obstacle animation with proper tracking
            self._animate_obstacles(ax, current_time, x_min, x_max)
            
            # Fixed trajectory trail rendering - always show the last N points
            trail_length = min(50, frame + 1)  # +1 to include current frame
            if trail_length > 1:
                trail_start = max(0, frame + 1 - trail_length)
                trail_x = states[trail_start:frame+1, 0]
                trail_y = states[trail_start:frame+1, 1]
                ax.plot(trail_x, trail_y, color=self.colors['ego_trajectory'], 
                     linewidth=3, alpha=0.7, label='Trajectory')
            
            # Current ego vehicle with enhanced visualization
            self._draw_vehicle(ax, ego_x, ego_y, current_state[3], 
                           self.colors['ego_vehicle'], 'EGO', enhanced=True)
            
            # Update info panel text in its fixed position
            info_text = (f'Time: {current_time:.1f}s\n'
                      f'Position: ({ego_x:.1f}, {ego_y:.1f})\n'
                      f'Velocity: {current_state[2]:.1f} m/s\n'
                      f'Lane: {0 if ego_y < lane_width else 1}')
            info_text_obj.set_text(info_text)
            
            # Enhanced styling with more title padding
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-1, road_width + 1)
            ax.set_xlabel('Longitudinal Position (m)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Lateral Position (m)', fontsize=14, fontweight='bold')
            ax.set_title(f'Lane Change Animation - t={current_time:.1f}s', 
                      fontsize=16, fontweight='bold', pad=25)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_aspect('equal', adjustable='box')
            
            return []
        
        # Create animation with improved settings
        frames = min(len(states), 500)  # Limit frames for performance
        interval = max(50, int(1000 * SimulationConfig.DT))  # Adaptive interval
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=frames, 
                         interval=interval, blit=False, repeat=True)
        
        # Save animation
        try:
            filepath = os.path.join(self.output_dir, 'lane_change_animation.gif')
            writer = PillowWriter(fps=15, metadata=dict(artist='Lane Change MPC'))
            anim.save(filepath, writer=writer, dpi=150)
            print(f"✓ Animation saved to: {filepath}")
            
            # Also show the animation
            plt.show()
            
        except Exception as e:
            print(f"❌ Failed to save animation: {e}")
            print("Showing animation in window instead...")
            plt.show()
        
        return anim
    
    def _animate_obstacles(self, ax, current_time, x_min, x_max):
        """Enhanced obstacle animation with proper tracking - FIX for disappearing obstacles"""
        if not hasattr(self.scenario, 'obstacles') or not self.scenario.obstacles:
            return
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.scenario.obstacles)))
        
        for i, obs in enumerate(self.scenario.obstacles):
            obs_x_initial, obs_y, obs_speed = obs[0], obs[1], obs[2]
            
            # Calculate current obstacle position
            obs_x_current = obs_x_initial + obs_speed * current_time
            
            # Show obstacle if it's within extended view range (much larger to avoid pop-in)
            extended_range = 500  # Greatly increased range to prevent disappearing
            if x_min - extended_range < obs_x_current < x_max + extended_range:
                # Draw obstacle vehicle with consistent styling
                self._draw_vehicle(ax, obs_x_current, obs_y, 0, colors[i], f'OBS{i+1}')
                
                # Draw velocity vector with better visibility
                vector_length = obs_speed * 1.5  # Scale for visibility
                ax.arrow(obs_x_current, obs_y, vector_length, 0, 
                      head_width=0.4, head_length=1.5, fc=colors[i], ec=colors[i], 
                      alpha=0.7, zorder=5)
                
                # Speed label with consistent position
                ax.text(obs_x_current, obs_y + 1.8, f'{obs_speed:.1f} m/s', 
                     ha='center', va='bottom', fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                             alpha=0.8, edgecolor='black'), zorder=6)
    
    def _draw_vehicle(self, ax, x, y, heading, color, label, enhanced=False):
        """Draw vehicle rectangle with enhanced visualization"""
        length = VehicleConfig.LENGTH
        width = VehicleConfig.WIDTH
        
        # Vehicle corners relative to center
        corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2], 
            [length/2, width/2],
            [-length/2, width/2]
        ])
        
        # Rotate corners
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to vehicle position
        vehicle_corners = rotated_corners + np.array([x, y])
        
        # Draw vehicle
        vehicle = patches.Polygon(vehicle_corners, closed=True, 
                                 facecolor=color, edgecolor='black', 
                                 linewidth=2, alpha=0.8)
        ax.add_patch(vehicle)
        
        # Enhanced labeling
        if enhanced:
            # Direction arrow
            arrow_length = length * 0.7
            arrow_x = x + arrow_length * cos_h
            arrow_y = y + arrow_length * sin_h
            ax.arrow(x, y, arrow_x - x, arrow_y - y, 
                    head_width=0.5, head_length=1, fc='white', ec='black', linewidth=2)
        
        # Vehicle label
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')

if __name__ == "__main__":
    # Example usage
    try:
        from lane_change_mpc import LaneChangeScenario
        
        # Run simulation
        scenario = LaneChangeScenario()
        results = scenario.run_simulation()
        
        # Create visualizations
        visualizer = LaneChangeVisualizer(results, scenario)
        visualizer.plot_trajectory_overview()
        visualizer.plot_state_evolution()
        visualizer.plot_control_inputs()
        visualizer.plot_performance_metrics()
        visualizer.create_animation()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run this module from main_simulation.py for full functionality")
