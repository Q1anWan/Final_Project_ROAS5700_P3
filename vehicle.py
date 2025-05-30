"""
Vehicle dynamics and control
"""

import numpy as np
from config import *

class Vehicle:
    def __init__(self, x, y, v, target_lane):
        # Vehicle parameters from config - NO MORE HARDCODED VALUES
        self.length = VehicleConfig.LENGTH
        self.width = VehicleConfig.WIDTH
        self.max_velocity = VehicleConfig.MAX_SPEED
        self.max_acceleration = VehicleConfig.MAX_ACCELERATION
        self.max_deceleration = VehicleConfig.MAX_DECELERATION
        self.max_steering_angle = VehicleConfig.MAX_STEERING_ANGLE
        
        # Overtaking behavior parameters from config
        self.min_following_distance = OvertakingConfig.MIN_FOLLOWING_DISTANCE
        self.overtaking_threshold = ObstacleConfig.OVERTAKING_ACTIVATION_DISTANCE
        self.speed_threshold = OvertakingConfig.MIN_SPEED_DIFFERENCE
        
        # State variables
        self.x = x
        self.y = y
        self.v = v
        self.current_lane = 0 if y < RoadConfig.LANE_WIDTH else 1
        self.target_lane = target_lane
        
    def should_overtake(self, vehicles):
        # Overtaking logic using config parameters
        front_vehicle = self.get_front_vehicle(vehicles)
        if front_vehicle is None:
            return False
            
        distance = front_vehicle.x - self.x
        speed_diff = self.v - front_vehicle.v
        
        # Use config parameters instead of hardcoded values
        if (distance < self.overtaking_threshold and speed_diff > 1.0) or \
           (distance < self.min_following_distance):
            return self.check_overtaking_safety(vehicles)
        
        return False
    
    def check_overtaking_safety(self, vehicles):
        # Safety check using config parameters
        target_lane = 1 - self.current_lane
        safety_margin = OvertakingConfig.SAFETY_GAP_REQUIRED
        
        # Check for vehicles in target lane
        for vehicle in vehicles:
            if vehicle.current_lane == target_lane:
                distance = abs(vehicle.x - self.x)
                if distance < safety_margin:
                    return False
        
        return True
    
    def get_front_vehicle(self, vehicles):
        """Find the closest vehicle ahead in the same lane"""
        front_vehicle = None
        min_distance = float('inf')
        
        for vehicle in vehicles:
            if (vehicle.current_lane == self.current_lane and 
                vehicle.x > self.x and 
                vehicle.x - self.x < min_distance):
                min_distance = vehicle.x - self.x
                front_vehicle = vehicle
        
        return front_vehicle
    
    def update_position(self, dt):
        """Update vehicle position based on current velocity"""
        self.x += self.v * dt
        
        # Update current lane based on y position
        self.current_lane = 0 if self.y < RoadConfig.LANE_WIDTH else 1
