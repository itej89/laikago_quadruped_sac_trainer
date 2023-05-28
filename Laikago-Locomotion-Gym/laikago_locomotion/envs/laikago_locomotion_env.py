import math

import gym
import numpy as np
import pybullet as p

from laikago_locomotion.resources.laikago import Laikago
from laikago_locomotion.resources.plane import Plane

import matplotlib.pyplot as plt


class LaikagoLocomotionEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self,
                 fwd_dist_buffer_sz=5,
                 fwd_vel_rwd_weight=1.0,
                 base_pose_rwd_weight=1.0,
                 ctrl_rwd_weight=-1.0):
        self.action_space = gym.spaces.box.Box(
            low=np.array( [0.000383, 0.626537, -1.748481, -0.267296, 0.532887, -1.739073, 0.0, 0.271603, -1.839862, -0.248888, 0.242691, -1.835604]),
            high=np.array([0.22933,  0.805074, -0.982286,  0.139305, 0.82714,  -0.954666, 0.277363, 1.324073, -0.942585, 0.133936, 1.301261, -0.92895]))
        
        # base x, y, z, r, p, y (6) ; base linear and angular vel (6); joint position (16) ; joint velocity (16)
        # total dimension 44
        self.observation_space = gym.spaces.box.Box(
            low = np.array([-10000, -10000, 0, -3.14, -5, -5, 
                            -100, -100, -100, -100, -100, -100,
                            -3.14,-3.14,-3.14,-3.14,
                            -3.14,-3.14,-3.14,-3.14,
                            -3.14,-3.14,-3.14,-3.14,
                            -3.14,-3.14,-3.14,-3.14,
                            -100,-100,-100,-100,
                            -100,-100,-100,-100,
                            -100,-100,-100,-100,
                            -100,-100,-100,-100]),
            high= np.array( [10000,  10000, 10,  3.14,  5,  5,  
                            100, 100, 100, 100, 100, 100,
                            3.14,3.14,3.14,3.14,
                            3.14,3.14,3.14,3.14,
                            3.14,3.14,3.14,3.14,
                            3.14,3.14,3.14,3.14,
                            100,100,100,100,
                            100,100,100,100,
                            100,100,100,100,
                            100,100,100,100]) )
        
        self.client = p.connect(p.DIRECT)
        p.setGravity(0,0,-9.8, physicsClientId=self.client)
        p.setTimeStep(1./500, physicsClientId=self.client)
 
        self.laikago = None
        self.start = None
        self.done = False
        self.prev_dist_from_start = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self.np_random, _ = gym.utils.seeding.np_random()

        self.fwd_dist_buffer = []
        self.fwd_dist_buffer_sz = fwd_dist_buffer_sz

        self.fwd_vel_rwd_weight = fwd_vel_rwd_weight
        self.base_pose_rwd_weight = base_pose_rwd_weight
        self.ctrl_rwd_weight = ctrl_rwd_weight

        # the quaternion of the initial pose
        self.q0 = p.getQuaternionFromEuler([np.pi/2,0,-np.pi])

    def step(self, action):
        self.laikago.apply_action(action=action)
        p.stepSimulation(physicsClientId=self.client)

        laikago_ob = self.laikago.get_observation()   
        # Compute reward as L2 change in distance from start
        dist_from_start = math.sqrt(((laikago_ob[0] - self.start[0]) ** 2 +
                                  (laikago_ob[1] - self.start[1]) ** 2))
        # reward = max(self.prev_dist_from_start - dist_from_start, 0)
        self.prev_dist_from_start = dist_from_start

        reward = self.compute_reward(laikago_ob,action)

        # Done by falling
        if laikago_ob[3] <= 0:
            self.done = True

        # Done by reaching goal
        if dist_from_start > 100:
            self.done = True
            reward = 50

        ob = np.array(laikago_ob, dtype=np.float32)
        return ob, reward, self.done, dict()
    
    def compute_reward(self,curr_obs,action):

        # compute forward distance from a previous time step, which could be more than
        # the simulation step away (a single step is too small and distance traveled can be noisy)

        curr_y = curr_obs[1] # the robot is facing the positive y direction when spawned
        self.fwd_dist_buffer.append(curr_y) # use a buffer to store previous y coordinates
        if len(self.fwd_dist_buffer) > self.fwd_dist_buffer_sz:
            dist0 = self.fwd_dist_buffer.pop(0)
        else:
            dist0 = self.fwd_dist_buffer[0]
        dist1 = self.fwd_dist_buffer[-1]

        # compute difference of the base orientation from being flat; use quaternion and l2 norm
        curr_q = p.getQuaternionFromEuler(curr_obs[3:6])
        q_diff = np.inner(curr_q , self.q0) # q0 is the initial pose, use inner product to measure how different curr q is from q0 

        # compute control effort
        ctrl_effort = np.sum(np.power(action,2))

        reward = self.fwd_vel_rwd_weight*(dist1-dist0) + self.base_pose_rwd_weight*q_diff + self.ctrl_rwd_weight*ctrl_effort
        
        return reward


    def initModels(self):
        p.setGravity(0,0,-9.8, physicsClientId=self.client)
        p.setTimeStep(1./500, physicsClientId=self.client)
        
        Plane(self.client)
        self.laikago = Laikago(self.client)
        self.done = False
        
        self.start = (0,0)

        # Get observation to return
        laikago_ob = self.laikago.get_observation()
        
        self.prev_dist_from_start = math.sqrt(((laikago_ob[0] - self.start[0]) ** 2 +
                                           (laikago_ob[1] - self.start[1]) ** 2))
        return np.array(laikago_ob, dtype=np.float32)

    def reset(self):
        p.resetSimulation(self.client)
        return self.initModels()


    def render(self):
        p.disconnect()
        self.client = p.connect(p.GUI)
        self.initModels()

    def close(self):
        p.disconnect(self.client)    
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]