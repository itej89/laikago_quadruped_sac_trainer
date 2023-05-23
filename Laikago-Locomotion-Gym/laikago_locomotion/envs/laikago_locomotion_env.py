import math

import gym
import numpy as np
import pybullet as p

from laikago_locomotion.resources.laikago import Laikago
from laikago_locomotion.resources.plane import Plane

import matplotlib.pyplot as plt


class LaikagoLocomotionEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array( [0.000383, 0.626537, -1.748481, -0.267296, 0.532887, -1.739073, 0.0, 0.271603, -1.839862, -0.248888, 0.242691, -1.835604]),
            high=np.array([0.22933,  0.805074, -0.982286,  0.139305, 0.82714,  -0.954666, 0.277363, 1.324073, -0.942585, 0.133936, 1.301261, -0.92895]))
        self.observation_space = gym.spaces.box.Box(
            low = np.array([-10000, -10000, 0, -1, -5, -5, -10, -10]),
            high= np.array([10000,  10000, 1,  1,  5,  5,  10,  10]))
        
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

    def step(self, action):
        self.laikago.apply_action(action=action)
        p.stepSimulation(physicsClientId=self.client)

        laikago_ob = self.laikago.get_observation()   
        # Compute reward as L2 change in distance from start
        dist_from_start = math.sqrt(((laikago_ob[0] - self.start[0]) ** 2 +
                                  (laikago_ob[1] - self.start[1]) ** 2))
        reward = max(self.prev_dist_from_start - dist_from_start, 0)
        self.prev_dist_from_start = dist_from_start


        # Done by falling
        if laikago_ob[3] <= 0:
            self.done = True

        # Done by reaching goal
        if dist_from_start > 100:
            self.done = True
            reward = 50

        ob = np.array(laikago_ob, dtype=np.float32)
        return ob, reward, self.done, dict()
    
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