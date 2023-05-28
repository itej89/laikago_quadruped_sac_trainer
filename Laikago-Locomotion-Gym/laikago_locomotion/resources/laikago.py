import pybullet as p
import os
import math
import numpy as np

from laikago_locomotion.resources.joint_info import *

class Laikago:

    def enable_collsion(self):
        #enable collision between lower legs
        #2,5,8 and 11 are the lower legs
        lower_legs = [2,5,8,11]
        for l0 in lower_legs:
            for l1 in lower_legs:
                if (l1>l0):
                    enableCollision = 1
                    # print("collision for pair",l0,l1, p.getJointInfo(self.laikago,l0)[12],p.getJointInfo(self.laikago,l1)[12], "enabled=",enableCollision)
                    p.setCollisionFilterPair(self.laikago, self.laikago, l0,l1,enableCollision)

    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "data", "laikago" ,'laikago_toes.urdf')
        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.laikago = p.loadURDF(f_name , [0,0,.5],[0,0.5,0.5,0], flags = urdfFlags, useFixedBase=False,
                              physicsClientId=client)
        
        maxforce = 20

        self.FR_HIP   = joint_info("FR_HIP"  , 0,  -1,    0, maxforce)
        self.FR_UPPER = joint_info("FR_UPPER", 1,   1, -0.7, maxforce)
        self.FR_LOWER = joint_info("FR_LOWER", 2,   1,  0.7, maxforce)
        self.FL_HIP   = joint_info("FL_HIP",   4,   1,    0, maxforce)
        self.FL_UPPER = joint_info("FL_UPPER", 5,   1, -0.7, maxforce)
        self.FL_LOWER = joint_info("FL_LOWER", 6,   1,  0.7, maxforce)
        self.RR_HIP   = joint_info("RR_HIP",   8,  -1,    0, maxforce)
        self.RR_UPPER = joint_info("RR_UPPER", 9,   1, -0.7, maxforce)
        self.RR_LOWER = joint_info("RR_LOWER", 10,  1,  0.7, maxforce)
        self.RL_HIP   = joint_info("RL_HIP",   12,  1,    0, maxforce)
        self.RL_UPPER = joint_info("RL_UPPER", 13,  1, -0.7, maxforce)
        self.RL_LOWER = joint_info("RL_LOWER", 14,  1,  0.7, maxforce)

        # FR_HIP = joint_info("FR_TOE",   3,   1,  0, maxforce)
        # FR_HIP = joint_info("FL_TOE",   7,  -1, 0, maxforce)
        # FR_HIP = joint_info("RR_TOE",   11, -1, 0, maxforce)
        # FR_HIP = joint_info("RL_TOE",   15,  1, 0, maxforce)
        
        self.enable_collsion()        

    def get_ids(self):
        return self.client, self.laikago

    def apply_action(self, action):
        FR_HIP_Action, FR_UPPER_Action, FR_LOWER_Action, \
              FL_HIP_Action, FL_UPPER_Action, FL_LOWER_Action, \
              RR_HIP_Action, RR_UPPER_Action, RR_LOWER_Action, \
              RL_HIP_Action, RL_UPPER_Action, RL_LOWER_Action = action


        p.setJointMotorControl2(self.laikago,self.FR_HIP.id,p.POSITION_CONTROL,self.FR_HIP.direction*FR_HIP_Action+self.FR_HIP.offset, force=self.FR_HIP.maxforce)
        p.setJointMotorControl2(self.laikago,self.FR_UPPER.id,p.POSITION_CONTROL,self.FR_UPPER.direction*FR_UPPER_Action+self.FR_UPPER.offset, force=self.FR_UPPER.maxforce)
        p.setJointMotorControl2(self.laikago,self.FR_LOWER.id,p.POSITION_CONTROL,self.FR_LOWER.direction*FR_LOWER_Action+self.FR_LOWER.offset, force=self.FR_LOWER.maxforce)
        p.setJointMotorControl2(self.laikago,self.FL_HIP.id,p.POSITION_CONTROL,self.FL_HIP.direction*FL_HIP_Action+self.FL_HIP.offset, force=self.FL_HIP.maxforce)
        p.setJointMotorControl2(self.laikago,self.FL_UPPER.id,p.POSITION_CONTROL,self.FL_UPPER.direction*FL_UPPER_Action+self.FL_UPPER.offset, force=self.FL_UPPER.maxforce)
        p.setJointMotorControl2(self.laikago,self.FL_LOWER.id,p.POSITION_CONTROL,self.FL_LOWER.direction*FL_LOWER_Action+self.FL_LOWER.offset, force=self.FL_LOWER.maxforce)
        p.setJointMotorControl2(self.laikago,self.RR_HIP.id,p.POSITION_CONTROL,self.RR_HIP.direction*RR_HIP_Action+self.RR_HIP.offset, force=self.RR_HIP.maxforce)
        p.setJointMotorControl2(self.laikago,self.RR_UPPER.id,p.POSITION_CONTROL,self.RR_UPPER.direction*RR_UPPER_Action+self.RR_UPPER.offset, force=self.RR_UPPER.maxforce)
        p.setJointMotorControl2(self.laikago,self.RR_LOWER.id,p.POSITION_CONTROL,self.RR_LOWER.direction*RR_LOWER_Action+self.RR_LOWER.offset, force=self.RR_LOWER.maxforce)
        p.setJointMotorControl2(self.laikago,self.RL_HIP.id,p.POSITION_CONTROL,self.RL_HIP.direction*RL_HIP_Action+self.RL_HIP.offset, force=self.RL_HIP.maxforce)
        p.setJointMotorControl2(self.laikago,self.RL_UPPER.id,p.POSITION_CONTROL,self.RL_UPPER.direction*RL_UPPER_Action+self.RL_UPPER.offset, force=self.RL_UPPER.maxforce)
        p.setJointMotorControl2(self.laikago,self.RL_LOWER.id,p.POSITION_CONTROL,self.RL_LOWER.direction*RL_LOWER_Action+self.RL_LOWER.offset, force=self.RL_LOWER.maxforce)
        

    def get_observation(self,verbose=0):
        # Get the position and orientation of the laikago in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.laikago, self.client)
        ang = p.getEulerFromQuaternion(ang)
        vel = p.getBaseVelocity(self.laikago, self.client)[0][0:2]
        
        # also get the position and velocity of all joints
        info_list = p.getJointStates(self.laikago,self.client, np.arange(0, p.getNumJoints(self.laikago)))
        jointPos = np.array([info[0] for info in info_list])
        jointVel = np.array([info[1] for info in info_list])
        jointReactionWrench = np.array([info[2] for info in info_list]) # can add the wrench later

        # Concatenate position, orientation, velocity
        # observation = (pos + ang + vel)
        observation = tuple(pos) + tuple(ang) + tuple(vel) + tuple(jointPos) + tuple(jointVel)
        if verbose:
            print('current observation:',observation)

        return observation























#-------------------------JOINT_INFO-------------------------------------------
# (0, b'FR_hip_motor_2_chassis_joint', 0, 7, 6, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'FR_hip_motor', (0.0, 0.0, -1.0), (-0.0817145, -0.03, 0.199095), (0.0, 0.0, 0.0, 1.0), -1)
# (1, b'FR_upper_leg_2_hip_motor_joint', 0, 8, 7, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'FR_upper_leg', (1.0, 0.0, 0.0), (-0.073565, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
# (2, b'FR_lower_leg_2_upper_leg_joint', 0, 9, 8, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'FR_lower_leg', (1.0, 0.0, 0.0), (0.02069, -0.16832999999999998, -0.10219999999999999), (0.0, 0.0, 0.0, 1.0), 1)
# (3, b'jtoeFR', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'toeFR', (0.0, 0.0, 0.0), (0.0, -0.09, -0.0019999999999999983), (0.0, 0.0, 0.0, 1.0), 2)
# (4, b'FL_hip_motor_2_chassis_joint', 0, 10, 9, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'FL_hip_motor', (0.0, 0.0, 1.0), (0.0817145, -0.03, 0.199095), (0.0, 0.0, 0.0, 1.0), -1)
# (5, b'FL_upper_leg_2_hip_motor_joint', 0, 11, 10, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'FL_upper_leg', (1.0, 0.0, 0.0), (0.075855, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 4)
# (6, b'FL_lower_leg_2_upper_leg_joint', 0, 12, 11, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'FL_lower_leg', (1.0, 0.0, 0.0), (-0.02069, -0.16832999999999998, -0.10219999999999999), (0.0, 0.0, 0.0, 1.0), 5)
# (7, b'jtoeFL', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'toeFL', (0.0, 0.0, 0.0), (0.0, -0.09, -0.0019999999999999983), (0.0, 0.0, 0.0, 1.0), 6)
# (8, b'RR_hip_motor_2_chassis_joint', 0, 13, 12, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'RR_hip_motor', (0.0, 0.0, -1.0), (-0.0817145, -0.03, -0.238195), (0.0, 0.0, 0.0, 1.0), -1)
# (9, b'RR_upper_leg_2_hip_motor_joint', 0, 14, 13, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'RR_upper_leg', (1.0, 0.0, 0.0), (-0.073565, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 8)
# (10, b'RR_lower_leg_2_upper_leg_joint', 0, 15, 14, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'RR_lower_leg', (1.0, 0.0, 0.0), (0.02069, -0.16832999999999998, -0.10219999999999999), (0.0, 0.0, 0.0, 1.0), 9)
# (11, b'jtoeRR', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'toeRR', (0.0, 0.0, 0.0), (0.0, -0.09, -0.0019999999999999983), (0.0, 0.0, 0.0, 1.0), 10)
# (12, b'RL_hip_motor_2_chassis_joint', 0, 16, 15, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'RL_hip_motor', (0.0, 0.0, 1.0), (0.0817145, -0.03, -0.238195), (0.0, 0.0, 0.0, 1.0), -1)
# (13, b'RL_upper_leg_2_hip_motor_joint', 0, 17, 16, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'RL_upper_leg', (1.0, 0.0, 0.0), (0.075855, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 12)
# (14, b'RL_lower_leg_2_upper_leg_joint', 0, 18, 17, 1, 0.0, 0.0, 0.0, -1.0, 100.0, 100.0, b'RL_lower_leg', (1.0, 0.0, 0.0), (-0.02069, -0.16832999999999998, -0.10219999999999999), (0.0, 0.0, 0.0, 1.0), 13)
# (15, b'jtoeRL', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'toeRL', (0.0, 0.0, 0.0), (0.0, -0.09, -0.0019999999999999983), (0.0, 0.0, 0.0, 1.0), 14)