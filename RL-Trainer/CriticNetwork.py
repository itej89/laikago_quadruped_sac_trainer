import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

#Class that contains the model for Critic network
class CriticNetwork(keras.Model):
    def __init__(self,  n_actions, fc1_dims=256, fc2_dims=256,
                 name='critic', chkpt_dir='model/sac'):
        super(CriticNetwork, self).__init__()

        #modle parameters
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        #Create layers
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        #out put q layer
        self.q_func = Dense(1, activation=None)

    #run forward propagation
    def call(self, state, action):

        input  = tf.concat([state, action], axis = 1)
        output = self.fc1(input)
        output = self.fc2(output)
        q_value = self.q_func(output)

        return q_value

        