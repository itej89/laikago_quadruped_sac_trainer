import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

#Class contains the model that approximates the value function
class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, 
                 name='Value', chkpt_dir='model/sac'):
        super(ValueNetwork, self).__init__()

        #network parameters
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        #fully connected layers
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')

        #out put layers that predicts value
        self.v = Dense(1, activation=None)

    #Forward propagation
    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)

        state_value = self.v(output)

        return state_value