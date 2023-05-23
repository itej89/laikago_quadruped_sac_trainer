import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.layers import Dense


#Class contains the model for the SAC actor network
class ActorNetwork(keras.Model):

    #Model Constructor
    def __init__(self, action_lim, fc1_dims=256, fc2_dims=256,
                 name='actor', n_actions=2, chkpt_dir='model/sac'):
        super(ActorNetwork, self).__init__()

        
        self.n_actions = n_actions
        self.model_name=  name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_sac')

        self.noise = 1e-6
        
        #Create network layers
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')

        #Two output layers to generate PDF parameters
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

        #Thresholding parameters for actions
        self._action_lim = action_lim
        self._action_range = tf.convert_to_tensor(self._action_lim[:,[1]]-self._action_lim[:,[0]],dtype=tf.float32)
        self._action_lowerbound = tf.convert_to_tensor(self._action_lim[:,[0]],dtype=tf.float32)
        self._log_const = tf.math.reduce_sum(tf.math.log(0.5*self._action_range))

    #Forward propagation
    def call(self, state):
        #Run first two layers
        prob = self.fc1(state)
        prob = self.fc2(prob)

        #compute mean and stdev layers for PDF
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = tf.clip_by_value(sigma, self.noise, 1)
        
        #return pdf parameters
        return mu, sigma
    
    #Funciton to sample actions usign the PDF parameters
    def sample_normal(self, state):
        
        mu, sigma = self.call(state) # assume batch size is n, then mu and sigma should both be n by n_action\n",
        distribution = tfp.distributions.Normal(mu,sigma) # n sample by n_action distributions\n",

        # sample action from distribution\n",
        u = distribution.sample() # n by n_action\n",
        #print(u)\n",
        log_prob_u = distribution.log_prob(u) # n by n_action\n",
        log_prob_u = tf.reduce_sum(log_prob_u,axis=1,keepdims=True) # n by 1 (sum over action dimension for each state (row) )\n",
    
        # map within the action range\n",
        tanh_u = tf.math.tanh(u) # squash action into (-1,1)\n",
        action = (tanh_u+1.)*0.5*tf.transpose(self._action_range)+tf.transpose(self._action_lowerbound)  # n by n_action\n",
        # compute log prob due to the determinant inverse\n",
        log_prob_det = -self._log_const-tf.reduce_sum(1-tf.math.pow(tanh_u,2), axis=1,keepdims=True)
        # compute log prob\n",
        log_prob_action = log_prob_u + log_prob_det
        
        return action, log_prob_action