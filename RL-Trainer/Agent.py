import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.optimizers import Adam

from ReplayBuffer import *
from CriticNetwork import *
from ValueNetwork import *
from ActorNetwork import *

#Class implemenbts funcitonality of a soft actor critic network
class Agent:
    def __init__(self, policy_lr=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 target_entropy=1.0,alpha_lr=1e-4):
        
        #Decalre SAC agent parameters----------------------------
        #explore and exploit parameters
        self.gamma= gamma
        #variable controls the amount of update to the target network compared to value network
        self.tau = tau
        #Create a reply network
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size=batch_size
        self.n_actions = n_actions 

        #Create actor
        self.action_lim = np.array([[env.action_space.low[i],env.action_space.high[i]] for i in range(n_actions)])
        self.actor = ActorNetwork(n_actions=n_actions, name='actor',
                                #   max_action=env.action_space.high)
                                action_lim = self.action_lim)

        #Create 2 critic network result of the lowest output among the both willb e used for computation
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')

        #Create value network, target network is used to control 
        # the stability for environment exploration
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        #build all teh required netwroks
        self.actor.compile(optimizer=Adam(learning_rate=policy_lr))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))
        

        self.scale = reward_scale
        #make a copy of the value network weights into the target value network
        self.update_network_parameters(tau=1)

        # intialize alpha 
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha,tf.exp)
        self.alpha_optimizer = tf.optimizers.Adam(alpha_lr, name='alpha_optimizer')
        
        self.target_entropy = target_entropy

    # Fucntion to choose random action for exploring the environement
    def choose_random_action(self):
        u = np.random.rand(self.n_actions,1)
        action = u*(self.action_lim[:,[1]]-self.action_lim[:,[0]]) + self.action_lim[:,[0]]
        return np.squeeze(action)
    
    # Fucntion to choose the best action by exploting the learned policy
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        actions, _ = self.actor.sample_normal(state)

        return actions[0]
    
    #store the state in the replay buffer for training
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transaction(state, action, reward, new_state, done)
    
    #Fucntion that copies the value network parameters to target network
    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            #update the target network weights incrementally as per the tau value
            weights.append(weight*tau + targets[i]*(1-tau))
        
        #set target value weights
        self.target_value.set_weights(weights)


    #Function to  save all teh nwetwork weights
    def save_models(self):
        print("Saving models...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    #Funciton to load daved weights of the network
    def load_models(self):
        print("Loading models...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    #Funciton to learn the model using SAC algorithms
    def learn(self):
        #weight till enough number of samples are available in the buffer
        if self.memory.mem_cntr < self.batch_size:
            return
        
        #sample training data from the buffer
        state, action, reward, new_state, done = \
                    self.memory.sample_buffer(self.batch_size)
        
        #convert values to tensors
        states = tf.convert_to_tensor(state, dtype=np.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=np.float32)
        rewards = tf.convert_to_tensor(reward, dtype=np.float32)
        actions = tf.convert_to_tensor(action, dtype=np.float32)

        # ------- update value function -------- #
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)

            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy,
                                                     q2_new_policy), 1)
            
            alpha = tf.convert_to_tensor(self.alpha)
            value_target = critic_value - alpha*log_probs
            value_loss = 0.5*keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradient, \
                                                 self.value.trainable_variables))

        # ------- update actor function -------- #
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)

            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy,
                                                     q2_new_policy), 1)
            
            alpha = tf.convert_to_tensor(self.alpha)
            actor_loss = alpha*log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
          actor_network_gradient,   self.actor.trainable_variables))
        
        # -------- update critic function ---------#
        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale*rewards + self.gamma * value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state,actions), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state,actions), 1)
            critic_1_loss = 0.5*keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5*keras.losses.MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(critic_1_loss, 
                            self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss, 
                            self.critic_2.trainable_variables)
        
        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient,
                            self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient,
                            self.critic_2.trainable_variables))
        
        self.update_network_parameters()

        # ----------- update alpha ----------- #
        # alpha = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape() as tape:
            action,log_prob = self.actor.sample_normal(states)
            alpha_losses = self.alpha*tf.stop_gradient(-log_prob-self.target_entropy)
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
        
        alpha_gradients = tape.gradient(alpha_loss,[self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients,[self.log_alpha]))

