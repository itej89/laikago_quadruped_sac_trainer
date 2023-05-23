import numpy as np

#Class provides the buffer to store and sample training data 
#obtained by exploring the environment
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        
        #contains the buffer size and current length
        self.mem_size = max_size
        self.mem_cntr = 0
        
        #buffers for all the required variables
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_meory = np.zeros(self.mem_size)
        self.terminal_memeory = np.zeros(self.mem_size, dtype=np.bool)

    #Function to store the transaction in teh replay buffer
    def store_transaction(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_meory[index] = reward
        self.action_memory[index] = action
        self.terminal_memeory[index] = done

        self.mem_cntr += 1

    #Funciton to sample the transaction from the replay buffer
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        #sample random batch indices
        batch = np.random.choice(max_mem, batch_size)

        #fetch samples
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_meory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memeory[batch]

        return states, actions, rewards, states_, dones
    