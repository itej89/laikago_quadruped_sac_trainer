import pybullet_envs
import gym
import laikago_locomotion
import numpy as np
from Agent import *
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #Create Environment
    env = gym.make('LaikagoLocomotion-v0')
    env.render()

    #Total number of episodes to play
    Totalepisodes = 500

    #score buffer to compute the average and filter the rewards
    score_buffer = [0]*5

    #Create Agent
    agent = Agent(input_dims=env.observation_space.shape, env=env,
              n_actions=env.action_space.shape[0],
             target_entropy=-env.action_space.shape[0],
             alpha_lr=1e-5)
    
    #Load pretrained model from "./model/sac" directory
    agent.load_models()


    #Training loop------------------------------------------------------------
    try:
        for i in tqdm(range(Totalepisodes), desc="Testing the network : "):
            
            #Reset environment for each episode 
            observation = env.reset()
            
            #Termination condition
            done = False

            #score per episode
            episodeReward = 0

            while not done:

                #get best action from policy
                action = agent.choose_action(observation)

                #execute the action
                new_observation, reward, done, info = env.step(action)

                #accumulate reward
                episodeReward += reward

                #update the observation
                observation = new_observation

            #save score in the buffer
            score_buffer.append(episodeReward)

            print(f"episode {i} score {episodeReward} average score = {np.mean(score_buffer[-5:])} alpha {agent.alpha.numpy()}")
    except Exception as e:
        print(f"Error in training : {e}")
    #---------------------------------------------------------------------



    env.close()

        








