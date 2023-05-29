import numpy as np
import matplotlib.pyplot as plt

import pybullet_envs
import gym
import laikago_locomotion

from Agent import *
from tqdm import tqdm


if __name__ == '__main__':

    #Create environment
    env = gym.make('LaikagoLocomotion-v0',
                   fwd_dist_buffer_sz=5,
                   fwd_vel_rwd_weight=10.0,
                   base_pose_rwd_weight=0.0,
                   ctrl_rwd_weight=0.0,
                   z_terminate_height=0.1)
    
    #Create agent
    input_dims = env.observation_space.shape
    agent = Agent(input_dims=input_dims, env=env,
              n_actions=env.action_space.shape[0],
             target_entropy=-env.action_space.shape[0],
             alpha_lr=1e-7)
    print('alpha learning rate is:',agent.alpha_optimizer.learning_rate)

    #Training variables----------------------------
    #Total number of epochs
    TOTAL_EPOCHS = 1000
    MAX_EPOCH_LENGTH = 2000

    #To save the best network
    best_score = env.reward_range[0]

    #To compute average score
    score_buffer = [0]*5
    #----------------------------------------------


    #Load pretrained model (Comment this line of training for the first time)
    # agent.load_models()


    #Collect samples before starting to trian the network-------------------
    #This helps in avoiding getting stuck in a local minima-----------------
    for i in tqdm(range(500), desc="Collecting random samples  :"):
        observation = env.reset()
        done = False
        count = 0
        while not done and count < 1000:
            count += 1
            action = agent.choose_random_action()
            new_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)
            observation = new_observation
    #-----------------------------------------------------------------------


    #open files to store the alpha value and reward for creating plots
    with open("./model/sac/alpha.txt", 'w') as fAlpha:
        with open("./model/sac/reward.txt", 'w') as fReward:

            #Trainign loop---------------------------------------------------------------
            try:
                with tqdm(range(TOTAL_EPOCHS), desc="Trainign network : ") as pbar:
                    for i in pbar:

                        #reset the environment at the start of every epoch
                        observation = env.reset()

                        done = False
                        episodeReward = 0
                        count = 0
                        
                        #run episode (used counter to limit number of steps)
                        while not done:
                            pbar.set_description(f"Training network : {i}/{count}")
                            count +=1 

                            #get action from the network
                            action = agent.choose_action(observation)
                            #perform the action
                            new_observation, reward, done, info = env.step(action)
                            #accumulate the reward
                            episodeReward += reward

                            #store the step information in the replay buffer
                            agent.remember(observation, action, reward, new_observation, done)
                            #train the network
                            agent.learn()

                            #upodate the observation 
                            observation = new_observation

                            # terminate if reach max length
                            if count>=MAX_EPOCH_LENGTH:
                                done = 1
                        
                        #write the alpha and score int othe files
                        fReward.write(f"\n{episodeReward}")
                        fAlpha.write(f"\n{agent.alpha}")

                        #save the score
                        score_buffer.append(episodeReward)
                        avg_score = np.mean(score_buffer[-5:])
                        
                        #check if the model performs better than earlier models
                        if avg_score > best_score:
                            best_score = avg_score
                            #save the model
                            agent.save_models()

                        print(f"\nEpisode {i} score {episodeReward} average score = {np.mean(score_buffer[-5:])} alpha {agent.alpha.numpy()}\n")
            except Exception as e:
                print(e)
            #------------------------------------------------------------------------------

    #Plot rewards for the whole training loop---------
    rewards = [i+1 for i in range(len(score_buffer))]
    plt.plot(rewards, score_buffer)
    plt.show()
    #-------------------------------------------------
 
    env.close()

        








