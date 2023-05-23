import os
import gym
import laikago_locomotion

env = gym.make('LaikagoLocomotion-v0')
env.render()

try:
   with open(os.path.join(os.path.dirname(__file__), "move.txt")) as filestream:
      for line in filestream:
         print(line)
         currentline = line.split(",")
         frame = currentline[0]
         t = currentline[1]
         joints=currentline[2:14]
         targetPos = []
         for j in range (12):
               targetPos.append(float(joints[j]))
         env.step(targetPos)

   # t = 0
   # while True:
   #    t += 1
   #    # env.render()
   #    print(observation)
   #    action = env.action_space.sample()
   #    observation, reward, done, info = env.step(action)
   #    print(observation, reward, done, info)
   #    if done:
   #       print("Episode finished after {} timesteps".format(t+1))
   #       observation = env.reset()
except Exception as e: 
   print(f"Exception : {e}")     
   env.close()