from gym.envs.registration import register
register( id='LaikagoLocomotion-v0', 
         entry_point='laikago_locomotion.envs:LaikagoLocomotionEnv'
         )