# aisample.py
# Created by Rick Henderson
# Created on April 29, 2016
#
# Created to explore the OpenAI gym framework.

# Original code from https://gym.openai.com/docs

import gym

# Create the environment. Make can take a number of values including:
# MountainCar-v0, CartPole-v0, MsPacman-v0, or Hopper-v0, BeamRider-v0, Berzerk-v0
env = gym.make('Berzerk-v0')

# Reset the environment before begining
env.reset()

# Classic Agent -> Environment loop
# Start the monitor for uploading experiments
env.monitor.start('/tmp/berzerk-experiment-2')

# Requires 100 episodes for evaluation
for i_episodes in xrange(100):
	# Reset the environment for the current episode
	observation = env.reset()
	# Set up a loop to perform 1000 steps
	for t in xrange(100):
    		env.render()
		# Step returns 4 values: observation (object)
       		#                        reward (float)
		#			 done (boolean)
		# 			 info (dictionary)
		print observation
		action = env.action_space.sample() # take a random action
		observation, reward, done, info = env.step(action)
		if done:
			print "Episode finished after {} timepsteps.".format(t+1)
			break

# End the monitor
env.monitor.close()
