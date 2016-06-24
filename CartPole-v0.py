import gym
import numpy

env = gym.make('CartPole-v0')

param = numpy.random.rand(4) - 0.5
env.monitor.start('/tmp/OpenAI-CartPole', force=True)

best_point = 0
for x in range(1000):
    observation = env.reset()
    points = 0
    while True:
        action = numpy.dot(param, observation)
        action = 1 if action > 0 else 0
        observation, reward, done, _ = env.step(action)
        points += reward
        if done:
            if points > best_point:
                best_point = points
                param += observation
            elif points < 200:
                param -= observation
            break
env.monitor.close()
