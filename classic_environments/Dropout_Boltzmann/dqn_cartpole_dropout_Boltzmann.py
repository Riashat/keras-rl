import sys
sys.path.append("/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/")
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgentDropout
from rl.memory import SequentialMemory
from rl.policy import Dropout_Boltzmann



ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
#building the neural network model here for the DQN Agent
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)

'''
Using Dropout + Boltzmann Exploration
'''
policy = Dropout_Boltzmann()

dqn = DQNAgentDropout(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

'''
Using ADAM Optimizer for training the Deep Q Networks
'''
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
[total_train_episodes, steps_per_episode, reward_per_episode] = dqn.fit(env, policy=policy, nb_steps=300000, visualize=True, verbose=2)

print "Saving DQN Training Results"
np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/DQN_CartPole_Dropout_Boltzmann/'  + 'Dropout_Boltzmann_' +  'Train_Episodes' +'.npy', total_train_episodes)
np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/DQN_CartPole_Dropout_Boltzmann/'  + 'Dropout_Boltzmann_' +  'Train_Steps_Episodes' +'.npy', steps_per_episode)
np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/DQN_CartPole_Dropout_Boltzmann/'  + 'Dropout_Boltzmann_' +  'Train_Rewards_Episodes' +'.npy', reward_per_episode)


# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#total episodes for testing the DQN algorithm to be defined as an argument
[total_test_reward, total_test_steps] = dqn.test(env, nb_episodes=50, visualize=True)

print "Experiment DONE"
print "Saving Results"

np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/DQN_CartPole_Dropout_Boltzmann/'  + 'Dropout_Boltzmann_' +  'Test_Steps_Episodes' +'.npy', total_test_steps)
np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/DQN_CartPole_Dropout_Boltzmann/'  + 'Dropout_Boltzmann_' +  'Test_Rewards_Episodes' +'.npy', total_test_reward)












