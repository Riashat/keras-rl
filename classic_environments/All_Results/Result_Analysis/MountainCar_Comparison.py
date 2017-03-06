import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 492
eps = range(eps)


# Boltzmann_Steps = np.load('Boltzmann_Train_Steps_Episodes.npy')
# Boltzmann_Rewards = np.load('Boltzmann_Train_Rewards_Episodes.npy')



# Epsilon_Steps = np.load('EpsilonGreedy_Train_Steps_Episodes.npy')
# Epsilon_Rewards = np.load('EpsilonGreedy_Train_Rewards_Episodes.npy')


Dropout_Steps = np.load('Dropout_Train_Steps_Episodes.npy')
Dropout_Rewards = np.load('Dropout_Train_Rewards_Episodes.npy')
print "Shape 1", Dropout_Steps.shape
print "Shape 2", Dropout_Rewards.shape





def comparing_exploration(stats1, stats2,  stats3, eps,  smoothing_window=1, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_7 = pd.Series(stats7).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_8 = pd.Series(stats8).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_9 = pd.Series(stats9).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="MC Dropout + Boltzmann")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="MC Dropout + Epsilon Greedy")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="MC Dropout + Highest Variance") 
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="MC Dropout + Thompson Sampling") 
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Epsilon Greedy") 
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="Boltzmann Exploration") 
    cum_rwd_7, = plt.plot(eps, rewards_smoothed_7, label="Random exploration") 
    cum_rwd_8, = plt.plot(eps, rewards_smoothed_8, label="Thompson Sampling Exploration") 
    cum_rwd_9, = plt.plot(eps, rewards_smoothed_9, label="MC Dropout Decaying Prob + Highest Variance") 


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6, cum_rwd_7, cum_rwd_8, cum_rwd_9])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN")  
    plt.show()



    return fig



def single_runs(stats1, eps,  smoothing_window=200, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Dropout Uncertainty Exploration")    


    plt.legend(handles=[cum_rwd_1])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN MountainCar with Dropout Uncertainty Exploration")  
    plt.show()

    return fig



def main():
    single_runs(Dropout_Rewards, eps)



if __name__ == '__main__':
    main()


