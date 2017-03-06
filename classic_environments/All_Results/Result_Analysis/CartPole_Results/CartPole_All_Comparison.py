import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 691

eps_ind = 691
eps_ind = range(eps)



# Boltzmann_Train = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/Boltzmann_Train_Rewards_Episodes.npy')
Boltzmann_Train = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/Boltzmann_Train_Rewards_Episodes2.npy')
Boltzmann_Train = Boltzmann_Train[0:eps]


Dropout_Boltzmann = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/Dropout_Boltzmann_Train_Rewards_Episodes.npy')
Dropout_Boltzmann = Dropout_Boltzmann[0:eps]

Dropout_Epsilon = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/Dropout_Epsilon_Train_Rewards_Episodes.npy')
Dropout_Epsilon = Dropout_Epsilon[0:eps]


Dropout_Greedy = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/Dropout_Greedy_Train_Rewards_Episodes.npy')
Dropout_Greedy = Dropout_Greedy[0:eps]


Dropout_Highest_Variance = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/Dropout_Train_Rewards_Episodes.npy')
Dropout_Highest_Variance = Dropout_Highest_Variance[0:eps]


# Epsilon = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/EpsilonGreedy_Train_Rewards_Episodes.npy')
Epsilon = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/Keras-RL/keras-rl/classic_environments/All_Results/Result_Analysis/CartPole_Results/Raw_Results/EpsilonGreedy_Train_Rewards_Episodes2.npy')
Epsilon = Epsilon[0:eps]


print "Boltzmann", Boltzmann_Train.shape
print "Dropout Boltzmann", Dropout_Boltzmann.shape
print "Dropout Epslon", Dropout_Epsilon.shape
print "Dropout Greedy",Dropout_Greedy.shape
print "Dropout Highest Variance", Dropout_Highest_Variance.shape



def comparing_exploration(stats1, stats2,  stats3, stats4, stats5, stats6,  eps,  smoothing_window=500, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Boltzmann")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="MC Dropout + Boltzmann")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="MC Dropout + Epsilon") 
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="MC Dropout + Greedy") 
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="MC Dropout + Highest Variance") 
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="Epsilon Greedy") 


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Exploration Strategies in DQN on CartPole Environment")  
    plt.show()



    return fig



def single_runs(stats1, eps,  smoothing_window=200, noshow=False):


    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Dropout +  Boltzmann Exploration")    


    plt.legend(handles=[cum_rwd_1])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN on CartPole with Dropout + Boltzmann Exploration")  
    plt.show()

    return fig



def main():
    comparing_exploration(Boltzmann_Train, Dropout_Boltzmann, Dropout_Epsilon, Dropout_Greedy, Dropout_Highest_Variance, Epsilon, eps_ind)



if __name__ == '__main__':
	main()


