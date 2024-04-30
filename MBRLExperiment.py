#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot,ComparisonPlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    
    wind_proportions=[0.9,1.0]
    n_planning_updates = [1,3,5] 
    agents = DynaAgent, PrioritizedSweepingAgent
    
    #make plot for eacht agent
    for agent in agents:
        plot = LearningCurvePlot(title=str(agent) + ' learning curves')
        #loop over wind_proportions and n_planning_updates
        for wind_proportion in wind_proportions:
            for n_planning_update in n_planning_updates:
                print('wind:' + str(wind_proportion) + ', planning_update: ' + str(n_planning_update))
                #run repetitions
                mean_rewards = run_repetitions(n_timesteps, n_repetitions, gamma, learning_rate, epsilon, wind_proportion, n_planning_update, eval_interval, agent)
                smoothed_rewards = smooth(mean_rewards, window=151)
                #add the mean rewards to plot
                plot.add_curve(np.arange(n_timesteps), smoothed_rewards, label='wind:' + str(wind_proportion)+' ,planning_update: ' + str(n_planning_update))
        if agent == DynaAgent:
            plot.save('Dyna_learning_curves.png')
        else:
            plot.save('PrioritizedSweeping_learning_curves.png')
    
    #make comparison plot
    best_parameters = [[1, 1],
                       [1, 3]]
    plot = ComparisonPlot(title='comparisonplot')
    for agent in range(len(agents)):
        for wind_proportion in range(len(wind_proportions)):
            mean_rewards = run_repetitions(n_timesteps, n_repetitions, gamma, learning_rate, epsilon, wind_proportions[wind_proportion], best_parameters[agent][wind_proportion], eval_interval, agents[agent])
            smoothed_rewards = smooth(mean_rewards, window=151)
            if agents[agent] == DynaAgent:
                plot.add_curve(np.arange(n_timesteps), smoothed_rewards, label='dyna; wind: ' + str(wind_proportions[wind_proportion]) + ', planning_update: ' + str(best_parameters[agent][wind_proportion]))
            else:
                plot.add_curve(np.arange(n_timesteps), smoothed_rewards, label='prioritized sweeping; wind: ' + str(wind_proportions[wind_proportion]) + ', planning_update: ' + str(best_parameters[agent][wind_proportion]))
    
    #add baseline
    mean_rewards = run_repetitions(n_timesteps, n_repetitions, gamma, learning_rate, epsilon, 0.9, 0, eval_interval, agents[0])
    smoothed_rewards = smooth(mean_rewards, window=151)
    plot.add_curve(np.arange(n_timesteps), smoothed_rewards, label='baseline')
    plot.save('comparisonplot.png')    
            

def run_repetitions(n_timesteps, n_repetitions, gamma, learning_rate, epsilon, wind_proportion, n_planning_updates, eval_interval, agents):
    # IMPLEMENT YOUR FUNCTION HERE
    # This function should run n_repetitions of the Dyna-Q algorithm on the WindyGridworld environment
    # It should return the average return per timestep over all repetitions
    # (This is the value that you should plot in the end)
    env = WindyGridworld(wind_proportion=wind_proportion)
    agent = agents(env.n_states, env.n_actions, learning_rate, gamma)
    mean_rewards = np.zeros((n_timesteps))
    for repetition in range(n_repetitions):
        #print(repetition)
        s = env.reset()
        for timestep in range(n_timesteps):
            #evaluate every 250 timesteps
            if timestep % eval_interval == 0:
                agent.evaluate(env)
            #select action and update 
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            #add reward to mean_rewards
            mean_rewards[timestep] += r
            if done:
                break
            s = s_next
    #calculate mean rewards
    mean_rewards /= n_repetitions
    return mean_rewards
    
    
            
    
    
    
    
if __name__ == '__main__':
    experiment()
    #run_repetitions()