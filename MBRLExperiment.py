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
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updatess = [1,3,5] 
    
    # IMPLEMENT YOUR EXPERIMENT HERE

def run_repetitions(n_timesteps=1000, n_repetitions=10, gamma=1.0, learning_rate=0.2, epsilon=0.1, wind_proportion=0.9, n_planning_updates=5, eval_interval=250):
    # IMPLEMENT YOUR FUNCTION HERE
    # This function should run n_repetitions of the Dyna-Q algorithm on the WindyGridworld environment
    # It should return the average return per timestep over all repetitions
    # (This is the value that you should plot in the end)
    env = WindyGridworld(wind_proportion=wind_proportion)
    agent = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)
    mean_rewards = np.zeros((n_timesteps))
    for repetition in range(n_repetitions):
        print(repetition)
        s = env.reset()
        for timestep in range(n_timesteps):
            if timestep % eval_interval == 0:
                agent.evaluate(env)
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            mean_rewards[timestep] += r
            if done:
                break
            s = s_next
    mean_rewards /= n_repetitions
    print(mean_rewards)
    
    plot = LearningCurvePlot()
    plot.add_curve(np.arange(n_timesteps), mean_rewards)
    plot.save('test.png')
            
    
    
    
    
if __name__ == '__main__':
    experiment()
    run_repetitions()