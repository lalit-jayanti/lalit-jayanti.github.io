---
layout: page
title: Reinforcement Learning
description: Implementation of various RL agents
img: assets/img/course_projects/intraoption_option1.gif
importance: 3
category: work
---

This project provides a collection of reinforcement learning agents and offers a unified framework to train, evaluate and visualize the performance of the agents on various [Gym](https://gymnasium.farama.org/) environments.

The following agents have been implemented:
- Tabular learning
    - Q-Learning ```QLearningAgent```
    - SARSA Learning ```SARSAAgent```
- Deep-Learning based function approximation
    - Dueling Deep Q-Network ```DDQNAgent```
    - REINFORCE algorithm ```ReinforceMCwithBaselineAgent, ReinforceMCwithoutBaselineAgent```
- Hierarchical Learning
    - Semi-Markov Decision Process Q-Learning ```SMDPQLearningAgent```
    - Intra-Option Q-Learning ```IntraOptionQLearningAgent```

More information about this project can be found at this [repository](https://github.com/lalit-jayanti/Reinforcement-Learning-Agents)