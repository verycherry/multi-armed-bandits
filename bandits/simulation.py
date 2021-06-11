from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bandits.agent import Agent
from bandits.bandit import MultiArmedBandit

sns.set_style("white")
sns.set_context("talk")


class Simulator:
    def __init__(self, bandit: MultiArmedBandit, agents: List[Agent]):
        self.bandit = bandit
        self.agents = agents

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(
        self, trials: int = 100, experiments: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rewards = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(rewards)
        actions = np.zeros((self.bandit.k, len(self.agents)))

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    rewards[t, i] += reward
                    actions[action, i] += 1
                    if is_optimal:
                        optimal[t, i] += 1

        return rewards / experiments, optimal / experiments, actions

    def plot_average_reward(self, rewards: np.ndarray):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(rewards, linewidth=1)
        ax.set_ylabel("average reward")
        ax.set_xlabel("time step")
        ax.legend(self.agents, loc=4, prop={"size": 11})
        plt.show()

    def plot_percentage_optimal(self, is_optimal: np.ndarray):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(is_optimal * 100, linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% optimal arm")
        ax.set_xlabel("time step")
        ax.legend(self.agents, loc=4, prop={"size": 11})
        plt.show()

    def plot_number_of_actions_taken(self, actions: np.ndarray):
        bins = np.linspace(0, self.bandit.k - 1, self.bandit.k)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(self.agents)):
            ax.bar(
                bins + i / len(self.agents),
                actions[:, i],
                width=1 / len(self.agents),
                label=self.agents[i],
            )
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 0.7))
        plt.xticks(range(0, self.bandit.k))
        ax.set_ylabel("number of actions taken")
        ax.set_xlabel("arm")
        plt.show()
