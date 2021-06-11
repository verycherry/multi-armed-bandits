import numpy as np


class Agent:
    """
    An agent is able to take one of a set of actions at each time step. The action is
    chosen using a strategy based on the history of prior actions and outcome
    observations.
    """
    def __init__(self, bandit, policy):
        self.policy = policy
        self.k = bandit.k
        self._value_estimates = np.zeros(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None
        # TODO: saving all rewards here for thompson is not scalable
        self.all_rewards = [[] for i in range(self.k)]

    def __str__(self):
        return f"{str(self.policy)}"

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = 0
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0
        self.all_rewards = [[] for i in range(self.k)]

    def choose(self):
        action = self.policy.choose(agent=self)
        self.last_action = action
        return action

    def observe(self, reward: float):
        self.action_attempts[self.last_action] += 1
        n_actions = self.action_attempts[self.last_action]
        average_reward = self._value_estimates[self.last_action]
        self._value_estimates[self.last_action] += (
            1 / n_actions * (reward - average_reward)
        )
        self.t += 1
        self.all_rewards[self.last_action].append(reward)

    @property
    def history(self):
        return self._value_estimates
