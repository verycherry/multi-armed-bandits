import numpy as np
np.seterr(all="ignore")

from bandits.agent import Agent


class FixedPolicy:
    """
    The fixed policy will always choose the fixed arm specified during initialization.
    """
    def __init__(self, arm: int):
        self._arm = arm

    def __str__(self):
        return "Fixed policy"

    def choose(self, agent: Agent) -> int:
        return self._arm


class GreedyPolicy:
    """
    The Greedy policy will always choose the best arm. If multiple arms are the best, a
    random arm is chosen.
    """
    def __init__(self):
        pass

    def __str__(self):
        return "Greedy Policy"

    @staticmethod
    def choose(agent: Agent) -> int:
        action = np.argmax(agent.history)
        greedy_arms = np.where(agent.history == agent.history[action])[0]
        if len(greedy_arms) == 1:
            return action
        else:
            return np.random.choice(greedy_arms)


class RandomPolicy:
    """
    The Random policy will always choose a random arm.
    """
    def __init__(self):
        pass

    def __str__(self):
        return "Random Policy"

    @staticmethod
    def choose(agent: Agent) -> int:
        return np.random.choice(agent.k)


class EpsilonGreedyPolicy:
    """
    The Epsilon-Greedy policy will choose a random action with probability epsilon and
    take the best apparent approach with probability 1-epsilon. If multiple actions are
    tied for best choice, then a random action from that subset is selected.
    """
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __str__(self):
        return f"Epsilon Greedy Policy, e {self.epsilon}"

    def choose(self, agent: Agent) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(agent.k)
        else:
            action = np.argmax(agent.history)
            greedy_arms = np.where(agent.history == agent.history[action])[0]
            if len(greedy_arms) == 1:
                return action
            else:
                return np.random.choice(greedy_arms)


class UCBPolicy:
    """
    The Upper Confidence Bound algorithm (UCB). It applies an exploration factor to the
    expected value of each arm which can influence a greedy selection strategy to more
    intelligently explore less confident options.
    """
    def __init__(self, c: int = 2):
        self.c = c

    def __str__(self):
        return f"UCB_{self.c} policy"

    def choose(self, agent: Agent) -> int:
        exploration = np.divide(self.c * np.log(agent.t + 1), agent.action_attempts)
        exploration[np.isnan(exploration)] = 0
        exploration = np.sqrt(exploration)

        q = agent.history + exploration
        action = np.argmax(q)
        greedy_arms = np.where(q == q[action])[0]
        if len(greedy_arms) == 1:
            return action
        else:
            return np.random.choice(greedy_arms)


class ThompsonSampling:
    """ "
    TODO: explanation here.
    """
    def __init__(self, method: str = "gaussian"):
        self._method = method

    def __str__(self):
        return "Thompson sampling"

    def choose(self, agent: Agent) -> int:
        if self._method == "gaussian":
            std = np.zeros(agent.k)
            for arm in range(agent.k):
                std[arm] = np.std(agent.all_rewards[arm])

            std[(np.isnan(std)) | (std == 0)] = 1
            exploration = np.divide(2 * std, np.sqrt(agent.action_attempts))

            q = agent.history + exploration
            action = np.argmax(q)
            greedy_arms = np.where(q == q[action])[0]
            if len(greedy_arms) == 1:
                return action
            else:
                return np.random.choice(greedy_arms)
