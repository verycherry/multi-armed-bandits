import numpy as np


class MultiArmedBandit:
    """
    A multi-armed bandit.
    """
    def __init__(self, k: int):
        self.k = k
        self.rewards = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.rewards = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action: int):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k: int, mu: float = 0, sigma: float = 1):
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.rewards = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.rewards)

    def pull(self, action: int):
        return np.random.normal(self.rewards[action]), action == self.optimal
