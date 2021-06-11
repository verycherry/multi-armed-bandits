import numpy as np

from bandits import policy as pc
from bandits.agent import Agent
from bandits.bandit import GaussianBandit
from bandits.simulation import Simulator


def test_correct_reset():
    n_arms = 3
    bandit = GaussianBandit(k=n_arms, mu=0, sigma=1)
    agents = [Agent(bandit, pc.FixedPolicy(0))]
    sim = Simulator(bandit, agents)
    _ = sim.run(1, 1)

    bandit_rewards = sim.bandit.rewards
    agent_value_estimates = sim.agents[0]._value_estimates

    # assert that we see a pulled first arm
    assert agent_value_estimates[0] != 0
    assert agent_value_estimates[1] == 0
    assert agent_value_estimates[2] == 0
    assert sim.agents[0].t == 1
    np.testing.assert_array_equal(sim.agents[0].action_attempts, [1, 0, 0])

    sim.reset()
    assert sim.agents[0].t == 0
    np.testing.assert_array_equal(agent_value_estimates, [0, 0, 0])
    np.testing.assert_array_equal(sim.agents[0].action_attempts, [0, 0, 0])
    # we expect different bandit rewards after resetting
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(bandit_rewards, sim.bandit.rewards)
