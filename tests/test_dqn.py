import numpy as np
import torch

from sc2learner.agents.dqn_agent import (
    DQNAgent, double_dqn_mmc_target, epsilon_schedule,
)
from sc2learner.agents.dqn_networks import NonspatialDuelingQNet
from sc2learner.agents.replay_memory import LocalReplayMemory


class DiscreteSpace:
    n = 3

    def sample(self):
        return 2


def test_dueling_network_matches_value_plus_centered_advantage():
    torch.manual_seed(20)
    network = NonspatialDuelingQNet(5, 3)
    inputs = torch.randn(4, 5)
    value = network.value_fc4(torch.relu(network.value_fc3(
        torch.relu(network.value_fc2(torch.relu(network.value_fc1(inputs)))))))
    advantage = network.adv_fc4(torch.relu(network.adv_fc3(
        torch.relu(network.adv_fc2(torch.relu(network.adv_fc1(inputs)))))))
    expected = value + advantage - advantage.mean(dim=1, keepdim=True)
    torch.testing.assert_close(network(inputs), expected)


def test_double_dqn_mmc_target_matches_frozen_formula():
    online = torch.tensor([[1.0, 4.0], [5.0, 2.0]])
    target = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    rewards = torch.tensor([2.0, 3.0])
    dones = torch.tensor([0.0, 1.0])
    mc = torch.tensor([7.0, 8.0])
    result = double_dqn_mmc_target(
        online, target, rewards, dones, mc, discount=0.5, mmc_beta=0.9)
    expected_one_step = torch.tensor([12.0, 3.0])
    torch.testing.assert_close(result, 0.9 * expected_one_step + 0.1 * mc)


def test_single_item_optimize_preserves_vector_shapes_and_clips_gradients():
    torch.manual_seed(21)
    agent = DQNAgent(NonspatialDuelingQNet(5, 3), DiscreteSpace(), device="cpu")
    loss = agent.optimize_step(
        torch.randn(1, 5), torch.randn(1, 5), torch.tensor([1]),
        torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([2.0]),
        discount=0.99, mmc_beta=0.9, gradient_clipping=0.05,
        adam_eps=1e-7, learning_rate=1e-4, target_update_interval=10,
    )
    assert np.isfinite(loss)
    assert all(
        parameter.grad is None or parameter.grad.abs().max() <= 0.050001
        for parameter in agent._network.parameters()
    )


def test_epsilon_schedule_preserves_two_stage_behavior():
    assert epsilon_schedule(0, 1.0, 0.1, 100, 1000) == 1.0
    assert epsilon_schedule(50, 1.0, 0.1, 100, 1000) == 0.55
    assert epsilon_schedule(100, 1.0, 0.1, 100, 1000) == 0.1
    assert epsilon_schedule(1000, 1.0, 0.1, 100, 1000) == 0.01


def test_dqn_checkpoint_round_trip_includes_training_state(tmp_path):
    torch.manual_seed(22)
    agent = DQNAgent(NonspatialDuelingQNet(5, 3), DiscreteSpace())
    agent.optimize_step(
        torch.randn(2, 5), torch.randn(2, 5), torch.tensor([0, 2]),
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]), discount=0.99, mmc_beta=0.9,
        gradient_clipping=10.0, adam_eps=1e-7, learning_rate=1e-4,
        target_update_interval=10,
    )
    path = tmp_path / "dqn.pt"
    agent.save(path)
    restored = DQNAgent(NonspatialDuelingQNet(5, 3), DiscreteSpace())
    restored.load(path, load_training_state=True)
    assert restored._num_optim_steps == 1
    for name, value in agent.read_params().items():
        torch.testing.assert_close(value, restored.read_params()[name])


def test_local_replay_capacity_sampling_and_total_count():
    memory = LocalReplayMemory(2)
    for index in range(3):
        observation = np.asarray([index], dtype=np.float32)
        memory.push(observation, index, float(index), observation + 1, False, 0.0)
    assert memory.total == 3
    sample = memory.sample(2)
    assert {transition.action for transition in sample} == {1, 2}
