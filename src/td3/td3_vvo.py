"""Implement canonical Twin Delayed DDPG for physical regulator ratios.

The algorithm follows Fujimoto, van Hoof, and Meger, "Addressing Function
Approximation Error in Actor-Critic Methods":
https://proceedings.mlr.press/v80/fujimoto18a.html
and the author's implementation:
https://github.com/sfujim/TD3/blob/master/TD3.py

This module uses paper-aligned 400/300 hidden widths with the explicitly
recorded 3e-4 learning rates. The previous baseline used the author-code
256/256 recipe. Its actor emits normalized values
in [-1, 1]. The agent maps them by ``midpoint + half_range * value`` so every
critic input, replay action, exploration perturbation, and smoothing
perturbation outside the tanh uses physical p.u. ratios. Normalized smoothing
sigma 0.2 and clip 0.5 become 0.02 and 0.05 p.u. on [0.90, 1.10]. Normalized
exploration sigma 0.1 becomes 0.01 p.u.

The implementation is adapted from code released under the MIT License:

Copyright (c) 2018 Scott Fujimoto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class ReplayBuffer:
    """Store single-precision transitions and sample them with replacement."""

    def __init__(self, capacity, observation_size, action_count, rng):
        self.capacity = capacity
        self.count = 0
        self.rng = rng
        self.states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity, action_count), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """Append one transition and replace the oldest after capacity."""
        index = self.count % self.capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.count += 1

    def sample(self, batch_size):
        """Draw a uniform batch with replacement through the learner generator."""
        available = min(self.count, self.capacity)
        indices = self.rng.choice(available, batch_size, replace=True)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class Actor(nn.Module):
    """Map observations to normalized actions through two ReLU hidden layers."""

    def __init__(self, observation_size, action_count, layer_sizes=(400, 300)):
        super().__init__()
        first, second = layer_sizes
        self.first = nn.Linear(observation_size, first)
        self.second = nn.Linear(first, second)
        self.output = nn.Linear(second, action_count)

    def forward(self, state):
        """Return one tanh-bounded normalized action per regulator."""
        action = functional.relu(self.first(state))
        action = functional.relu(self.second(action))
        return torch.tanh(self.output(action))


class Critic(nn.Module):
    """Estimate two independent values from observations and physical actions."""

    def __init__(self, observation_size, action_count, layer_sizes=(400, 300)):
        super().__init__()
        first, second = layer_sizes
        combined = observation_size + action_count
        self.q1_first = nn.Linear(combined, first)
        self.q1_second = nn.Linear(first, second)
        self.q1_output = nn.Linear(second, 1)
        self.q2_first = nn.Linear(combined, first)
        self.q2_second = nn.Linear(first, second)
        self.q2_output = nn.Linear(second, 1)

    def forward(self, state, action):
        """Return both branch values for each observation-action pair."""
        combined = torch.cat((state, action), dim=1)
        q1 = functional.relu(self.q1_first(combined))
        q1 = functional.relu(self.q1_second(q1))
        q2 = functional.relu(self.q2_first(combined))
        q2 = functional.relu(self.q2_second(q2))
        return self.q1_output(q1), self.q2_output(q2)

    def q1(self, state, action):
        """Return only the first branch for the deterministic policy update."""
        combined = torch.cat((state, action), dim=1)
        value = functional.relu(self.q1_first(combined))
        value = functional.relu(self.q1_second(value))
        return self.q1_output(value)


class Agent:
    """Coordinate canonical TD3 networks, replay, and delayed updates.

    This class follows the TD3 paper at
    https://proceedings.mlr.press/v80/fujimoto18a.html and the author's code at
    https://github.com/sfujim/TD3/blob/master/TD3.py. The networks use the
    paper-aligned 400/300 hidden widths and explicitly recorded 3e-4 rates.

    Actor values use normalized coordinates only inside the tanh. The physical
    map adds the action midpoint and multiplies by half the action range. Target
    smoothing sigma and clip are constructor arguments in physical p.u. units;
    exploration remains in the training loop and uses physical p.u. units too.
    The learner has no episode-length assumption.
    """

    def __init__(
        self,
        *,
        observation_size,
        action_count,
        action_low,
        action_high,
        gamma,
        rng,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        replay_capacity=1_000_000,
        tau=0.005,
        policy_delay=2,
        target_smoothing_sigma=0.02,
        target_smoothing_clip=0.05,
        layer_sizes=(400, 300),
    ):
        """Build online networks, exact target copies, optimizers, and replay."""
        if action_low >= action_high:
            raise ValueError("action_low must be less than action_high")
        self.device = torch.device("cpu")
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_midpoint = (self.action_low + self.action_high) / 2
        self.action_half_range = (self.action_high - self.action_low) / 2
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_smoothing_sigma = target_smoothing_sigma
        self.target_smoothing_clip = target_smoothing_clip
        self.memory = ReplayBuffer(replay_capacity, observation_size, action_count, rng)

        self.actor = Actor(observation_size, action_count, layer_sizes)
        self.critic = Critic(observation_size, action_count, layer_sizes)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        for parameter in (*self.target_actor.parameters(), *self.target_critic.parameters()):
            parameter.requires_grad_(False)

        self.critic_iterations = 0
        self.actor_updates = 0
        self.target_updates = 0

    def physical_action(self, normalized):
        """Map clipped normalized actor values to continuous physical p.u. ratios."""
        bounded = torch.clamp(normalized, -1.0, 1.0)
        return self.action_midpoint + self.action_half_range * bounded

    def act(self, observation):
        """Return deterministic continuous physical p.u. ratios without noise."""
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.physical_action(self.actor(state))
        return action.cpu().numpy().astype(float)

    def remember(self, state, executed_action, reward, next_state, done):
        """Store executed physical ratios and the true terminal indicator."""
        self.memory.store(state, executed_action, reward, next_state, done)

    def _smoothed_target_action(self, next_states):
        """Add clipped physical smoothing noise and retain continuous bounded ratios."""
        target_action = self.physical_action(self.target_actor(next_states))
        noise = torch.randn_like(target_action) * self.target_smoothing_sigma
        noise = torch.clamp(noise, -self.target_smoothing_clip, self.target_smoothing_clip)
        return torch.clamp(target_action + noise, self.action_low, self.action_high)

    def learn(self):
        """Apply one twin-critic iteration and each scheduled delayed update.

        Replay contains executed physical p.u. ratios. Target smoothing is an
        independent physical p.u. perturbation and remains continuous. The
        terminal mask removes the bootstrap term after a true terminal state.
        Both critics update every iteration. The actor maximizes Q1 every
        ``policy_delay`` iterations, while critic parameters remain frozen, and
        only then do all targets receive one Polyak update.
        """
        if self.memory.count < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device).view(-1, 1)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actions = self._smoothed_target_action(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)
        q1_loss = functional.mse_loss(current_q1, target)
        q2_loss = functional.mse_loss(current_q2, target)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_iterations += 1

        actor_loss_value = None
        if self.critic_iterations % self.policy_delay == 0:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            policy_actions = self.physical_action(self.actor(states))
            actor_loss = -self.critic.q1(states, policy_actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
            self.actor_updates += 1
            actor_loss_value = float(actor_loss.item())
            self._update_targets()

        return float(critic_loss.item()), actor_loss_value

    def _update_targets(self):
        """Polyak-update actor and twin-critic targets after an actor update."""
        with torch.no_grad():
            for target, online in zip(
                self.target_actor.parameters(), self.actor.parameters(), strict=True
            ):
                target.copy_(self.tau * online + (1.0 - self.tau) * target)
            for target, online in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target.copy_(self.tau * online + (1.0 - self.tau) * target)
        self.target_updates += 1


def save_models(agent, checkpoint_dir):
    """Write online actor and twin-critic state dictionaries for inference."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), checkpoint_dir / "Actor_td3")
    torch.save(agent.critic.state_dict(), checkpoint_dir / "Critic_td3")


def load_models(agent, checkpoint_dir):
    """Load both required online networks into an existing inference agent."""
    actor_path = checkpoint_dir / "Actor_td3"
    critic_path = checkpoint_dir / "Critic_td3"
    for path in (actor_path, critic_path):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {path}")
    agent.actor.load_state_dict(
        torch.load(actor_path, map_location=agent.device, weights_only=True)
    )
    agent.critic.load_state_dict(
        torch.load(critic_path, map_location=agent.device, weights_only=True)
    )
    agent.actor.eval()
    agent.critic.eval()
