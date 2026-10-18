"""Implement the shared CPU DDPG actor, critic, replay, and learning update.

The actor proposes continuous physical tap ratios. Replay stores ratios that
the feeder actually executed after grid projection. Online networks learn from
sampled transitions, while target networks change only through Polyak updates.
"""

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

OUTPUT_INIT = 0.003  # This bound applies to both output layers.


def raw_to_physical_torch(raw, lo, hi):
    """Map clipped actor outputs from [-1, 1] to physical p.u. action bounds."""
    x = T.clamp(raw, -1.0, 1.0)
    return lo + 0.5 * (x + 1.0) * (hi - lo)


class GaussianActionNoise:
    """Generate Gaussian exploration noise in raw actor coordinates.

    The training loops decay ``sigma`` once per one-shot episode or complete
    sequential day, never below ``sigma_floor``.
    """

    def __init__(self, n, sigma=0.4, sigma_floor=0.15, decay=0.99977):
        self.n, self.sigma, self.sigma_floor, self.decay = n, sigma, sigma_floor, decay

    def __call__(self):
        """Draw one independent raw-unit noise value per action dimension."""
        return self.sigma * np.random.normal(size=self.n)

    def decay_sigma(self):
        """Decay the noise scale once while enforcing its configured floor."""
        self.sigma = max(self.sigma * self.decay, self.sigma_floor)


class ReplayBuffer:
    """Store a circular collection of single-precision DDPG transitions.

    Actions are executed physical tap ratios in regulator order. The array named
    ``terminal_memory`` stores the bootstrap mask ``1 - done``: one continues a
    target through the next state, while zero marks a terminal transition.
    """

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size, self.mem_cntr = max_size, 0
        self.state_memory = np.zeros((max_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.float32)  # This mask equals 1 - done.

    def store_transition(self, state, action, reward, state_, done):
        """Append one transition, overwriting the oldest slot after capacity."""
        i = self.mem_cntr % self.mem_size
        self.state_memory[i] = state
        self.new_state_memory[i] = state_
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = 1.0 - float(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """Sample distinct stored transitions using NumPy's global random state."""
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.new_state_memory[batch],
            self.terminal_memory[batch],
        )


def fan_in_init_(layer, bound=None):
    """Apply fan-in initialization unless the caller gives a bound."""
    b = bound if bound is not None else 1.0 / np.sqrt(layer.in_features)
    layer.weight.data.uniform_(-b, b)
    layer.bias.data.uniform_(-b, b)


class CriticNetwork(nn.Module):
    """Estimate expected return from an observation and physical tap ratios."""

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, weight_decay=0.0):
        """Build the online or target critic with the configured hidden widths."""
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.fc_action = nn.Linear(n_actions, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        for layer in (self.fc1, self.fc2, self.fc_action, self.fc3):
            fan_in_init_(layer)
        fan_in_init_(self.q, OUTPUT_INIT)
        # The default recipe sets weight decay to zero.
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=weight_decay)

    def forward(self, state, action):
        """Return one scalar action-value estimate per observation-action pair."""
        s = F.relu(self.ln1(self.fc1(state)))
        s = F.relu(self.ln2(self.fc2(s)))
        a = self.fc_action(action)
        x = F.relu(self.ln3(self.fc3(s + a)))
        return self.q(x)


class ActorNetwork(nn.Module):
    """Map observations to bounded raw actions, one per controlled regulator."""

    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        """Build the online or target actor with the configured hidden widths."""
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        fan_in_init_(self.fc1)
        fan_in_init_(self.fc2)
        fan_in_init_(self.mu, OUTPUT_INIT)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        """Return tanh-bounded raw actions in regulator order."""
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return T.tanh(self.mu(x))


class Agent:
    """Coordinate DDPG networks, replay, exploration, and CPU updates.

    The online actor and critic own optimizers. Target networks begin as exact
    copies, remain gradient-free, and follow their online counterparts through
    Polyak updates. Seeding occurs in the training entry points.
    """

    def __init__(
        self,
        action_low,
        action_high,
        alpha,
        beta,
        input_dims,
        tau,
        n_actions,
        gamma=0.0,
        max_size=1000000,
        layer1_size=400,
        layer2_size=300,
        batch_size=64,
        critic_wd=0.0,
        noise_sigma=0.4,
        noise_floor=0.15,
        noise_decay=0.99977,
    ):
        """Construct a learner for one observation width and action grid bounds."""
        self.lo, self.hi = float(action_low), float(action_high)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.device = T.device("cpu")
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions)
        self.critic = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size, n_actions, critic_wd
        )
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions)
        self.target_critic = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size, n_actions, critic_wd
        )
        del self.target_actor.optimizer, self.target_critic.optimizer  # Use only Polyak updates.
        self.update_network_parameters(tau=1.0)
        for p in list(self.target_actor.parameters()) + list(self.target_critic.parameters()):
            p.requires_grad_(False)
        self.noise = GaussianActionNoise(n_actions, noise_sigma, noise_floor, noise_decay)

    def raw_to_physical(self, raw):
        """Map raw actor coordinates to continuous physical tap ratios in p.u."""
        return raw_to_physical_torch(raw, self.lo, self.hi)

    def choose_action(self, observation, explore=True):
        """Return continuous proposed ratios and the added raw-unit noise.

        Exploration adds Gaussian noise before clipping and physical scaling.
        The caller still projects proposed ratios to executable tap positions.
        """
        self.actor.eval()
        with T.no_grad():
            s = T.as_tensor(np.asarray(observation), dtype=T.float32, device=self.device)
            raw = self.actor(s)
            noise = (
                np.asarray(self.noise(), dtype=np.float32)
                if explore
                else np.zeros(raw.shape[-1], np.float32)
            )
            proposed = self.raw_to_physical(raw + T.as_tensor(noise))
        self.actor.train()
        return proposed.cpu().numpy().astype(float), noise

    def choose_action_deterministic(self, observation):
        """Return continuous actor ratios without exploration noise."""
        return self.choose_action(observation, explore=False)[0]

    def remember(self, state, executed_action, reward, new_state, done):
        """Store a transition using executed physical ratios and terminal status."""
        self.memory.store_transition(state, executed_action, reward, new_state, done)

    def learn(self):
        """Apply one critic update, actor update, and target-network update.

        Returns ``None`` until replay holds a complete batch. The critic target is
        ``reward + gamma * target_Q * (1 - done)`` and remains gradient-free.
        The critic learns from executed replay actions. The actor learns through
        continuous physical ratios while critic parameters are frozen. Both
        online gradients are clipped before their optimizer steps.
        """
        if self.memory.mem_cntr < self.batch_size:  # Wait for one complete batch.
            return None
        states, actions, rewards, states_, not_done = self.memory.sample_buffer(self.batch_size)
        states = T.as_tensor(states, device=self.device)
        actions = T.as_tensor(actions, device=self.device)
        rewards = T.as_tensor(rewards, device=self.device).view(-1, 1)
        states_ = T.as_tensor(states_, device=self.device)
        not_done = T.as_tensor(not_done, device=self.device).view(-1, 1)

        with T.no_grad():  # Keep the Bellman label constant.
            next_action = self.raw_to_physical(self.target_actor(states_))
            target = rewards + self.gamma * self.target_critic(states_, next_action) * not_done

        self.critic.train()
        q = self.critic(states, actions)  # Replay contains executed actions.
        critic_loss = F.mse_loss(q, target)
        self.critic.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic.optimizer.step()

        for p in self.critic.parameters():  # Freeze the critic for this actor update.
            p.requires_grad_(False)
        self.critic.eval()
        policy_action = self.raw_to_physical(self.actor(states))
        actor_loss = -self.critic(states, policy_action).mean()
        self.actor.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        self.update_network_parameters()
        return float(critic_loss.item()), float(actor_loss.item())

    def update_network_parameters(self, tau=None):
        """Polyak-update both target networks from their online networks.

        ``tau=1`` performs the initial exact copy. Later calls use the configured
        fraction unless the caller supplies another value.
        """
        tau = self.tau if tau is None else tau
        with T.no_grad():
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.copy_(tau * p + (1.0 - tau) * tp)
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.copy_(tau * p + (1.0 - tau) * tp)


def save_models(agent, checkpoint_dir):
    """Write online actor and critic state dictionaries beneath one run."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    T.save(agent.actor.state_dict(), checkpoint_dir / "Actor_ddpg")
    T.save(agent.critic.state_dict(), checkpoint_dir / "Critic_ddpg")


def load_models(agent, checkpoint_dir):
    """Load a required actor and optional critic into an existing agent.

    This direct loader does not validate run identity or tensor shapes in
    advance. Evaluation entry points use ``ddpg.checkpoints`` instead.
    """
    actor = checkpoint_dir / "Actor_ddpg"
    if not actor.is_file():
        raise FileNotFoundError(f"actor checkpoint is missing: {actor}")
    device = agent.device
    agent.actor.load_state_dict(T.load(actor, map_location=device))
    critic = checkpoint_dir / "Critic_ddpg"
    if critic.is_file():
        agent.critic.load_state_dict(T.load(critic, map_location=device))
    agent.actor.eval()
    agent.critic.eval()
