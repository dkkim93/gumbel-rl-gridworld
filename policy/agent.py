import numpy as np
from policy.policy_base import PolicyBase
from misc.replay_buffer import ReplayBuffer


class Agent(PolicyBase):
    def __init__(self, env, tb_writer, log, args, name):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, name=name)

        self.set_dim()
        self.set_policy()
        self.memory = ReplayBuffer()
        self.epsilon = 1  # For exploration

    def set_dim(self):
        self.actor_input_dim = self.env.observation_space.shape[0]
        self.actor_output_dim = self.env.action_space.n
        self.critic_input_dim = self.actor_input_dim + self.actor_output_dim
        self.n_hidden = self.args.n_hidden

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))

    def select_deterministic_action(self, obs):
        action = self.policy.select_action(obs)
        assert not np.isnan(action).any()

        return action

    def select_stochastic_action(self, obs, total_timesteps):
        if np.random.rand() > self.epsilon:
            action = self.policy.select_action(obs)
        else:
            action = np.zeros((self.args.n_action,), dtype=np.float32)
            action[np.random.randint(low=0, high=self.args.n_action, size=(1,))] = 1

            if self.epsilon > 0.05:
                self.epsilon *= 0.9999  # Reduce epsilon over time

        assert not np.isnan(action).any()

        self.tb_writer.add_scalar(
            "debug/epsilon", self.epsilon, total_timesteps)

        return action

    def add_memory(self, obs, new_obs, action, reward, done):
        self.memory.add((obs, new_obs, action, reward, done))

    def clear_tmp_memory(self):
        self.tmp_memory.clear()

    def update_policy(self, total_timesteps):
        if len(self.memory) > self.args.ep_max_timesteps:
            debug = self.policy.train(
                replay_buffer=self.memory,
                iterations=self.args.ep_max_timesteps,
                batch_size=self.args.batch_size, 
                discount=self.args.discount, 
                tau=self.args.tau, 
                policy_freq=self.args.policy_freq)

            self.tb_writer.add_scalars(
                "loss/actor", {self.name: debug["actor_loss"]}, total_timesteps)
            self.tb_writer.add_scalars(
                "loss/critic", {self.name: debug["critic_loss"]}, total_timesteps)
