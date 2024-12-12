import torch
from config import BaseConfig, Configurable
from sampling import SampleBuffer
from darrl import DARRL
from torch_util import Module, DummyModuleWrapper, device


class DR(Configurable, Module):
    class Config(BaseConfig):
        darrl_cfg = DARRL.Config()
        safe_horizon = 100
        buffer_min = 5
        buffer_max = 10**6

    def __init__(self, config, env, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)

        self.env = env
        self.state_dim, self.action_dim = state_dim, action_dim

        self.solver = DARRL(self.darrl_cfg, self.state_dim, self.action_dim)

        self.replay_buffer = self._create_buffer(self.buffer_max)

        self.safe_interaction = 0
        self.safe_horizon = 10
        self.env_model_error = 0.0

    @property
    def actor(self):
        return self.solver.actor

    def _create_buffer(self, capacity):
        buffer = SafetySampleBuffer(self.state_dim, self.action_dim, capacity)
        buffer.to(device)
        return DummyModuleWrapper(buffer)

    def update_solver(self):
        solver = self.solver
        # samples = self.replay_buffer.sample(solver.batch_size)
        solver.update(self.replay_buffer)
        # solver.update_actor(samples[0])

    def interaction(self, max_episode_steps, score, v, v_epi, speed_range, max_a, n_epi, warm_up=10, scale=10.0):
        episode = self._create_buffer(max_episode_steps)
        state = self.env.reset()

        for t in range(max_episode_steps):
            print("时间步: ",t)
            if n_epi > warm_up:
                self.update_solver()
            mu, _, pi = self.actor(torch.tensor(state))
            action = max_a*pi.item()

            reward, next_state, done, r_, cost, info = self.env.step(action)
            print("reward:",reward)
            for buffer in [episode, self.replay_buffer]:
                buffer.append(states=torch.tensor(state), actions=pi.detach(), next_states=torch.tensor(next_state), rewards=reward/scale, costs=cost, dones=done)

            state = next_state
            score += reward
            v.append(state[24]*speed_range)
            v_epi.append(state[24]*speed_range)
            xa = info[0]
            ya = info[1]

            if done is False:
                self.safe_interaction += 1
            else:
                break

        self.safe_horizon = self.safe_interaction
        self.safe_interaction = 0

        if (n_epi+1) % 100 == 0:
            self.solver.save_model(int(score), self.env.spec.id)
            # self.save_model(self.env_model_error)
            print("#The models are saved!#", n_epi+1)
            print("======>env_model_error:", self.env_model_error)

        return score, v, v_epi, xa, ya, done
    def save_model(self, model_name):
        name = "./models/" + self.env.spec.id + "/env_model%d" % model_name
        torch.save([self.model_ensemble.state_dict(), self.model_ensemble.state_normalizer.mean, \
                    self.model_ensemble.state_normalizer.std, self.model_ensemble.state_normalizer.epsilon], "{}.pkl".format(name))


class SafetySampleBuffer(SampleBuffer):
    # COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, 'violations')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._create_buffer('violations', torch.bool, [])
