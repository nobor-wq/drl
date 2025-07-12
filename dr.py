import torch
from config import BaseConfig, Configurable
from sampling import SampleBuffer
from darrl import DARRL
from torch_util import Module, DummyModuleWrapper
from FGSM import *
import wandb
from utils import get_config

parser = get_config()
args = parser.parse_args()

class DR(Configurable, Module):
    class Config(BaseConfig):
        darrl_cfg = DARRL.Config()
        buffer_min = 5
        buffer_max = 10**6
        solver_updates_per_step = 5

    def __init__(self, config, env, state_dim, action_dim, adv_action_dim, device):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.device = device

        self.env = env
        self.state_dim, self.action_dim, self.adv_action_dim = state_dim, action_dim, adv_action_dim
        self.solver = DARRL(self.darrl_cfg, self.state_dim, self.action_dim, self.adv_action_dim, self.device).to(self.device)

        self.replay_buffer = self._create_buffer(self.buffer_max)
        # self.replay_buffer_eps = self._create_buffer(self.buffer_max)
        # self.replay_buffer_adv = self._create_buffer(self.buffer_max)

        self.test_episode = 100
        self.is_wandb = False
        self.env_name = args.env_name
        self.epsilon = args.episode



    @property
    def actor(self):
        return self.solver.actor.to(self.device)

    @property
    def adv_actor(self):
        return self.solver.actor_adv.to(self.device)

    @property
    def adv_critic(self):
        return self.solver.critic_adv.to(self.device)


    def _create_buffer(self, capacity):
        buffer = SafetySampleBuffer(self.state_dim, self.action_dim, capacity)
        buffer.to(self.device)
        return DummyModuleWrapper(buffer)

    def update_solver(self, loss_path):
        # for _ in range(self.solver_updates_per_step):
        solver = self.solver
        solver.update(self.replay_buffer, self.epsilon, loss_path)

    def test(self, is_attack):
        cn = 0.0
        sn = 0.0
        all_stds = []
        for episode in range(self.test_episode):
            state, _ = self.env.reset(options = "seed")
            for _ in range(30):
                if is_attack:
                    adv_action, _, _ = self.adv_actor(torch.tensor(state, dtype=torch.float32, device=self.device))
                    state_adv = FGSM_vdarrl(adv_action, self.actor, (torch.tensor(state, dtype=torch.float32)), methond="test", epsilon=self.epsilon, algo="drl")
                    ego_action_attack, _, _  = self.actor(state_adv)
                    next_state, reward, done, _, info = self.env.step(ego_action_attack)
                else:
                    ego_action, std, _  = self.actor(torch.tensor(state, dtype=torch.float32))
                    all_stds.append(std.detach().mean().item())
                    next_state, reward, done, _, info = self.env.step(ego_action)


                state = next_state

                xa = info['x_position']
                ya = info['y_position']

                if done:
                    cn += 1
                    break

            if self.env_name == 'TrafficEnv1-v0' or self.env_name == 'TrafficEnv3-v1' or self.env_name == 'TrafficEnv6-v0':
                if xa < -50.0 and ya > 4.0 and done is False:
                    sn += 1
            elif self.env_name == 'TrafficEnv2-v0':
                if xa > 50.0 and ya > -5.0 and done is False:
                    sn += 1
            elif self.env_name == 'TrafficEnv4-v0':
                if ya < -50.0 and done is False:
                    sn += 1
            elif self.env_name == 'TrafficEnv8-v0':
                if ya == 10.0 and done is False:
                    sn += 1
        self.env.close()
        # 计算碰撞率
        ct = cn / self.test_episode
        st = sn / self.test_episode

        if is_attack:
            wandb.log({"Test-cn-WithAttack": ct})
            return ct
        else:

            stds = np.array(all_stds)
            std_mean = stds.mean()
            std_std = stds.std()

            wandb.log({
                "Test-sn": st,
                "Test-std_mean": std_mean,
                "Test-std_std": std_std,
            })
            return st


    def interaction(self, max_episode_steps, score, step, step_attack, step_attack_sn, v, v_epi, speed_range, n_epi, modelSavedPath, loss_path, warm_up=30):

        state, _ = self.env.reset()

        for t in range(max_episode_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            step += 1
            if n_epi > warm_up:
                self.update_solver(loss_path)
            # print("state: ", state)
            _, action_adv, _  = self.adv_actor(state_tensor)
            adv_state = FGSM_vdarrl(action_adv, self.actor, state_tensor,
                                    epsilon=self.epsilon, algo=args.algo)

            _, _, ego_action_attack = self.actor(adv_state)
            # if n_epi > warm_up:
            #     with open(self.log_path, 'a', newline='') as f:
            #         writer = csv.writer(f)
            #         writer.writerow([
            #             action_loss.detach().cpu().item(),
            #             policy_loss.detach().cpu().item(),
            #             self.lam1,
            #             self.lam2,
            #             actor_loss.detach().cpu().item()
            #         ])

            # state = adv_state
            action = ego_action_attack
            next_state, reward, done, _, info = self.env.step(action)

            for buffer in [self.replay_buffer]:
                buffer.append(states=state_tensor,
                              adv_states=adv_state.clone().detach().to(self.device),
                              actions=action.detach().to(self.device),
                              adv_actions=action_adv.detach().to(self.device),
                              next_states=torch.tensor(next_state).to(self.device),
                              rewards=torch.tensor(reward, dtype=torch.float32, device=self.device),
                              costs=torch.tensor(info['cost'], dtype=torch.float32, device=self.device),
                              dones=torch.tensor(done, dtype=torch.bool, device=self.device))


            state = next_state
            score += reward

            v.append(state[24] * speed_range)
            v_epi.append(state[24] * speed_range)
            xa = info['x_position']
            ya = info['y_position']

            if done:
                break
                
        self.env.close()

        if (n_epi + 1) % 100 == 0 and (n_epi + 1) > 5000:
            self.solver.save_model(int(score),  modelSavedPath)
            print("#The models are saved!#", n_epi + 1)

        if (n_epi + 1) % 500 == 0:
            if self.is_wandb:
                st = self.test(False)
                ct = self.test(True)

        return score, step, step_attack, step_attack_sn, v, v_epi, xa, ya, done,


class SafetySampleBuffer(SampleBuffer):
    # COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, 'violations')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._create_buffer('violations', torch.bool, [])