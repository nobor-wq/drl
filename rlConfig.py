# coding=utf8
import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='IL_STA', formatter_class=argparse.RawDescriptionHelpFormatter)

    # env parameters
    parser.add_argument('--env_name', default="TrafficEnv3-v0", help='name of the environment to run')
    parser.add_argument('--addition_msg', default="", help='additional message of the training process')
    parser.add_argument('--algo', default="PPO", help='training algorithm')
    parser.add_argument('--adv_algo', default="PPO", help='training adv algorithm')
    parser.add_argument('--path', default="./logs/eval/", help='path of the trained model')
    parser.add_argument('--train_step', type=int, default=200, help='number of training episodes')
    parser.add_argument('--T_horizon', type=int, default=30, help='number of training steps per episode')
    parser.add_argument('--print_interval', default=10)
    parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
    parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
    parser.add_argument('--state_dim', default=26)
    parser.add_argument('--action_dim', default=1)
    parser.add_argument('--best_model', default=False, help='whether to load best model')
    parser.add_argument('--eval', default=False, help='eval flag')

    # training parameters
    parser.add_argument('--use_cuda', default=True, help='Use GPU if available')
    parser.add_argument('--cuda_number', type=int, default=1, help='CUDA device number to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed for network')
    parser.add_argument('--save_freq', type=int, default=100000, help='frequency of saving the model')
    parser.add_argument('--n_steps', type=int, default=256, help='control n_rollout_steps, for PPO')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of training epochs')

    # log parameters
    parser.add_argument('--no_wandb', action='store_true', help='whether to use wandb logging')
    # fni parameters
    parser.add_argument('--model_path', default="", help='model path for FNI/DARRL/IL')

    # random env parameters

    # Attack parameters
    parser.add_argument('--attack', action='store_true', help='whether to change the env for attack')
    parser.add_argument('--adv_steps', type=int, default=4, help='number of adversarial attack steps')
    parser.add_argument('--attack_method', default="fgsm", help='which attack method to be applied')
    parser.add_argument('--attack_eps', default=0.1, help='epsilon-ball around init state')
    parser.add_argument('--attack_iteration', default=50, help='iterations for attack method')
    parser.add_argument('--step_size', default=0.0075, help='step size for fgsm')

    # ablation parameters
    parser.add_argument('--padding', action='store_true', help='whether to use padding or not')
    parser.add_argument('--clipping', action='store_true', help='whether to use clipping or not')
    parser.add_argument('--v_action_obs_flag', action='store_true',
                        help='whether to add the action of victim agent into the obs of the adversary')
    parser.add_argument('--remain_attack_time_flag', action='store_true',
                        help='whether to add the remain attack times into the obs of the adversary')

    # baseline parameters
    parser.add_argument('--unlimited_attack', action='store_true', help='For unlimited attack times baseline')

    return parser
