import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='IL_STA', formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument('--seed', type=int, default=4)

    parser.add_argument('--T_horizon', default=30)
    parser.add_argument('--print_interval', default=10)
    parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
    parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
    parser.add_argument('--state_dim', default=26)
    parser.add_argument('--action_dim', default=1)
    parser.add_argument('--Advaction_dim', default=1)

    parser.add_argument('--wandb', action='store_true', help='whether to use wandb logging')
    parser.add_argument('--swanlab', action='store_true', help='whether to use wandb logging')
    parser.add_argument('--addition_msg', default="", help='additional message of the training process')
    parser.add_argument('--device', default="cpu", help='cpu or cuda:0 pr cuda:1')
    parser.add_argument('--episode', type=float, default=0.03, help='episode')
    parser.add_argument('--model_name', default="264_18", help='actor model name')
    parser.add_argument('--best_model', default=False, help='whether to load best model')
    parser.add_argument('--attack_method', default="fgsm", help='which mode to render, False or rgb_array or human')

    parser.add_argument('--use_gui', action='store_true', help='whether to use GUI')
    parser.add_argument('--to_gif', action='store_true', help='whether to convert picture to gif')
    parser.add_argument('--render_mode', default=None, help='which mode to render, False or rgb_array or human')
    parser.add_argument("--duration", type=int, default=500, help="Duration of each frame")
    parser.add_argument('--diff_between', action='store_true', help='whether to use GUI')

    parser.add_argument('--env_name', default="TrafficEnv3-v1", help='name of the environment to run')
    parser.add_argument('--train_step', type=int, default=30)
    parser.add_argument('--algo', default="drl", help='name of the alg')
    parser.add_argument('--adv_algo', default="drl", help='training adv algorithm')
    parser.add_argument('--attack', type=bool, default=True, help='control n_rollout_steps, for PPO')
    parser.add_argument('--algo_name', default="policy_v266_20250710_0457_2_23.pkl", help='defender algorithm')
    parser.add_argument('--age_path', default="./models/")
    parser.add_argument('--adv_path', default="./models/")
    parser.add_argument('--adv_algo_name', default="policy_v247_20250710_1726_2_23.pkl", help='attack algorithm')



    return parser