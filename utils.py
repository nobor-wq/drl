import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='IL_STA', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--env_name', default="TrafficEnv3-v1", help='name of the environment to run')
    parser.add_argument('--algo', default="drl", help='name of the alg')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--train_step', type=int, default=12)

    parser.add_argument('--T_horizon', default=30)
    parser.add_argument('--print_interval', default=10)
    parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
    parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
    parser.add_argument('--state_dim', default=26)
    parser.add_argument('--action_dim', default=1)


  # parser.add_argument('--wandb', type=bool, default=False, help='whether to use wandb logging')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb logging')
    parser.add_argument('--swanlab', action='store_true', help='whether to use wandb logging')
    parser.add_argument('--addition_msg', default="", help='additional message of the training process')
    parser.add_argument('--device', default="cpu", help='cpu or cuda:0 pr cuda:1')
    parser.add_argument('--parallel_id', type=int, default=0,  help='' )

    parser.add_argument('--attacker', action='store_true', help='whether to train for attacker')
    parser.add_argument('--critic', action='store_true', help='约束1中评估扰动状态还是干净状态')
    parser.add_argument('--update', action='store_true', help='更新方法修改')
    parser.add_argument('--grad', action='store_true', help='更新梯度修改')
    parser.add_argument('--replay', action='store_true', help='是否使用碰撞缓冲区')
    parser.add_argument('--lag', action='store_true', help='是否使用标准拉格朗日乘数法')
    parser.add_argument('--get', action='store_true', help='采样时不添加扰动')
    parser.add_argument('--frequency', action='store_true', help='防御者和攻击者更新频率')
    parser.add_argument('--epsilon', type=float, default=0.03, help='扰动强度')
    parser.add_argument('--model_name', default="264_18", help='训练攻击者时使用的防御者模型')
    parser.add_argument('--method', default="m1", help='防御者约束方法')
    parser.add_argument('--attack_option', default="a2", help='攻击方法')

    return parser
