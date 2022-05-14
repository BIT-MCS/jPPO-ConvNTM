import glob
import os
import time
from collections import deque

from uav2_charge1.exper_ppo_convmap.algo.ppo import *
from uav2_charge1.exper_ppo_convmap.arguments import get_args
from uav2_charge1.exper_ppo_convmap.envs import *
from uav2_charge1.exper_ppo_convmap.storage import RolloutStorage

from uav2_charge1.exper_ppo_convmap.model import Policy
from util.utils import *

args = get_args()

torch.set_num_threads(1)
device = torch.device("cuda:2" if args.cuda else "cpu")
MODEL_PATH = '/home/linc/one_storage/experiment_ppo_data/experiment3/uav2_charge1/exper_ppo_convmap/2019/03-03/21-48-00/0/model.pth'
test_num = 50
util = Util(device)

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
# TODO new reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# TODO new reproducibility

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def action_convert(action):
    action = (util.to_numpy(action[0]), util.to_numpy(action[1]))
    num_of_process = action[0].shape[0]
    num_of_uav = int(action[1].shape[1] / 2)
    action_new = np.zeros([num_of_process, num_of_uav, 3], dtype=np.float32)

    for i in range(num_of_process):
        state = action[0][i, 0]
        for j in range(num_of_uav):
            if state % 2 == 0:
                action_new[i, j, 0] = 1  # collect data
            else:
                action_new[i, j, 0] = -1  # charge
            state = max(0, state - 1)

            action_new[i, j, 1:] = action[1][i, j * 2:j * 2 + 2]

    action_new = action_new.reshape([num_of_process, -1])
    return action_new


def main():
    if args.vis:
        from visdom import Visdom

        viz = Visdom(port=args.port)
        win = None

    envs = Make_Env(test_num, device, args.num_steps)

    actor_critic = Policy(envs.observ_shape, envs.num_of_uav, args.cat_ratio, args.dia_ratio, device,
                          base_kwargs={'recurrent': args.recurrent_policy,
                                       })
    actor_critic.to(device)

    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                args.value_loss_coef, args.entropy_coef, lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss=True  # TODO new
                )

    rollouts = RolloutStorage(args.num_steps, test_num,
                              envs.observ_shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              args.seq_length)

    rollouts.to(device)
    episode_rewards = deque(maxlen=10)

    start = time.time()
    v_l_lst = []
    a_l_lst = []
    entr_lst = []
    max_effi = 0.
    max_data_collection = 0.
    max_episode = 0
    actor_critic.load_state_dict(torch.load(MODEL_PATH))
    actor_critic.eval()  # TODO new 2018-11-21
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step],
                rollouts.recurrent_hidden_states_h[step],
                rollouts.masks[step])

        # Obser reward and next obs

        obs, reward, done, info = envs.step(action_convert(action), current_step=step)

        # for info in infos:
        #     if 'episode' in info.keys():
        #         episode_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                   for done_ in done])
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
    envs.draw_path(step=0)
    envs.test_summary()
    print('mean_effi {:.2f} mean_data_collection_ratio {:.2f}  '.format(envs.mean_efficiency,
                                                                        envs.mean_data_collection_ratio))


if __name__ == "__main__":
    main()
