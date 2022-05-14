import glob
import os
import time
from collections import deque

from uav2_charge1.exper_ppo_convmap.algo.ppo import *
from uav2_charge1.exper_ppo_convmap.arguments import get_args
from uav2_charge1.exper_ppo_convmap.envs import *
from uav2_charge1.exper_ppo_convmap.storage import RolloutStorage
from uav2_charge1.exper_ppo_convmap.visualize import visdom_plot

from uav2_charge1.exper_ppo_convmap.model import Policy
from util.utils import *

args = get_args()

torch.set_num_threads(1)
device = torch.device("cuda:1" if args.cuda else "cpu")
util = Util(device)

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes
print(args.value_loss_coef,args.gamma)
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

    envs = Make_Env(args.num_processes, device, args.num_steps)

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

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
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
    for j in range(num_updates):
        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        actor_critic.eval()  # TODO new 2018-11-21
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
        envs.draw_path(step=j)
        # TODO new save model
        if envs.mean_efficiency > max_effi:
            # save
            model_path = envs.log_path + '/model.pth'
            torch.save(actor_critic.state_dict(), model_path)
            print('model has been save to', model_path)
            max_effi = envs.mean_efficiency
            max_data_collection = envs.mean_data_collection_ratio
            max_episode = j
        # TODO new save model
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states_h[-1],

                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        actor_critic.train()  # TODO new 2018-11-21
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()
        v_l_lst.append(value_loss)
        a_l_lst.append(action_loss)
        entr_lst.append(dist_entropy)
        np.savez(envs.log_path + '/loss.npz', np.asarray(v_l_lst), np.asarray(a_l_lst), np.asarray(entr_lst))

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            # if args.cuda:
            #     save_model = copy.deepcopy(actor_critic).cpu()
            #     torch.save(save_model, os.path.join(save_path, 'uav' + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        # TODO new save model
        f_path = envs.log_path + '/train_log.txt'
        f = open(f_path, 'a')
        if j % args.log_interval == 0:
            end = time.time()
            print_str = "Updates {}, num timesteps {}, FPS {} \n mean_episod_reward {:.1f} max/min_epsoidreward {:.1f}/{:.1f} value_loss {} action_loss {}\nmean_eff {:.2f} mean_dc {:.2f} save epsoide {} max_effi {:.2f} max_data_collection {:.2f}".format(
                j, total_num_steps,
                int(total_num_steps / (end - start)),
                info['mean_episod_reward'],
                info['max_episod_reward'],
                info['min_episod_reward'],
                value_loss, action_loss,
                envs.mean_efficiency,
                envs.mean_data_collection_ratio,
                max_episode,
                max_effi,
                max_data_collection
            )
            print(print_str)
            f.writelines(print_str + '\n')
        f.close()
        # TODO new save model
        # if (args.eval_interval is not None
        #     and len(episode_rewards) > 1
        #     and j % args.eval_interval == 0):
        #     eval_envs = make_vec_envs(
        #         args.env_name, args.seed + args.num_processes, args.num_processes,
        #         args.gamma, eval_log_dir, args.add_timestep, device, True)
        #
        #     vec_norm = get_vec_normalize(eval_envs)
        #     if vec_norm is not None:
        #         vec_norm.eval()
        #         vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
        #
        #     eval_episode_rewards = []
        #
        #     obs = eval_envs.reset()
        #     eval_recurrent_hidden_states = torch.zeros(args.num_processes,
        #                                                actor_critic.recurrent_hidden_state_size, device=device)
        #     eval_masks = torch.zeros(args.num_processes, 1, device=device)
        #
        #     while len(eval_episode_rewards) < 10:
        #         with torch.no_grad():
        #             _, action, _, eval_recurrent_hidden_states = actor_critic.act(
        #                 obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
        #
        #         # Obser reward and next obs
        #         obs, reward, done, infos = eval_envs.step(action)
        #
        #         eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
        #                                         for done_ in done])
        #         for info in infos:
        #             if 'episode' in info.keys():
        #                 eval_episode_rewards.append(info['episode']['r'])
        #
        #     eval_envs.close()
        #
        #     print(" Evaluation using {} episodes: mean reward {:.5f}\n".
        #           format(len(eval_episode_rewards),
        #                  np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass


if __name__ == "__main__":
    main()
