import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size, seq_length):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states_h = torch.zeros(num_steps + 1, num_processes, *recurrent_hidden_state_size)
        # self.recurrent_hidden_states_c = torch.zeros(num_steps + 1, num_processes, *recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        self.action_cat = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        self.action_dia = torch.zeros(num_steps, num_processes, int(action_space / 3 * 2))

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.seq_length = seq_length

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states_h = self.recurrent_hidden_states_h.to(device)
        # self.recurrent_hidden_states_c = self.recurrent_hidden_states_c.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_cat = self.action_cat.to(device)
        self.action_dia = self.action_dia.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states_h[self.step + 1].copy_(recurrent_hidden_states)
        # self.recurrent_hidden_states_c[self.step + 1].copy_(recurrent_hidden_states[1])
        self.action_cat[self.step].copy_(actions[0])
        self.action_dia[self.step].copy_(actions[1])
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states_h[0].copy_(self.recurrent_hidden_states_h[-1])
        # self.recurrent_hidden_states_c[0].copy_(self.recurrent_hidden_states_c[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    # the best one
    def recurrent_cut_more_random_more_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]  # 500 8
        batch_size = (num_steps / self.seq_length)  # (500/5) (num_step/seq_length)

        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch  # 100/4   choose 25 seqs from 8 envs, every seq is 5
        # sampler = BatchSampler(SubsetRandomSampler(range((num_steps - self.seq_length)* num_processes )),
        #                        (num_steps * num_processes // num_mini_batch), drop_last=False)
        sampler = BatchSampler(SubsetRandomSampler(range((num_steps - self.seq_length) * num_processes)),
                               (400), drop_last=False)
        # print num_steps * num_processes // num_mini_batch
        # sampler = BatchSampler(SubsetRandomSampler(range((num_steps - self.seq_length)* num_processes )),
        #                        (batch_size * num_processes // num_mini_batch), drop_last=False)

        obs_batch = []
        recurrent_hidden_states_batch_h = []
        # recurrent_hidden_states_batch_c = []
        action_batch_cat = []
        action_batch_dia = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        old_action_log_probs_batch = []
        adv_targ = []
        for start_ind in range(num_steps - self.seq_length):

            start = start_ind
            end = start + self.seq_length
            if end > num_steps:
                break

            obs_batch.append(self.obs[start:end, :])  # [5,8,3,80,80]
            recurrent_hidden_states_batch_h.append(self.recurrent_hidden_states_h[start:start + 1, :])
            # recurrent_hidden_states_batch_c.append(self.recurrent_hidden_states_c[start:start + 1, :])
            action_batch_cat.append(self.action_cat[start:end, :])
            action_batch_dia.append(self.action_dia[start:end, :])
            value_preds_batch.append(self.value_preds[start:end, :])
            return_batch.append(self.returns[start:end, :])
            masks_batch.append(self.masks[start:end, :])
            old_action_log_probs_batch.append(self.action_log_probs[start:end, :])
            adv_targ.append(advantages[start:end, :])
        # obs_batch = [] -> 495*[5,8,3,80,80]

        T, N = self.seq_length, (num_steps - self.seq_length) * num_processes  # 5  495*8
        # These are all tensors of size (T, N, -1)
        obs_batch = torch.cat(obs_batch, 1)  # (5, 800, 3, 80, 80)
        action_batch_cat = torch.cat(action_batch_cat, 1)
        action_batch_dia = torch.cat(action_batch_dia, 1)
        value_preds_batch = torch.cat(value_preds_batch, 1)
        return_batch = torch.cat(return_batch, 1)
        masks_batch = torch.cat(masks_batch, 1)
        old_action_log_probs_batch = torch.cat(old_action_log_probs_batch, 1)
        adv_targ = torch.cat(adv_targ, 1)
        # States is just a (N, -1) tensor
        # print(torch.cat(recurrent_hidden_states_batch_h, 1).shape)
        recurrent_hidden_states_batch_h = torch.cat(recurrent_hidden_states_batch_h, 1).squeeze(0)
        # recurrent_hidden_states_batch_c = torch.cat(recurrent_hidden_states_batch_c, 1).squeeze(0)

        # for indices in sampler:



        for indices in sampler:
            # Flatten the (T, N, ...) tensors to (T * N, ...)
            N_ = len(indices)

            obs_batch_ = _flatten_helper(T, N_, obs_batch[:, indices])
            action_batch_cat_ = _flatten_helper(T, N_, action_batch_cat[:, indices])
            action_batch_dia_ = _flatten_helper(T, N_, action_batch_dia[:, indices])
            value_preds_batch_ = _flatten_helper(T, N_, value_preds_batch[:, indices])
            return_batch_ = _flatten_helper(T, N_, return_batch[:, indices])
            masks_batch_ = _flatten_helper(T, N_, masks_batch[:, indices])
            old_action_log_probs_batch_ = _flatten_helper(T, N_, old_action_log_probs_batch[:, indices])
            adv_targ_ = _flatten_helper(T, N_, adv_targ[:, indices])
            recurrent_hidden_states_batch_ = recurrent_hidden_states_batch_h[indices]
            # recurrent_hidden_states_batch_c[indices])
            yield obs_batch_, recurrent_hidden_states_batch_, (action_batch_cat_, action_batch_dia_), \
                  value_preds_batch_, return_batch_, masks_batch_, old_action_log_probs_batch_, adv_targ_
