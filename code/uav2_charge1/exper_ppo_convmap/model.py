from uav2_charge1.exper_ppo_convmap.distributions import Categorical, DiagGaussian

# from conv_map.conv_map_cell_2d_bn_1 import *
from conv_map.new_conv_map_cell_2d_brd_f1_l21_s6 import *
from uav2_charge1.exper_ppo_convmap.utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, num_of_uav, cat_ratio, dia_ratio, device, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.cat_ratio = cat_ratio
        self.dia_ratio = dia_ratio
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], device, **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        # self.dist_cat = nn.ModuleList([Categorical(self.base.output_size, 2) for i in range(num_uav)])
        # self.dist_dia = nn.ModuleList([DiagGaussian(self.base.output_size, 2) for i in range(num_uav)]) # dx dy

        # action_space --> num of uav
        self.dist_cat = Categorical(self.base.output_size, 2 ** num_of_uav, device)  # action_space[0]-->num of uav
        self.dist_dia = DiagGaussian(self.base.output_size, 2 * num_of_uav, device)  # (dx dy)*num_of_uav

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist_cat = self.dist_cat(actor_features)
        dist_dia = self.dist_dia(actor_features)

        action_cat = dist_cat.mode()
        action_dia = dist_dia.sample()
        # print action_cat[0].cpu().numpy()
        # print action_dia[0].cpu().numpy()
        action_log_probs_cat = dist_cat.log_probs(action_cat)
        dist_entropy_cat = dist_cat.entropy().mean()
        action_log_probs_dia = dist_dia.log_probs(action_dia)
        dist_entropy_dia = dist_dia.entropy().mean()

        return value, \
               (action_cat, action_dia), \
               (action_log_probs_cat * self.cat_ratio + action_log_probs_dia * self.dia_ratio), \
               rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist_cat = self.dist_cat(actor_features)
        dist_dia = self.dist_dia(actor_features)

        action_log_probs_cat = dist_cat.log_probs(action[0])
        dist_entropy_cat = dist_cat.entropy().mean()
        action_log_probs_dia = dist_dia.log_probs(action[1])
        dist_entropy_dia = dist_dia.entropy().mean()

        return value, \
               (action_log_probs_cat * self.cat_ratio + action_log_probs_dia * self.dia_ratio), \
               (dist_entropy_cat * self.cat_ratio + dist_entropy_dia * self.dia_ratio), \
               rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, hidden_size, feature_size, device):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._feature_size = feature_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = ConvMapCell(m_features=hidden_size[0],
                                   m_h=21*16,
                                   m_x=hidden_size[2],
                                   m_y=hidden_size[3],
                                   device=device)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._feature_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            size = hxs.size()
            hxs = (hxs.view(hxs.size(0), -1) * masks).view(size)

            x, hxs = self.gru(x, hxs)

        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, *x.shape[1:])

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hxs = (hxs.view(hxs.size(0), -1) * masks[i]).view(hxs.size())
                x_, hxs = self.gru(x[i], hxs)
                outputs.append(x_)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, device, recurrent=False, hidden_size=(16, int(21*16/16), 6, 6), feature_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, feature_size, device)
        self.feature_size = feature_size
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               # nn.init.xavier_uniform_,# TODO 2018-11-25
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            # nn.ELU(),
            # nn.LayerNorm([32, 19, 19]),
            # nn.LocalResponseNorm(32),
            nn.BatchNorm2d(32),  # TODO 2018-11-21
            # nn.Dropout(0.5),  # TODO 2018-11-21
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            # nn.ELU(),
            # nn.LayerNorm([64, 8, 8]),
            # nn.LocalResponseNorm(64),
            nn.BatchNorm2d(64),  # TODO 2018-11-21
            # nn.Dropout(0.5),  # TODO 2018-11-21

            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            # nn.ELU(),
            # nn.LayerNorm([32, 6, 6]),
            # nn.LocalResponseNorm(32),
            nn.BatchNorm2d(32),  # TODO 2018-11-21
            # nn.Dropout(0.5),  # TODO 2018-11-21

            # Flatten(),
            # init_(nn.Linear(32 * 6 * 6, hidden_size)),
            # nn.ReLU()
        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.convlstm_to_linear = nn.Sequential(
            Flatten(),
            # nn.Dropout(0.5),    #TODO 2018-11-21

            init_(nn.Linear(32 * 6 * 6, self.feature_size)),
            # nn.BatchNorm1d(self.feature_size),  # TODO 2018-11-21
            nn.ReLU(),
            # nn.ELU(),
        )

        self.critic_linear = nn.Sequential(
            # nn.Dropout(0.5),    # TODO 2018-11-21
            # nn.BatchNorm1d(self.feature_size),  # TODO 2018-11-21
            init_(nn.Linear(self.feature_size, 1)),
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
            x = self.convlstm_to_linear(x)
        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
