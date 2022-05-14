import torch
import torch.nn as nn

from uav2_charge1.exper_ppo_convmap.utils import AddBias, init, init_normc_

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        self.device = device
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = nn.Sequential(
            # nn.Dropout(0.5),#TODO 2018-11-21
            # nn.BatchNorm1d(num_inputs),  # TODO 2018-11-21
            init_(nn.Linear(num_inputs, num_outputs)),
        )

    def forward(self, x):
        x = self.linear(x)
        # print 'before',x[0].cpu().numpy()
        # x = F.softmax(x)
        # print 'after',x[0].cpu().numpy()
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        self.device = device
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = nn.Sequential(
            # nn.Dropout(0.5),#TODO 2018-11-21
            # nn.BatchNorm1d(num_inputs),  # TODO 2018-11-21
            init_(nn.Linear(num_inputs, num_outputs)),
        )
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)  # TODO new
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.to(self.device)

        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)  # TODO new
        return FixedNormal(action_mean, action_logstd.exp())
