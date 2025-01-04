import numpy as np
import torch
import gym
import copy
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 新增一层
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 新增层的激活
        probs = self.fc3(x)
        log_probs = F.log_softmax(probs, dim=1)
        return torch.exp(log_probs)  # 去掉不必要的 probs 计算


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 新增一层
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 新增层的激活
        return self.fc3(x)

class nnModel(object):
    def __init__(self, url1, url2, container_size, hidden_dim):
        area = container_size[0]*container_size[1]
        self.hidden_dim = hidden_dim
        self.alen = area * 6 + 1
        self.olen = area * 4
        self.height = container_size[2]
        self.device = torch.device("cpu")
        self.policy_net, self.value_net = self._load_model(url1, url2)

    def _load_model(self, url1, url2):
        # Load pre-trained model and observation normalization data (ob_rms)
        policy_model_pretrained = torch.load(url1)
        value_model_pretrained = torch.load(url2)
        policy_net = PolicyNet(self.olen, self.hidden_dim, self.alen)
        value_net = ValueNet(self.olen, self.hidden_dim)
        policy_net.load_state_dict(policy_model_pretrained)
        value_net.load_state_dict(value_model_pretrained)
        return policy_net, value_net

    def evaluate(self, obs, action_mask=None):
        """
        根据输入状态 obs 评估策略分布和价值。
        :param obs: 输入状态（单个状态或批量状态）
        :param use_mask: 是否应用动作掩码
        :param action_mask: 动作掩码（用于屏蔽非法动作），形状与动作维度一致
        :return: (value, action_probs)
                 value: 状态的价值
                 action_probs: 动作的概率分布（可能被 mask 修改）
        """
        # 转换输入为 tensor，并将其移动到指定设备
        x = copy.deepcopy(obs)
        x = torch.tensor([x], dtype=torch.float).to(self.device)
        # 获取策略分布（action_probs）和状态价值（value）
        with torch.no_grad():
            action_probs = self.policy_net(x)  # 策略网络输出 logits
            value = self.value_net(x)  # 价值网络输出
        # 使用 softmax 计算动作的概率分布
        action_probs = action_probs.numpy()  # 转为 numpy 数组
        value = value.squeeze().item()  # 获取标量值
        # 如果启用了动作掩码，应用掩码

        if action_mask is not None:
            assert action_mask.shape == action_probs.shape[1:], "掩码形状必须与动作空间匹配"
            action_probs = action_probs * action_mask
            action_probs /= np.sum(action_probs, axis=1, keepdims=True)  # 重新归一化

        action_probs = action_probs.reshape(-1)
        return value, action_probs


    def sample_action(self, obs, action_mask=None, mask_weight=10.0):
        """
        从给定状态中采样动作。
        :param obs: 当前状态，形状为 (state_dim,)
        :param action_mask: 动作掩码，形状为 (action_dim,)；可选，为 None 时表示无掩码。
        :param mask_weight: 掩码的权重放大因子。
        :return: 当前状态的价值 (value)，以及采样的动作 (action)
        """
        # 将观测转化为张量并移动到设备
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # 加入批次维度
        # 使用 ValueNet 估算状态价值
        value = self.value_net(x).item()
        # 使用 PolicyNet 预测动作概率分布
        probs = self.policy_net(x).squeeze(0)  # 去掉批次维度
        #probs = torch.softmax(logits, dim=-1)  # 转化为概率分布
        # 如果提供了动作掩码，调整动作概率分布
        if action_mask is not None:
            action_mask = torch.FloatTensor(action_mask).to(self.device)  # 转化为张量
            adjusted_probs = probs * action_mask + (1 - action_mask) * 1e-8  # 避免非法动作概率为 0
            adjusted_logits = torch.log(adjusted_probs + 1e-8)  # 转化回 logits
            adjusted_logits += action_mask * mask_weight  # 增强合法动作的概率
        else:
            adjusted_logits = logits
        # 使用调整后的 logits 创建动作分布
        action_dist = torch.distributions.Categorical(logits=adjusted_logits)
        # 从分布中采样动作
        action = int(action_dist.sample().item())
        return value, action