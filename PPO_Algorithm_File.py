import torch
import torch.nn.functional as F
import numpy as np

def compute_advantage(gamma, lmbda, td_delta):
    """用来计算广义优势估计"""
    # td_delta为每一个时间步的时序差分误差
    # 它等于当前时刻的reward加上gamma乘下一个状态的state value，再减去这个时间步的state value
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    "输入时某个状态，输出是状态的价值"

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    """定义PPO截断算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 定义网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练的轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.state_dim = state_dim

    def take_action(self, state):
        state = (
            torch.tensor(state, dtype=torch.float)
            .view(-1, self.state_dim)
            .to(self.device)
        )
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states_np = np.array(transition_dict["states"])
        next_states_np = np.array(transition_dict["next_states"])
        states = torch.tensor(states_np, dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(next_states_np, dtype=torch.float).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        td_target = rewards + self.gamma * self.critic(next_states) * (
            1 - dones
        )  # 时序差分目标
        td_delta = td_target - self.critic(states)  # 时序差分误差

        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )

        old_log_probs = torch.log(
            self.actor(states).gather(1, actions)
        ).detach()  # 这一步计算出了ln π(a|s)，该计算基于旧的actor，动作是该时间步真实进行的动作，注意我们通过detach方法使我们求导时不对这部分求导

        # 对一个episode采集得到的数据，训练epochs轮，在这里训练10轮，每多训练一轮，参数就会朝着从这轮采集得到的数据来看最好的方向进行
        # 但是，我们并不知道这轮采集到的数据好不好以及此时的critic网络的估值是否准确，所以我们也不能训练太多epochs，以避免过拟合
        for _ in range(self.epochs):
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )  # 该步求出了critic网络的损失函数
            # 下面先对critic网络进行更新
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()  # 更新critic网络参数

            # 之后对actor网络进行更新
            log_probs = torch.log(
                self.actor(states).gather(1, actions)
            )  # 算出新的策略下的ln π(a|s)
            # 一开始新旧策略相同，即比值为1，目标函数就是优势函数，我们仍旧可以通过求导得出此时应该进行梯度下降的方向
            ratio = torch.exp(log_probs - old_log_probs)  # 算出两策略的比值
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(
                -torch.min(surr1, surr2)
            )  # 算出目标函数（加符号变为损失函数），我们的目标是对actor网络的参数求导，使得actor网络的参数向着损失值最小的方向变化
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
