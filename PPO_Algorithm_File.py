import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PPO_1_Test:
    """定义PPO截断算法用于测试，只包含actor网络"""

    def __init__(self, state_dim, hidden_dim, action_dim, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.device = device
        self.state_dim = state_dim

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.state_dim).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def load_model(self, model_path):
        model_states = torch.load(model_path)
        self.actor.load_state_dict(model_states['actor_state_dict'])
        self.actor.to(self.device)


def continue_training(env, model_path, num_episodes_to_continue):
    actor_lr = 8e-4
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.unwrapped.set_view(False)   # 设置训练过程小车运动可视化

    # 加载保存的模型状态
    saved_model_states = torch.load(model_path)

    # 创建PPO_1类的实例
    agent = PPO_1(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device
    )

    # 应用保存的状态到模型
    agent.actor.load_state_dict(saved_model_states['actor_state_dict'])
    agent.critic.load_state_dict(saved_model_states['critic_state_dict'])

    # 继续训练
    return train_on_policy_agent(env, agent, num_episodes_to_continue)


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
    advantage_np = np.array(advantage_list)
    return torch.tensor(advantage_np, dtype=torch.float)


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

class PPO_1:
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

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % (i + 1)) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }
                state, _ = env.reset(seed=0)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                    # env.render()   # 设置训练过程小车运动可视化
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episodes": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)

    # 创建包含多个状态字典的大字典
    model_states = {
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
    }
    
    # 保存这个大字典到一个文件
    model_path = "ppo_combined_model.pth"
    torch.save(model_states, model_path)

    return return_list

def moving_average(a, window_size):
    """使用moving_average的方法对数据进行平滑处理，a为传入的return数据列表，window_size为取平均的个数"""
    """以window_size=9为例，该平均算法的逻辑是，每个位置，取自己和前面4个数据以及后面4个数据共9个数据进行平均操作"""
    """针对前4个数据与后4个数据，该算法做特殊处理"""
    """第一个数据：直接用其本身"""
    """第二个数据：用前3个数据之和除以3"""
    """第三个数据：用前5个数据之和除以5"""
    """第四个数据：用前7个数据之和除以7"""
    """倒数第一个数据：直接用其本身"""
    """倒数第二个数据：用后3个数据之和除以3"""
    """倒数第三个数据：用后5个数据之和除以5"""
    """第四个数据：用后7个数据之和除以7"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  # 在a前插入0之后，算每个位置之前的所有元素的和
    # 加入0的作用是能够得到第1个元素到第9个元素的累积和
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size  # 利用累计和切片：（第9个元素到最后 - 第1个元素到倒数第10个元素）/ 9 = 第5个位置到倒数第5个位置的均值
    r = np.arange(1, window_size - 1, 2)  # 得到r = [1 ,3, 5, 7]
    begin = (
        np.cumsum(a[: window_size - 1])[::2] / r
    )  # 得到前8个元素的和并分别取1、3、5、7位置的和除以r作为前4个位置的均值
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[
        ::-1
    ]  # 得到后8个元素的和并分别取最后1、3、5、7位置的和除以r作为后4个位置的均值
    if window_size % 2 == 0:  # 该条件补充了window_size为偶数的情况，若不加该语句，则偶数时平滑后的输出比原数组元素少1
        end = np.insert(
            end,
            -int((window_size / 2)),
            (cumulative_sum[-1] - cumulative_sum[-window_size]) / (r[-1] + 2),
        )
    return np.concatenate((begin, middle, end))  # 将其拼到一起，则为平滑后的数据

