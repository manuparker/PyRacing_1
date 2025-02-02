import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race

from PPO_Algorithm_File import compute_advantage
from PPO_Algorithm_File import PPO_1
from PPO_Algorithm_File import train_on_policy_agent
from PPO_Algorithm_File import moving_average
from PPO_Algorithm_File import PPO_1_Test
from PPO_Algorithm_File import continue_training

from stable_baselines3 import PPO
import torch



def load_and_play():
    print("Start loading history")
    history_list = ['30000.npy']

    # load data from history file
    print("Start updating q_table")
    discount_factor = 0.99
    for list in history_list:
        history = load_data(list)
        learning_rate = get_learning_rate(0)
        print(list)
        file_size = len(history)
        print("file size : " + str(file_size))
        i = 0
        for data in history:
            state_0, action, reward, state, done = data
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            if done == True:
                i += 1
                learning_rate = get_learning_rate(i)

    print("Updating q_table is complete")


    # play game
    env.set_view(True)
    reward_count = 0
    for episode in range(NUM_EPISODES):
        obv = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0
        for t in range(MAX_T):
            action = select_action(state_0, 0.01)
            obv, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = state_to_bucket(obv)
            total_reward += reward
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            state_0 = state
            env.render()
            if done or t >= MAX_T - 1:
                print("Episode %d finished after %i time steps with total reward = %f."
                      % (episode, t, total_reward))
                break
        if total_reward >= 1000:
            reward_count += 1
        else:
            reward_count = 0

        if reward_count >= 10:
            env.set_view(True)

        learning_rate = get_learning_rate(i + episode)


def load_and_simulate():
    print("Start loading history")
    history_list = ['30000.npy']

    # load data from history file
    print("Start updating q_table")
    discount_factor = 0.99
    i = 0
    for list in history_list:
        history = load_data(list)
        learning_rate = get_learning_rate(0)
        print(list)
        file_size = len(history)
        print("file size : " + str(file_size))
        for data in history:
            state_0, action, reward, state, done = data
            env.remember(state_0, action, reward, state, done)
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            if done == True:
                i += 1
                learning_rate = get_learning_rate(i)

    print("Updating q_table is complete")


    # simulate
    env.set_view(False)
    for episode in range(NUM_EPISODES):
        obv = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0

        if episode > 3000 and episode <= 3010:
            if episode == 3001:
                env.save_memory('3000_aft')
            env.set_view(True)
        elif episode > 5000 and episode <= 5010:
            if episode == 5001:
                env.save_memory('5000_aft')
            env.set_view(True)

        for t in range(MAX_T):
            action = select_action(state_0, 0.01)
            obv, reward, done, _ = env.step(action)
            state = state_to_bucket(obv)
            env.remember(state_0, action, reward, state, done)
            state_0 = state
            total_reward += reward
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            env.render()
            if done or t >= MAX_T - 1:
                print("Episode %d finished after %i time steps with total reward = %f."
                      % (episode, t, total_reward))
                break

        learning_rate = get_learning_rate(i + episode)



def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q_table[state]))
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def load_data(file):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(file)
    np.load = np_load_old
    return data

def simulate():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99
    total_reward = 0
    total_rewards = []
    training_done = False
    threshold = 1000
    env.set_view(True)
    for episode in range(NUM_EPISODES):

        total_rewards.append(total_reward)
        if episode == 50000:
            plt.plot(total_rewards)
            plt.ylabel('rewards')
            plt.show()
            env.save_memory('50000')
            break

        obv, _ = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate)
            obv, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = state_to_bucket(obv)
            env.remember(state_0, action, reward, state, done)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state
            env.render()
            if done or t >= MAX_T - 1:
                print("Episode %d finished after %i time steps with total reward = %f."
                      % (episode, t, total_reward))
                break
        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def test_model(model, num_episodes):
    env = gym.make("Pyrace-v0")  # 可视化只能在初始化时指定
    env.unwrapped.set_view(True)
    sum_length = 0

    for i_episode in range(num_episodes):
        obs, _ = env.reset(seed=0)
        done1, done2 = False, False
        episode_len = 0
        episode_return = 0
        while not (done1 or done2):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done1, done2, info = env.step(action)

            episode_return += reward
            episode_len += 1
            env.render()
        sum_length += episode_len

        print(f"Episode {i_episode + 1}: Total Reward = {episode_return}")
        print(f"Episode {i_episode + 1}: Total Length = {episode_len}")

    average_len = int(sum_length / num_episodes)

    print(f"一个episode平均使用{average_len}步到达终点")


def MY_PPO_test(env, model_path, num_episodes):
    # 创建PPO_1_Test类的实例
    sum_length = 0
    hidden_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.unwrapped.set_view(True)   # 设置训练过程小车运动可视化
    agent = PPO_1_Test(
        state_dim,
        hidden_dim,
        action_dim,
        device
    )

    # 加载模型状态
    agent.load_model(model_path)

    for i_episode in range(num_episodes):
        state, _ = env.reset(seed=0)
        done = False
        episode_return = 0
        episode_len = 0
        while not done:
            action = agent.take_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_len += 1
            env.render()   # 可视化环境
        sum_length += episode_len

        print(f"Episode {i_episode + 1}: Total Reward = {episode_return}")
        print(f"Episode {i_episode + 1}: Total Length = {episode_len}")

    average_len = int(sum_length/num_episodes)

    print(f"一个episode平均使用{average_len}步到达终点")

def MY_PPO_train(env, env_name, model_path):
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 200
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
        device,
    )

    return_list, len_list = train_on_policy_agent(env, agent, num_episodes, model_path)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("PPO on {}".format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("Moving_average PPO on {}".format(env_name))
    plt.show()

    plt.plot(episodes_list, len_list)
    plt.xlabel("Episodes")
    plt.ylabel("Length")
    plt.title("Length on {}".format(env_name))
    plt.show()

    mv_length = moving_average(len_list, 9)
    plt.plot(episodes_list, mv_length)
    plt.xlabel("Episodes")
    plt.ylabel("Length")
    plt.title("Moving_average Length on {}".format(env_name))
    plt.show()


if __name__ == "__main__":

    env_name = "Pyrace-v0"
    env = gym.make(env_name)
    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    print(DECAY_FACTOR)

    NUM_EPISODES = 9999999
    MAX_T = 2000
    #MAX_T = np.prod(NUM_BUCKETS, dtype=int) * 100

    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    "训练过程"
    # model = PPO("MlpPolicy", env, verbose=1,
    #             learning_rate=3e-3, batch_size=256,
    #             tensorboard_log='./log/ppo_5')  # 创建模型
    #
    # model.learn(total_timesteps=200000)  # 训练模型
    #
    # model.save("Model/ppo_5")

    "加载模型测试过程"
    # model = PPO.load("Model/ppo_4", env=env)
    # test_model(model, 10)

    "使用《动手学强化学习》的PPO代码进行训练"
    # MY_PPO_train(env, env_name, 'Model/my_ppo_3.pth')

    "使用《动手学强化学习》的PPO代码进行测试"
    MY_PPO_test(env, "Model/my_ppo_3.pth", 10)

    "使用《动手学强化学习》的PPO代码进行继续训练"
    # continue_training(env, "Model/my_ppo_3.pth", 20, 'Model/my_ppo_4.pth')

    # simulate()
    #load_and_play()
