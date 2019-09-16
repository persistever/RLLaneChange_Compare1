# coding:utf-8

from lane_change_env import Env
from RL_brain import DQN
from surrounding import Traffic
import numpy as np
import random
import time


def run_task(env, no_gui, max_episode, net=None, traffic=None):
    start = time.clock()
    step = 0
    for episode in range(max_episode):

        # generate traffic for each episode
        if traffic is not None:
            traffics_base = random.uniform(0.8*traffic, 1.2*traffic)
        else:
            traffics_base = random.uniform(0.5, 0.7)
        traffics = Traffic(trafficBase=traffics_base, trafficList=None)
        done = False

        # check plt and gui
        if (episode+1) % 500 == 0:
            net.plot_cost(length=100)
            observation = env.reset(nogui=False)
        else:
            observation = env.reset(nogui=no_gui)
        observation = np.array(observation)

        # main loop
        while done is False:
            print('Episode '+str((episode+1)))
            print('Make decision '+str(step))
            action_high, action_low = net.choose_action(observation)
            observation_, reward, done, info = env.step(action_high=action_high, action_low=action_low)
            observation_ = np.array(observation_)
            print("action_high: " + str(action_high) + " action_low: " + str(action_low))
            if done is not True:
                net.store_transition(observation, action_low, reward, observation_)
                if step > 50 and step % 2 == 0:
                    net.learn()
            print("reward: " + str(reward))
            print("info: "+str(info))
            print("-------------------------")
            if (step+1) % 100 == 0:
                elapsed = (time.clock() - start)
                print("Step: "+str(step+1)+" Time Used: "+str(elapsed))
            observation = observation_
            step += 1

        # save weight
        if (episode + 1) % 5 == 0:
            net.save()


if __name__ == "__main__":
    LC_env = Env(ego_start_time=100)

    # start learn
    dqn = DQN(n_features=7, e_greedy_start=0.6, e_greedy_increment=0.01, is_save=True, is_restore=False)
    run_task(env=LC_env, no_gui=True, max_episode=50, net=dqn, traffic=None)

    # # continue learn
    # dqn = DQN(n_features=7, e_greedy_start=0.8, e_greedy_increment=0.01, is_save=True, is_restore=True)
    # run_task(env=LC_env, no_gui=True, max_episode=200, net=dqn, traffic=None)

    # # show learn result
    # dqn = DQN(n_features=7, e_greedy=1, e_greedy_start=1, e_greedy_increment=0.01, is_save=True, is_restore=True)
    # dqn.plot_cost()
    # dqn.plot_cost(length=300)
    # run_task(env=LC_env, no_gui=False, max_episode=4, net=dqn, traffic=None)



