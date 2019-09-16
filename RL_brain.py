# coding:utf-8
import numpy as np
import tensorflow as tf
import random
import math
import operator

np.random.seed(1)
tf.set_random_seed(1)


class DQN:
    def __init__(
            self,
            n_features,
            n_actions_l=5,
            n_actions_r=5,
            learning_rate=0.01,
            reward_decay=0.7,
            e_greedy=0.99,
            replace_target_iter=100,
            memory_size=500,
            batch_size=32,
            e_greedy_start=0.5,
            e_greedy_increment=None,
            output_graph=False,
            is_restore=False,
            is_save=False,
            save_path="data/model/",
            restore_path=None
    ):
        # global model parameter
        self.n_actions_l = n_actions_l
        self.n_actions_r = n_actions_r
        self.n_actions = n_actions_l + 1 + n_actions_r
        self.n_features = n_features
        self.n_left = 18
        self.n_mid = 18
        self.n_right = 18
        self.n_state = self.n_features + self.n_right + self.n_mid + self.n_left
        self.gamma = reward_decay

        # trainer parameter
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_start if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter = 0

        # saver and restorer parameter
        self.is_restore = is_restore
        self.is_save = is_save
        self.save_path = save_path
        if restore_path is not None:
            self.restore_path = restore_path
        else:
            self.restore_path = self.save_path

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, (n_features+54)*2+2))

        # initialize cost list
        self.cost_his = []

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.restore()

    def conv1d(self, x, weight, strides):
        return tf.nn.conv1d(x, weight, strides, padding='SAME', data_format="NWC")

    def _build_net(self):
        # eval net
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions])
        self.s_left = tf.placeholder(tf.float32, [None, 18, 1], name='s_left')
        self.s_mid = tf.placeholder(tf.float32, [None, 18, 1], name='s_mid')
        self.s_right = tf.placeholder(tf.float32, [None, 18, 1], name='s_right')
        self.s_feature = tf.placeholder(tf.float32, [None, self.n_features], name='s_feature')

        # self. q_eval (dim = 11)
        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES],\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            n_conv_l1 = 8
            n_conv_l2 = 8

            # build left cnn
            with tf.variable_scope('l_conv_l1'):
                w_l_conv_l1 = tf.get_variable('w_l_conv_l1', [3, 1, n_conv_l1], initializer=w_initializer, collections=c_names)
                b_l_conv_l1 = tf.get_variable('b_l_conv_l1', [1, n_conv_l1], initializer=b_initializer, collections=c_names)
                l_conv_l1 = tf.nn.relu(self.conv1d(self.s_left, w_l_conv_l1, 3)+b_l_conv_l1)
            with tf.variable_scope('l_conv_l2'):
                w_l_conv_l2 = tf.get_variable('w_l_conv_l2', [1, n_conv_l1, n_conv_l2], initializer=w_initializer, collections=c_names)
                b_l_conv_l2 = tf.get_variable('b_l_conv_l2', [1, n_conv_l2], initializer=b_initializer, collections=c_names)
                l_conv_l2 = tf.nn.relu(self.conv1d(l_conv_l1, w_l_conv_l2, 1)+b_l_conv_l2)
            l_conv_output = tf.reshape(l_conv_l2, [-1, 6 * 8])

            # build mid cnn
            with tf.variable_scope('m_conv_l1'):
                w_m_conv_l1 = tf.get_variable('w_m_conv_l1', [3, 1, n_conv_l1], initializer=w_initializer, collections=c_names)
                b_m_conv_l1 = tf.get_variable('b_m_conv_l1', [1, n_conv_l1], initializer=b_initializer, collections=c_names)
                m_conv_l1 = tf.nn.relu(self.conv1d(self.s_mid, w_m_conv_l1, 3)+b_m_conv_l1)
            with tf.variable_scope('m_conv_l2'):
                w_m_conv_l2 = tf.get_variable('w_m_conv_l2', [1, n_conv_l1, n_conv_l2], initializer=w_initializer, collections=c_names)
                b_m_conv_l2 = tf.get_variable('b_m_conv_l2', [1, n_conv_l2], initializer=b_initializer, collections=c_names)
                m_conv_l2 = tf.nn.relu(self.conv1d(m_conv_l1, w_m_conv_l2, 1)+b_m_conv_l2)
            m_conv_output = tf.reshape(m_conv_l2, [-1, 6 * 8])

            # build right cnn
            with tf.variable_scope('r_conv_l1'):
                w_r_conv_l1 = tf.get_variable('w_r_conv_l1', [3, 1, n_conv_l1], initializer=w_initializer, collections=c_names)
                b_r_conv_l1 = tf.get_variable('b_r_conv_l1', [1, n_conv_l1], initializer=b_initializer, collections=c_names)
                r_conv_l1 = tf.nn.relu(self.conv1d(self.s_right, w_r_conv_l1, 3)+b_r_conv_l1)
            with tf.variable_scope('r_conv_l2'):
                w_r_conv_l2 = tf.get_variable('w_r_conv_l2', [1, n_conv_l1, n_conv_l2], initializer=w_initializer, collections=c_names)
                b_r_conv_l2 = tf.get_variable('b_r_conv_l2', [1, n_conv_l2], initializer=b_initializer, collections=c_names)
                r_conv_l2 = tf.nn.relu(self.conv1d(r_conv_l1, w_r_conv_l2, 1)+b_r_conv_l2)
            r_conv_output = tf.reshape(r_conv_l2, [-1, 6 * 8])
            # merge high data
            fully_connected_input = tf.concat([self.s_feature, l_conv_output, m_conv_output, r_conv_output], 1)

            # build fully connected layer
            n_fully_l1 = 50
            with tf.variable_scope('high_l1'):
                w_fully_l1 = tf.get_variable('w_fully_l1', [self.n_features+6*8*3, n_fully_l1], initializer=w_initializer, collections=c_names)
                b_fully_l1 = tf.get_variable('b_fully_l1', [1, n_fully_l1], initializer=b_initializer, collections=c_names)
                fully_l1 = tf.nn.relu(tf.matmul(fully_connected_input, w_fully_l1) + b_fully_l1)
            with tf.variable_scope('fully_l2'):
                w_fully_l2 = tf.get_variable('w_fully_l2', [n_fully_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b_fully_l2 = tf.get_variable('b_fully_l2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(fully_l1, w_fully_l2) + b_fully_l2

        # loss and train
        with tf.variable_scope('loss_h'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('tran_h'):
            train_loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope='eval_net/l_conv_l1|eval_net/l_conv_l2|eval_net/m_conv_l1|eval_net/m_conv_l2|eval_net/r_conv_l1|eval_net/r_conv_l2|eval_net/fully_l1|eval_net/fully_l2')
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=train_loss_vars)

        # target net
        self.s_left_ = tf.placeholder(tf.float32, [None, 18, 1], name='s_left_')
        self.s_mid_ = tf.placeholder(tf.float32, [None, 18, 1], name='s_mid_')
        self.s_right_ = tf.placeholder(tf.float32, [None, 18, 1], name='s_right_')
        self.s_feature_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_feature_')

        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_conv_l1 = 8
            n_conv_l2 = 8

            # build high left cnn
            with tf.variable_scope('l_conv_l1'):
                w_l_conv_l1 = tf.get_variable('w_l_conv_l1', [3, 1, n_conv_l1], initializer=w_initializer,
                                                collections=c_names)
                b_l_conv_l1 = tf.get_variable('b_l_conv_l1', [1, n_conv_l1], initializer=b_initializer,
                                                collections=c_names)
                l_conv_l1_ = tf.nn.relu(self.conv1d(self.s_left_, w_l_conv_l1, 3) + b_l_conv_l1)
            with tf.variable_scope('l_conv_l2'):
                w_l_conv_l2 = tf.get_variable('w_l_conv_l2', [1, n_conv_l1, n_conv_l2], initializer=w_initializer,
                                                collections=c_names)
                b_l_conv_l2 = tf.get_variable('b_l_conv_l2', [1, n_conv_l2], initializer=b_initializer,
                                                collections=c_names)
                l_conv_l2_ = tf.nn.relu(self.conv1d(l_conv_l1_, w_l_conv_l2, 1) + b_l_conv_l2)
                print(l_conv_l2)
            l_conv_output_ = tf.reshape(l_conv_l2_, [-1, 6 * 8])

            # build high mid cnn
            with tf.variable_scope('m_conv_l1'):
                w_m_conv_l1 = tf.get_variable('w_m_conv_l1', [3, 1, n_conv_l1], initializer=w_initializer,
                                                collections=c_names)
                b_m_conv_l1 = tf.get_variable('b_m_conv_l1', [1, n_conv_l1], initializer=b_initializer,
                                                collections=c_names)
                m_conv_l1_ = tf.nn.relu(self.conv1d(self.s_mid_, w_m_conv_l1, 3) + b_m_conv_l1)
            with tf.variable_scope('h_m_conv_l2'):
                w_m_conv_l2 = tf.get_variable('w_m_conv_l2', [1, n_conv_l1, n_conv_l2], initializer=w_initializer,
                                                collections=c_names)
                b_m_conv_l2 = tf.get_variable('b_m_conv_l2', [1, n_conv_l2], initializer=b_initializer,
                                                collections=c_names)
                m_conv_l2_ = tf.nn.relu(self.conv1d(m_conv_l1_, w_m_conv_l2, 1) + b_m_conv_l2)
            m_conv_output_ = tf.reshape(m_conv_l2_, [-1, 6 * 8])

            # build high right cnn
            with tf.variable_scope('h_r_conv_l1'):
                w_r_conv_l1 = tf.get_variable('w_r_conv_l1', [3, 1, n_conv_l1], initializer=w_initializer,
                                                collections=c_names)
                b_r_conv_l1 = tf.get_variable('b_r_conv_l1', [1, n_conv_l1], initializer=b_initializer,
                                                collections=c_names)
                r_conv_l1_ = tf.nn.relu(self.conv1d(self.s_right_, w_r_conv_l1, 3) + b_r_conv_l1)
            with tf.variable_scope('h_r_conv_l2'):
                w_r_conv_l2 = tf.get_variable('w_r_conv_l2', [1, n_conv_l1, n_conv_l2], initializer=w_initializer,
                                                collections=c_names)
                b_r_conv_l2 = tf.get_variable('b_r_conv_l2', [1, n_conv_l2], initializer=b_initializer,
                                                collections=c_names)
                r_conv_l2_ = tf.nn.relu(self.conv1d(r_conv_l1_, w_r_conv_l2, 1) + b_r_conv_l2)
            r_conv_output_ = tf.reshape(r_conv_l2_, [-1, 6 * 8])

            # merge
            fully_connected_input_ = tf.concat([self.s_feature_, l_conv_output_, m_conv_output_, r_conv_output_], 1)

            # build fully connected layer
            n_fully_l1 = 50
            with tf.variable_scope('fully_l1'):
                w_fully_l1 = tf.get_variable('w_fully_l1', [self.n_features + 6 * 8 * 3, n_fully_l1],
                                            initializer=w_initializer, collections=c_names)
                b_fully_l1 = tf.get_variable('b_high_l1', [1, n_fully_l1], initializer=b_initializer, collections=c_names)
                fully_l1 = tf.nn.relu(tf.matmul(fully_connected_input_, w_fully_l1) + b_fully_l1)
            with tf.variable_scope('fully_l2'):
                w_fully_l2 = tf.get_variable('w_fully_l2', [n_fully_l1, self.n_actions], initializer=w_initializer,
                                            collections=c_names)
                b_fully_l2 = tf.get_variable('b_high_l2', [1, self.n_actions], initializer=b_initializer,
                                            collections=c_names)
                self.q_next = tf.matmul(fully_l1, w_fully_l2) + b_fully_l2

    def store_transition(self, s, a_num, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        a = a_num
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation_s_left = observation[:self.n_left]
        observation_s_mid = observation[self.n_left:self.n_left+self.n_mid]
        observation_s_right = observation[self.n_left+self.n_mid:self.n_left+self.n_mid+self.n_right]
        observation_s_state = observation[self.n_left+self.n_mid+self.n_right:]
        observation_s_state = observation_s_state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            print("greedy choose")
            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s_left: observation_s_left.reshape(-1,18,1),
                                                     self.s_mid: observation_s_mid.reshape(-1,18,1),
                                                     self.s_right: observation_s_right.reshape(-1,18,1),
                                                     self.s_feature: observation_s_state
                                                     })
            action = np.argmax(actions_value)
            print(actions_value)
            if action < 5:
                actions_low = actions_value[:, :self.n_actions_l]
                action_low = np.argmax(actions_low, axis=1)[0]
                action_high = 0
            elif action == 5:
                action_low = 0
                action_high = 1
            else:
                actions_low = actions_value[:, -self.n_actions_r:]
                action_low = np.argmax(actions_low, axis=1)[0]
                action_high = 2
        else:
            print("random choose")
            action_high = np.random.randint(0, 3)
            if action_high == 0:
                action_low = np.random.randint(0, 5)
            elif action_high == 1:
                action_low = 0
            else:
                action_low = np.random.randint(0, 5)
        return [action_high, action_low]

    def learn(self):
        # replace data and increase epsilon
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # train low net 10 times every 20 steps
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_left_: batch_memory[:, -(self.n_features+self.n_right+self.n_mid+self.n_left):-(self.n_features+self.n_right+self.n_mid)].reshape(-1, 18, 1),
                self.s_mid_: batch_memory[:, -(self.n_features+self.n_right+self.n_mid):-(self.n_features+self.n_right)].reshape(-1, 18, 1),
                self.s_right_: batch_memory[:, -(self.n_features+self.n_right):-self.n_features].reshape(-1, 18, 1),
                self.s_feature_: batch_memory[:, -self.n_features:],  # fixed params
                self.s_left: batch_memory[:, :self.n_left].reshape(-1, 18, 1),  # newest params
                self.s_mid: batch_memory[:, self.n_left:self.n_left+self.n_mid].reshape(-1, 18, 1),
                self.s_right: batch_memory[:, self.n_left+self.n_mid:self.n_left+self.n_mid+self.n_right].reshape(-1, 18, 1),
                self.s_feature: batch_memory[:, self.n_left+self.n_mid+self.n_right:self.n_left+self.n_mid+self.n_right+self.n_features]
            })
        q_target = q_eval.copy()
        eval_act_index = batch_memory[:, self.n_state].astype(int)
        reward = batch_memory[:, self.n_state + 1]
        for batch_index in range(self.batch_size):
            next_action = np.argmax(q_next[batch_index, :])
            if next_action < 5:
                q_next = np.max(q_next[batch_index, :self.n_actions_l])
            elif next_action == 5:
                q_next = q_next[batch_index, self.n_actions_l]
            else:
                q_next = np.max(q_next[batch_index, -self.n_actions_r:])
            q_target[batch_index, eval_act_index[batch_index]] = reward[batch_index] + self.gamma * q_next
        _, cost = self.sess.run([self._train_op, self.loss],
                                       feed_dict={self.s_left: batch_memory[:, :self.n_left].reshape(-1, 18, 1),  # newest params
                                                   self.s_mid: batch_memory[:, self.n_left:self.n_left+self.n_mid].reshape(-1, 18, 1),
                                                   self.s_right: batch_memory[:, self.n_left+self.n_mid:self.n_left+self.n_mid+self.n_right].reshape(-1, 18, 1),
                                                   self.s_feature: batch_memory[:, self.n_left+self.n_mid+self.n_right:self.n_left+self.n_mid+self.n_right+self.n_features],
                                                   self.q_target: q_target})
        self.cost_his.append(cost)
        # increasing learn step counter
        self.learn_step_counter += 1

    def plot_cost(self, length=None):
        import matplotlib.pyplot as plt
        if length is None:
            plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        else:
            length_his = min(len(self.cost_his), length)
            plt.plot(np.arange(length_his), self.cost_his[-length_his:])
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save(self):
        if self.is_save is True:
            self.saver.save(self.sess, self.save_path + 'model.ckpt')
            print("save successfully")
            cost = np.array(self.cost_his)
            np.save(self.save_path+"cost.npy", cost)
            # np.save(self.save_path+"memory.npy", self.memory)

    def restore(self):
        if self.is_restore is True:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("restore successfully")
            self.cost_his = np.load(self.restore_path+"cost_h.npy").tolist()
            # self.memory = np.load(self.restore_path+"memory.npy")
            # self.memory_counter = len(self.memory[:, 0])
        else:
            pass
