import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import flags
from collections import namedtuple
from .dqn_utils import *

FLAGS = flags.FLAGS
OptimizerSpec = namedtuple(
    "OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete
    assert learning_starts % learning_freq == 0

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n
    num_actions_per_agent = int(np.sqrt(num_actions))

    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32, [None])
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph = tf.placeholder(tf.float32, [None])
    # placeholder for lr
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # For tensorboard plot
    mean_rew_ph = tf.placeholder(tf.float32)
    tf.summary.scalar("mean reward 100 episodes", mean_rew_ph)

    class Agent(object):
        def __init__(self, num_actions, opponent):
             self.num_actions = num_actions
             self.opponent = opponent

        def random_action(self):
            if self.opponent:
                return self.num_actions * np.random.choice(self.num_actions)
            else:
                return np.random.choice(self.num_actions)

        def construct_model(self):
            raise NotImplementedError

        def choose_action(self):
            raise NotImplementedError

        def train_step(self):
            raise NotImplementedError

    class RandomAgent(Agent):
        def __init__(self, num_actions, opponent):
            super(RandomAgent, self).__init__(num_actions, opponent)

        def construct_model(self):
            pass

        def choose_action(self, recent_obs):
            return self.random_action()

        def train_step(self, t):
            pass

    class QAgent(Agent):
        def __init__(self, num_actions, opponent, scope):
            super(QAgent, self).__init__(num_actions, opponent)
            self.scope = scope
            self.num_param_updates = 0

        def construct_model(self):
            # Q
            self.q = q_func(obs_t_float, self.num_actions, scope=self.scope+"q", reuse=False)
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"q")
            # Target Q
            target_q = q_func(obs_tp1_float, self.num_actions,scope=self.scope+"target_q", reuse=False)
            target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"target_q")
            # Choose the corresponding q value of the action
            if self.opponent:
                q_act = tf.reduce_sum(self.q * tf.one_hot(act_t_ph // self.num_actions, self.num_actions), axis=1)
            else:
                q_act = tf.reduce_sum(self.q * tf.one_hot(act_t_ph % self.num_actions, self.num_actions), axis=1)

            q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, axis=1)

            # Bellman error
            total_error = tf.nn.l2_loss(q_act - q_look_ahead) / batch_size

            # construct optimization op (with gradient clipping)
            optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
            self.train_fn = minimize_and_clip(optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_fn = []
            for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_fn.append(var_target.assign(var))
            self.update_target_fn = tf.group(*update_target_fn)

        def choose_action(self, recent_obs):
            q_values = session.run(self.q, feed_dict={obs_t_ph: recent_obs})
            if self.opponent:
                return self.num_actions * np.argmax(np.squeeze(q_values))
            else:
                return np.argmax(np.squeeze(q_values))

        def train_step(self, t):
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                replay_buffer.sample(batch_size)
            if self.opponent:
                rew_t_batch = -rew_t_batch

            session.run(self.train_fn, {
                obs_t_ph: obs_t_batch,
                act_t_ph: act_t_batch,
                rew_t_ph: rew_t_batch,
                obs_tp1_ph: obs_tp1_batch,
                done_mask_ph: done_mask,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            })

            self.num_param_updates += 1
            if self.num_param_updates % target_update_freq == 0:
                session.run(self.update_target_fn)

    class MinimaxQAgent(QAgent):
        def __init__(self, num_actions, opponent, scope):
            super(MinimaxQAgent, self).__init__(num_actions, opponent, scope)

        def construct_model(self):
            # Q
            self.q = q_func(obs_t_float, self.num_actions * self.num_actions, scope=self.scope+"q", reuse=False)
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"q")
            # Target Q
            target_q = q_func(obs_tp1_float, self.num_actions * self.num_actions, scope=self.scope+"target_q", reuse=False)
            target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"target_q")
            # Choose the corresponding minimax q value of the action
            q_act = tf.reduce_sum(self.q * tf.one_hot(act_t_ph, self.num_actions * self.num_actions), axis=1)
            # Reshape to look like a matrix game
            self.q = tf.reshape(self.q, [-1, self.num_actions, self.num_actions])
            target_q = tf.reshape(target_q, [-1, self.num_actions, self.num_actions])
            # Use min max as minimax
            # TODO use linear programming
            if self.opponent:
                q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(tf.reduce_min(target_q, axis=2), axis=1)
            else:
                q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(tf.reduce_min(target_q, axis=1), axis=1)

            # Bellman error
            total_error = tf.nn.l2_loss(q_act - q_look_ahead) / batch_size

            # construct optimization op (with gradient clipping)
            optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
            self.train_fn = minimize_and_clip(optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_fn = []
            for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_fn.append(var_target.assign(var))
            self.update_target_fn = tf.group(*update_target_fn)

        def choose_action(self, recent_obs):
            q_values = session.run(self.q, feed_dict={obs_t_ph: recent_obs})
            if self.opponent:
                return self.num_actions * np.argmax(np.min(np.squeeze(q_values), axis=1))
            else:
                return np.argmax(np.min(np.squeeze(q_values), axis=0))

    agents = []
    opponents = [False, True]
    scopes = ["", "opp_"]
    for i, (opponent, scope) in enumerate(zip(opponents, scopes)):
        if FLAGS.agents[i] == 'M':
            agents.append(MinimaxQAgent(num_actions=num_actions_per_agent, opponent=opponent, scope=scope))
        elif FLAGS.agents[i] == 'Q':
            agents.append(QAgent(num_actions=num_actions_per_agent, opponent=opponent, scope=scope))
        elif FLAGS.agents[i] == 'R':
            agents.append(RandomAgent(num_actions=num_actions_per_agent, opponent=opponent))
        else:
            raise NotImplementedError
        agents[i].construct_model()

    initialize_interdependent_variables(session, tf.global_variables())

    ###############
    # RUN ENV     #
    ###############

    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    writer = tf.summary.FileWriter('logs/' + FLAGS.agents)
    summ_op = tf.summary.merge_all()

    for t in itertools.count():
        # 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # 2. Step the env and store the transition
        ret = replay_buffer.store_frame(last_obs)
        eps = exploration.value(t)

        action = 0
        for agent in agents:
            if np.random.random() >= eps and t >= learning_starts:
                recent_obs = np.expand_dims(replay_buffer.encode_recent_observation(), axis=0)
                action += agent.choose_action(recent_obs=recent_obs)
            else:
                action += agent.random_action()

        last_obs, reward, done, info = env.step(action)
        if done:
            last_obs = env.reset()
        replay_buffer.store_effect(ret, action, reward, done)

        # 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t >= learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            for agent in agents:
                agent.train_step(t)

        # 4. Log progress
        LOG_EVERY_N_STEPS = 10000
        PLOT_EVERY_N_STEPS = 1000
        episode_rewards = get_wrapper_by_name(
            env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(
                best_mean_episode_reward, mean_episode_reward)
        if t % PLOT_EVERY_N_STEPS == 0:
            s = session.run(summ_op, feed_dict={mean_rew_ph: mean_episode_reward})
            writer.add_summary(s, t)
            writer.flush()
        if t % LOG_EVERY_N_STEPS == 0 and t >= learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
