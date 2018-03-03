import sys
import gym.spaces
import tensorflow as tf
from tensorflow.python.platform import flags
from collections import namedtuple
from .dqn_utils import *
from scipy.optimize import linprog

FLAGS = flags.FLAGS
OptimizerSpec = namedtuple(
    "OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          num_timesteps=1000000,
          replay_buffer_size=1000000,
          batch_size=64,
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
    # placeholder for minimax value for current state
    v_t_ph = tf.placeholder(tf.float32, [None])
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

        def choose_action(self, obs):
            raise NotImplementedError

        def train_step(self, t):
            raise NotImplementedError

    class RandomAgent(Agent):
        def __init__(self, num_actions, opponent):
            super(RandomAgent, self).__init__(num_actions, opponent)

        def construct_model(self):
            pass

        def choose_action(self, obs):
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
            self.q = q_func(obs_t_float, self.num_actions,
                            scope=self.scope+"q", reuse=False)
            q_func_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"q")
            # Target Q
            target_q = q_func(obs_tp1_float, self.num_actions,
                              scope=self.scope+"target_q", reuse=False)
            target_q_func_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"target_q")
            # Choose the corresponding q value of the action
            if self.opponent:
                q_act = tf.reduce_sum(
                    self.q * tf.one_hot(act_t_ph // self.num_actions, self.num_actions), axis=1)
            else:
                q_act = tf.reduce_sum(
                    self.q * tf.one_hot(act_t_ph % self.num_actions, self.num_actions), axis=1)

            q_look_ahead = rew_t_ph + \
                (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, axis=1)

            # Bellman error
            total_error = tf.nn.l2_loss(q_act - q_look_ahead) / batch_size

            # construct optimization op (with gradient clipping)
            optimizer = optimizer_spec.constructor(
                learning_rate=learning_rate, **optimizer_spec.kwargs)
            self.train_fn = minimize_and_clip(
                optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_fn = []
            for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_fn.append(var_target.assign(var))
            self.update_target_fn = tf.group(*update_target_fn)

        def choose_action(self, obs):
            obs = np.expand_dims(obs, axis=0)
            q_values = session.run(self.q, feed_dict={obs_t_ph: obs})
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
            self.q = q_func(obs_t_float, self.num_actions *
                            self.num_actions, scope=self.scope+"q", reuse=False)
            q_func_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"q")
            # Target Q
            target_q = q_func(obs_tp1_float, self.num_actions *
                              self.num_actions, scope=self.scope+"target_q", reuse=False)
            target_q_func_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"target_q")
            # Choose the corresponding minimax q value of the action
            q_act = tf.reduce_sum(
                self.q * tf.one_hot(act_t_ph, self.num_actions * self.num_actions), axis=1)
            # Reshape to look like a matrix game
            self.q = tf.reshape(
                self.q, [-1, self.num_actions, self.num_actions])
            self.target_q = tf.reshape(
                target_q, [-1, self.num_actions, self.num_actions])
            if self.opponent:
                self.q = tf.transpose(self.q, perm=[0, 2, 1])
                self.target_q = tf.transpose(self.target_q, perm=[0, 2, 1])

            # Use linear programming
            q_look_ahead = rew_t_ph + (1 - done_mask_ph) * gamma * v_t_ph

            # Bellman error
            total_error = tf.nn.l2_loss(q_act - q_look_ahead) / batch_size

            # construct optimization op (with gradient clipping)
            optimizer = optimizer_spec.constructor(
                learning_rate=learning_rate, **optimizer_spec.kwargs)
            self.train_fn = minimize_and_clip(
                optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_fn = []
            for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_fn.append(var_target.assign(var))
            self.update_target_fn = tf.group(*update_target_fn)

        def choose_action(self, obs):
            q_values = session.run(self.q, feed_dict={obs_t_ph: obs})
            _, pi_t = np.squeeze(self._choose_policy(
                q_values, need_policy=True))
            act = np.random.choice(a=range(self.num_actions), p=pi_t)
            if self.opponent:
                return self.num_actions * act
            else:
                return act

        def _choose_policy(self, q_values, need_policy=False):
            if need_policy:
                pi_t_batch = np.zeros(
                    (q_values.shape[0], num_actions_per_agent))
            v_t_batch = np.zeros((q_values.shape[0]))
            for i in range(q_values.shape[0]):
                c = np.zeros(num_actions_per_agent + 1)
                c[0] = -1
                A_ub = np.ones(
                    (num_actions_per_agent, num_actions_per_agent + 1))
                A_ub[:, 1:] = -q_values[i]
                b_ub = np.zeros(num_actions_per_agent)
                A_eq = np.ones((1, num_actions_per_agent + 1))
                A_eq[0, 0] = 0
                b_eq = [1]
                bounds = ((None, None), ) + ((0, 1), ) * num_actions_per_agent
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
                if res.success:
                    if need_policy:
                        pi_t_batch[i] = res.x[1:]
                    v_t_batch[i] = res.x[0]
                else:
                    # use max min
                    # TODO inspect how many lp failed
                    if need_policy:
                        pi_t_batch[i][np.argmax(
                            np.min(q_values[i], axis=0))] = 1
                    v_t_batch[i] = np.max(np.min(q_values[i], axis=0))
            if need_policy:
                return v_t_batch, pi_t_batch
            else:
                return v_t_batch

        def train_step(self, t):
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = \
                replay_buffer.sample(batch_size)
            if self.opponent:
                rew_t_batch = -rew_t_batch
            q_values = session.run(self.target_q, feed_dict={
                                   obs_tp1_ph: obs_tp1_batch})
            v_t_batch = self._choose_policy(q_values)

            session.run(self.train_fn, {
                obs_t_ph: obs_t_batch,
                act_t_ph: act_t_batch,
                rew_t_ph: rew_t_batch,
                v_t_ph: v_t_batch,
                done_mask_ph: done_mask,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            })

            self.num_param_updates += 1
            if self.num_param_updates % target_update_freq == 0:
                session.run(self.update_target_fn)

    HEIGHT = 7
    WIDTH = 4
    PLAYERS = 1

    class TabularQAgent(Agent):
        def __init__(self, num_actions, opponent, scope):
            super(TabularQAgent, self).__init__(num_actions, opponent)
            self.num_states = (HEIGHT * WIDTH) ** (2 * PLAYERS + 1)
            self.alpha = 1
            self.decay = 10 ** (-2 / num_timesteps)

        def construct_model(self):
            self.Q = np.random.random((self.num_states, self.num_actions))

        def choose_action(self, obs):
            obs = np.expand_dims(obs, axis=0)
            state = self._state_idx(obs)
            if self.opponent:
                return self.num_actions * np.argmax(self.Q[state])
            else:
                return np.argmax(self.Q[state])

        def train_step(self, t):
            obs, act, rew, obs_next, done = replay_buffer.sample(batch_size)
            obs = np.squeeze(obs)
            act = np.squeeze(act)
            rew = np.squeeze(rew)
            obs_next = np.squeeze(obs_next)
            done = np.squeeze(done)
            if self.opponent:
                rew = -rew
                act //= self.num_actions
            else:
                act %= self.num_actions

            state = self._state_idx(obs)
            next_state = self._state_idx(obs_next)

            self.Q[state, act] += self.alpha * \
                                  (rew +
                                   gamma * (1 - done) * np.max(self.Q[next_state]) -
                                   self.Q[state, act])
            self.alpha *= self.decay

        @staticmethod
        def _state_idx(state):
            assert state.ndim == 3
            idx = 0
            for i in range(2 * PLAYERS + 1):
                idx *= HEIGHT * WIDTH
                idx += np.argmax(state[:, :, i])

            return idx

    class TabularMinimaxQAgent(Agent):
        def __init__(self, num_actions, opponent, scope):
            super(TabularMinimaxQAgent, self).__init__(num_actions, opponent)
            self.num_states = (HEIGHT * WIDTH) ** (2 * PLAYERS + 1)
            self.alpha = 1
            self.decay = 10 ** (-2 / num_timesteps)

        def construct_model(self):
            self.Q = np.random.random((self.num_states, self.num_actions ** 2))
            self.V = np.random.random((self.num_states))
            self.pi = np.ones((self.num_states, self.num_actions)) / self.num_actions

        def choose_action(self, obs):
            state = self._state_idx(obs)
            if self.opponent:
                return self.num_actions * np.random.choice(self.num_actions, p=self.pi[state])
            else:
                return np.random.choice(self.num_actions, p=self.pi[state])

        def _choose_policy(self, q_values, need_policy=False):
            if need_policy:
                pi_t = np.zeros((num_actions_per_agent))
            c = np.zeros(num_actions_per_agent + 1)
            c[0] = -1
            A_ub = np.ones(
                (num_actions_per_agent, num_actions_per_agent + 1))
            A_ub[:, 1:] = -q_values
            b_ub = np.zeros(num_actions_per_agent)
            A_eq = np.ones((1, num_actions_per_agent + 1))
            A_eq[0, 0] = 0
            b_eq = [1]
            bounds = ((None, None), ) + ((0, 1), ) * num_actions_per_agent
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
            if res.success:
                if need_policy:
                    pi_t = res.x[1:]
                v_t = res.x[0]
            else:
                # use max min
                # TODO inspect how many lp failed
                if need_policy:
                    pi_t[np.argmax(np.min(q_values, axis=0))] = 1
                v_t = np.max(np.min(q_values, axis=0))
            if need_policy:
                return v_t, pi_t
            else:
                return v_t

        def train_step(self, t):
            obs, act, rew, obs_next, done = replay_buffer.sample(batch_size)
            obs = np.squeeze(obs)
            act = np.squeeze(act)
            rew = np.squeeze(rew)
            obs_next = np.squeeze(obs_next)
            done = np.squeeze(done)

            if self.opponent:
                rew = -rew

            state = self._state_idx(obs)
            next_state = self._state_idx(obs_next)
            self.Q[state, act] += self.alpha * \
                (rew + gamma * (1 - done) * self.V[next_state] - self.Q[state, act])

            # Update policy
            q_values = self.Q[state].reshape(
                (self.num_actions, self.num_actions))
            if self.opponent:
                q_values = np.transpose(q_values)
            self.V[state], self.pi[state] = self._choose_policy(q_values, need_policy=True)

            self.alpha *= self.decay

        @staticmethod
        def _state_idx(state):
            assert state.ndim == 3
            idx = 0
            for i in range(2 * PLAYERS + 1):
                idx *= HEIGHT * WIDTH
                idx += np.argmax(state[:, :, i])

            return idx

    agents = []
    opponents = [False, True]
    scopes = ["", "opp_"]
    # TODO: use factory method to create agent instances
    for i, (opponent, scope) in enumerate(zip(opponents, scopes)):
        if FLAGS.agents[i] == 'T':
            agents.append(TabularQAgent(
                num_actions=num_actions_per_agent, opponent=opponent, scope=scope))
        elif FLAGS.agents[i] == 'S':
            agents.append(TabularMinimaxQAgent(
                num_actions=num_actions_per_agent, opponent=opponent, scope=scope))
        elif FLAGS.agents[i] == 'M':
            agents.append(MinimaxQAgent(
                num_actions=num_actions_per_agent, opponent=opponent, scope=scope))
        elif FLAGS.agents[i] == 'Q':
            agents.append(QAgent(num_actions=num_actions_per_agent,
                                 opponent=opponent, scope=scope))
        elif FLAGS.agents[i] == 'R':
            agents.append(RandomAgent(
                num_actions=num_actions_per_agent, opponent=opponent))
        else:
            raise NotImplementedError
        agents[i].construct_model()
    if FLAGS.eval:
        random_challenger = RandomAgent(num_actions_per_agent, opponent=True)
    if FLAGS.challenge:
        q_challenger = TabularQAgent(
            num_actions_per_agent, opponent=True, scope="chal")
        q_challenger.construct_model()

    initialize_interdependent_variables(session, tf.global_variables())

    ###############
    # RUN ENV     #
    ###############

    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    writer = tf.summary.FileWriter('logs/' + FLAGS.name)
    summ_op = tf.summary.merge_all()

    def log_progress(t, eps, best_mean_episode_reward, offset=0):
        LOG_EVERY_N_STEPS = 10000
        PLOT_EVERY_N_STEPS = 1000
        episode_rewards = get_wrapper_by_name(
            env, "Monitor").get_episode_rewards()
        mean_episode_reward = -float('nan')
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(
                best_mean_episode_reward, mean_episode_reward)
        if t % PLOT_EVERY_N_STEPS == 0:
            s = session.run(summ_op, feed_dict={
                mean_rew_ph: mean_episode_reward})
            writer.add_summary(s, t + offset)
            writer.flush()
        if t % LOG_EVERY_N_STEPS == 0 and t >= learning_starts:
            print("Timestep %d" % (t + offset,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % eps)
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
        return best_mean_episode_reward

    for t in range(num_timesteps):
        # 1. Step the env and store the transition
        ret=replay_buffer.store_frame(last_obs)

        eps = exploration.value(t)
        action = 0
        for agent in agents:
            if np.random.random() >= eps and t >= learning_starts:
                recent_obs = replay_buffer.encode_recent_observation()
                action += agent.choose_action(obs=recent_obs)
            else:
                action += agent.random_action()

        last_obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(ret, action, reward, done)
        if done:
            last_obs = env.reset()

        # 2. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t >= learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            for agent in agents:
                agent.train_step(t)

        # 3
        best_mean_episode_reward = log_progress(t, eps, best_mean_episode_reward)

    def env_reset():
        while True:
            action = env.action_space.sample()
            last_obs, reward, done, info = env.step(action)
            if done:
                return env.reset()

    def evaluate(agent, challenger, learn, name):
        if learn:
            last_obs = env_reset()
            best_mean_episode_reward = -float('inf')
            # Give challenger more timesteps and exploration
            # Since the left side has fixed policy
            for t in range(2 * num_timesteps):
                ret = replay_buffer.store_frame(last_obs)
                eps = exploration.value(t // 2)

                recent_obs = replay_buffer.encode_recent_observation()
                action = agent.choose_action(obs=recent_obs)
                if np.random.random() >= eps and t >= learning_starts:
                    action += challenger.choose_action(obs=recent_obs)
                else:
                    action += challenger.random_action()

                last_obs, reward, done, info = env.step(action)
                if done:
                    last_obs = env.reset()
                replay_buffer.store_effect(ret, action, reward, done)

                if (t >= learning_starts and
                        t % learning_freq == 0 and
                        replay_buffer.can_sample(batch_size)):
                    challenger.train_step(t)
                best_mean_episode_reward = log_progress(
                    t, eps, best_mean_episode_reward, offset=num_timesteps)

        last_obs = env_reset()
        n_episodes = 0
        EVAL_EPISODES = 5000
        while n_episodes < EVAL_EPISODES:
            action = agent.choose_action(
                obs=last_obs) + challenger.choose_action(obs=last_obs)
            last_obs, reward, done, info = env.step(action)
            if done:
                last_obs = env.reset()
                n_episodes += 1
        episode_rewards = get_wrapper_by_name(
            env, "Monitor").get_episode_rewards()
        won = np.sum(np.array(episode_rewards[-EVAL_EPISODES:]) > 0)
        lost = np.sum(np.array(episode_rewards[-EVAL_EPISODES:]) < 0)
        draw = np.sum(np.array(episode_rewards[-EVAL_EPISODES:]) == 0)
        print(" vs %s: %d won, %d lost, %d draw. Win rate %g" %
              (name, won, lost, draw, won / EVAL_EPISODES))

    if FLAGS.eval:
        # Random challenger
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        evaluate(agents[0], random_challenger, learn=False, name="random")

    if FLAGS.challenge:
        # Q challenger
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        evaluate(agents[0], q_challenger, learn=True, name="challenger")
