import argparse
import gym
from gym import wrappers
from gym.envs.registration import register
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import flags
from Players import dqn
from Players.dqn_utils import *

register(
    id='DeepSoccer-v0',
    entry_point='Games.deep_soccer:DeepSoccer',
)
flags.DEFINE_string('agents', 'QR',
                    'RR, QR, QQ, MR, MM, MQ, ...')
flags.DEFINE_bool('eval', True,
                    'Use random policy and a Q-challenger to evaluate the first agent')


def deepsoccer_q_model(img_in, num_actions, scope, reuse=False):
    '''Fully connected: (H*W*(2N+1)) -> 512 -> 256 -> (5+N-1)^N'''
    with tf.variable_scope(scope, reuse=reuse):
        out = layers.flatten(img_in)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(
                out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(
                out, num_outputs=256, activation_fn=tf.nn.relu)
            out = layers.fully_connected(
                out, num_outputs=num_actions, activation_fn=None)
        return out


def deepsoccer_q_learn(env, session, num_timesteps):
    # This is just a rough estimate
    num_iterations = num_timesteps

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10,
         1e-4 * lr_multiplier),
        (num_iterations / 2,
         5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (num_iterations, 0.1),
        ], outside_value=0.1
    )

    dqn.learn(
        env,
        q_func=deepsoccer_q_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        num_timesteps=int(num_timesteps),
        replay_buffer_size=1000000,
        batch_size=8,
        gamma=0.9,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=1,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_env(env_id, seed):
    # env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/deepsoccer' + flags.FLAGS.agents + '/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    # env = wrap_deepmind(env)

    return env


def main():
    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env('DeepSoccer-v0', seed)
    session = get_session()
    # TODO set proper timesteps
    deepsoccer_q_learn(env, session, num_timesteps=2e6)


if __name__ == "__main__":
    main()
