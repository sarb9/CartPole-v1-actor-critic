from typing import Any, List, Sequence, Tuple

import tensorflow as tf
import numpy as np
import gym
import tqdm

from settings import (
    SEED,
    EPS,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISIDES,
    REWARD_TRESHOLD,
    RUNNING_REWARD,
    GAMMA,
)


class CartPoleEnvironment:

    seed = 42

    def __init__(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.env.seed(SEED)
        self.num_actions = self.env.action_space.n  # 2
        self.model = None

        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return (
            state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32),
        )

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(
            self.env_step, [action], [tf.float32, tf.int32, tf.int32]
        )

    def run_episode(
        self,
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int,
    ) -> List[tf.Tensor]:
        """Runs a single episode to collect training data."""
        if not self.model:
            print("No model set for running episodes.")

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)

            action_logits_t, value = model(state)

            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            values = values.write(t, tf.squeeze(value))

            action_probs = action_probs.write(t, action_probs_t[0, action])

            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(
        self,
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True,
    ) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = (returns - tf.math.reduce_mean(returns)) / (
                tf.math.reduce_std(returns) + EPS
            )

        return returns

    def compute_loss(
        self,
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor,
    ) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(
        self,
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int,
    ) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(
                initial_state, model, max_steps_per_episode
            )

            # Calculate expected returns
            returns = self.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def train(self):
        with tqdm.trange(MAX_EPISODES) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                episode_reward = int(
                    self.train_step(
                        initial_state,
                        self.model,
                        self.optimizer,
                        GAMMA,
                        MAX_STEPS_PER_EPISIDES,
                    )
                )

                running_reward = episode_reward * 0.01 + RUNNING_REWARD * 0.99

                t.set_description(f"Episode {i}")
                t.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward
                )

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > REWARD_TRESHOLD:
                    break
