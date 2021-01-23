from IPython import display as ipythondisplay
from PIL import Image
import tensorflow as tf
import numpy as np
import gym


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
    screen = env.render(mode="rgb_array")
    im = Image.fromarray(screen)

    images = [im]

    state = tf.constant(env.reset(), dtype=tf.float32)
    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_probs, _ = model(state)
        action = np.argmax(np.squeeze(action_probs))

        state, _, done, _ = env.step(action)
        state = tf.constant(state, dtype=tf.float32)

        # Render screen every 10 steps
        if i % 10 == 0:
            screen = env.render(mode="rgb_array")
            images.append(Image.fromarray(screen))

        if done:
            break

    return images
