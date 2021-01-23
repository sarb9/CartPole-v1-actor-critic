import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pyvirtualdisplay import Display

from model.model import ActorCritic
from environment.environment import CartPoleEnvironment
from settings import SEED, NUM_HIDDEN_UNITS, MAX_STEPS_PER_EPISIDES
from render import render_episode


def init():
    # Set seed for experiment reproducibility
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()


if __name__ == "__main__":
    init()
    environment = CartPoleEnvironment()
    model = ActorCritic(environment.num_actions, NUM_HIDDEN_UNITS)
    environment.model = model
    environment.train()

    display = Display(visible=0, size=(400, 300))
    display.start()
    # Save GIF image
    images = render_episode(environment.env, model, MAX_STEPS_PER_EPISIDES)
    image_file = "cartpole-v0.gif"
    # loop=0: loop forever, duration=1: play each frame for 1ms
    images[0].save(
        image_file, save_all=True, append_images=images[1:], loop=0, duration=1
    )
