import numpy as np
import random

class ReplayBuffer:
    """
    Stores and replays experiences collected from interactions 
    of the agent with the environment.
    """
    def __init__(self, capacity_buffer_ = 1000, batch_size_ = 32 ):
        self.replay_buffer = []
        self.capacity_buffer = capacity_buffer_
        self.batch_size = batch_size_

    def store_experience(self, state, control, cost, next_state):
        """
        Records an experience and if necessary resize the 
        length of the buffer to capacity_buffer
        """
        experience = (state, control, cost, next_state)
        self.replay_buffer.append(experience)

        del self.replay_buffer[:-self.capacity_buffer]

    def sample_batch(self):
        """
        Sample a batch of experince (size = batch_size) for training
        """
        batch = random.choices(self.replay_buffer, k=self.batch_size)
        x_batch, _, cost_batch, x_next_batch = list(zip(*batch))
        return x_batch, cost_batch, x_next_batch