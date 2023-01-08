import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Stores and replays experiences collected from interactions 
    of the agent with the environment.
    """
    def __init__(self, capacity_buffer_ = 1000, batch_size_ = 32):
        self.replay_buffer = deque(maxlen=capacity_buffer_)
        self.capacity_buffer = capacity_buffer_
        self.batch_size = batch_size_

    def store_experience(self, state, control, cost, next_state):
        """
        Records an experience and if necessary resize the 
        length of the buffer to capacity_buffer
        """
        experience = [state, control, cost, next_state]
        self.replay_buffer.append(experience)

    def sample_batch(self):
        """
        Sample a batch of experince (size = batch_size) for training
        """
        batch = random.choices(self.replay_buffer, k=self.batch_size)
        x_batch, u_batch, cost_batch, x_next_batch= list(zip(*batch))
        
        x_batch       = np.concatenate([x_batch], axis=1).T
        u_batch       = np.asarray(u_batch)
        cost_batch    = np.asarray(cost_batch)
        x_next_batch  = np.concatenate([x_next_batch], axis=1).T

        return x_batch, x_next_batch, cost_batch

    def get_length(self):
        return len(self.replay_buffer)