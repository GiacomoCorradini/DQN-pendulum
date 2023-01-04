import numpy as np
import random

class ReplayBuffer:
    """
    Stores and replays experiences collected from interactions 
    of the agent with the environment.
    """
    def __init__(self, capacity_buffer_ = 1000, batch_size_ = 32):
        self.replay_buffer = []
        self.capacity_buffer = capacity_buffer_
        self.batch_size = batch_size_

    def store_experience(self, state, control, cost, next_state, control_next):
        """
        Records an experience and if necessary resize the 
        length of the buffer to capacity_buffer
        """
        experience = [state, control, cost, next_state, control_next]
        self.replay_buffer.append(experience)

        del self.replay_buffer[:-self.capacity_buffer]

    def sample_batch(self):
        """
        Sample a batch of experince (size = batch_size) for training
        """
        batch = random.choices(self.replay_buffer, k=self.batch_size)
        x_batch, u_batch, cost_batch, x_next_batch, u_next_batch = list(zip(*batch))

        x_batch = np.concatenate(x_batch, axis=1)
        u_batch = np.asarray(u_batch)
        cost_batch = np.asarray(cost_batch)
        x_next_batch = np.concatenate(x_next_batch, axis=1)
        u_next_batch = np.asarray(u_next_batch)

        return x_batch, u_batch, cost_batch, x_next_batch, u_next_batch

    def get_length(self):
        return len(buffer.replay_buffer)

if __name__=="__main__":
    buffer = ReplayBuffer()
    buffer.store_experience([1,1],2,3,4,5)
    # buffer.store_experience([11,],22,33,44,55)
    # buffer.store_experience([111,],222,333,444,555)
    # buffer.store_experience([1111,],2222,3333,4444,5555)
    # buffer.store_experience([11111,],22222,33333,44444,55555)
    print(buffer.get_length())
    buffer.sample_batch()