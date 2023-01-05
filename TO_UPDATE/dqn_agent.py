import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
from numpy.random import randint, uniform
import matplotlib as plt
from pendulumn_dci import Pendulum_dci

np_config.enable_numpy_behavior()

class DQNagent:
    """
        DQN agent
    """

    def __init__(self, nx_, nu_, discount = 0.99, q_value_learning_rate = 1e-3):
        """
            nx: number of states
            nu: number of control input
            discount: discount factor (for the cost)
            q_value_learning_rate: alpha coefficient of Q learning algorithm
            Q: Q network
            Q_target: target network
            critic_optimizer: optimizer specifying the learning rates
        """
        self.nx = nx_
        self.nu = nu_
        self.DISCOUNT = discount
        self.QVALUE_LEARNING_RATE = q_value_learning_rate
        self.Q = self.get_critic()
        self.Q_target = self.get_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(self.QVALUE_LEARNING_RATE)

        self.Q_target.set_weights(self.Q.get_weights())


    def get_critic(self):
        """ 
            Create the neural network to represent the Q function
        """
        inputs = layers.Input(shape=(self.nx+self.nu,1))              # input
        state_out1 = layers.Dense(16, activation="relu")(inputs)      # hidden layer 1
        state_out2 = layers.Dense(32, activation="relu")(state_out1)  # hidden layer 2
        state_out3 = layers.Dense(64, activation="relu")(state_out2)  # hidden layer 3
        state_out4 = layers.Dense(64, activation="relu")(state_out3)  # hidden layer 4
        outputs = layers.Dense(1)(state_out4)                         # output

        model = tf.keras.Model(inputs, outputs)                       # create the NN

        return model

    def get_action(self, exploration_prob, env, x, EGREEDY):
        """
            epsilon-greedy policy
        """
        # with probability exploration_prob take a random control input
        if(uniform() < exploration_prob and EGREEDY == True):
            u = randint(0, env.nu)
        # otherwise take a greedy control
        else:
            action_values = self.Q.predict(x)
            print(action_values)
            best_action_index = tf.argmin(action_values)
            print(best_action_index)
            print(action_values[best_action_index])
            u = self.tf2np(action_values[best_action_index])
        return u

    def update(self, xu_batch, cost_batch, xu_next_batch):
        """ 
            Update the weights of the Q network using the specified batch of data 
            - all inputs are tf tensors
        """
        with tf.GradientTape() as tape:         
            # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
            # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
            # Tensors can be manually watched by invoking the watch method on this context manager.
            target_values = self.Q_target(xu_next_batch, training=True)   
            # Compute 1-step targets for the critic loss
            y = cost_batch + self.DISCOUNT*target_values                            
            # Compute batch of Values associated to the sampled batch of states
            Q_value = self.Q(xu_batch, training=True)                         
            # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
        # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
        Q_grad = tape.gradient(Q_loss, self.Q.trainable_variables)          
        # Update the critic backpropagating the gradients
        self.critic_optimizer.apply_gradients(zip(Q_grad, self.Q.trainable_variables))

    def update_Q_target(self):
        """
            Update the current Q_target with the weight of Q
        """
        self.Q_target.set_weights(self.Q.get_weights())

    def np2tf(self, y):
        """
            convert from numpy to tensorflow
        """
        out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
        return out
        
    def tf2np(self, y):
        """ 
            convert from tensorflow to numpy 
        """
        return tf.squeeze(y).numpy()

if __name__=="__main__":

    agent = DQNagent(2,1)
    env = Pendulum_dci(1)

    env.reset()

    print("x")
    print(env.x)
    u = agent.get_action(1, env, env.x, False)
    
    x_next, cost = env.step(u)

    print("x_next")
    print(x_next)
    u_next = agent.get_action(1, env, x_next, False)

    agent.Q.summary()
    agent.Q_target.summary()