import numpy as np
from numpy.random import randint, uniform
from pendulumn_dci import Pendulum_dci
from DQN_template import get_critic, dqn_learning, np2tf, tf2np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

def render_greedy_policy(env, Q, gamma, x0=None, maxiter=20):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    costToGo = 0.0
    gamma_i = 1
    for i in range(maxiter):
        action_values = Q.predict(x)
        best_action_index = tf.argmin(action_values)
        u = tf2np(action_values[best_action_index])
#        print("State", x, "Control", u, "Q", Q[x,u])
        x,c = env.step(u)
        costToGo += gamma_i*c
        gamma_i *= gamma
        env.render()
    print("Real cost to go of state", x0, ":", costToGo)


def compute_V_pi_from_Q(Q, vMax=5, xstep=20, nx=2):
    ''' Compute Value table and greedy policy pi from Q table. '''

    x = np.empty(shape = (nx,xstep+1))
    DQ = 2*np.pi/xstep
    DV = 2*vMax/xstep
    x[0,:] = np.arange(-np.pi,np.pi+DQ, DQ)
    x[1,:] = np.arange(-vMax, vMax+DV, DV)
    pi = np.empty(shape = (xstep+1,xstep+1))
    V = np.empty(shape = (xstep+1,xstep+1))

    for i in range(np.shape(x)[1]):
        for j in range(np.shape(x)[1]):
            action_values = Q.predict([[x[0,i]],[x[1,j]]])
            best_action_index = tf.argmin(action_values)
            pi[i,j] = tf2np(action_values[best_action_index])
            V[i,j]  = tf2np(tf.keras.backend.min(action_values))

    return V, pi, x
    
    # pi[x] = np.argmin(Q[x,:])
        # Rather than simply using argmin we do something slightly more complex
        # to ensure simmetry of the policy when multiply control inputs
        # result in the same value. In these cases we prefer the more extreme
        # actions
    # u_best = np.where(Q[:]==V)[0]
    # if(u_best[0]>env.c2du(0.0)):
    #     pi = u_best[-1]
    # elif(u_best[-1]<env.c2du(0.0)):
    #     pi = u_best[0]
    # else:
    #     pi = u_best[int(u_best.shape[0]/2)]

if __name__=='__main__':
    ### States and Control
    nx = 2 #Number of States (2 x joint)
    nu = 1 #Number of control

    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    NEPISODES                       = 100     # Number of training episodes
    NPRINT                          = 10      # print something every NPRINT episodes
    MAX_EPISODE_LENGTH              = 100     # Max episode length
    QVALUE_LEARNING_RATE            = 1e-3    # alpha coefficient of algorithm
    DISCOUNT                        = 0.99    # Discount factor 
    PLOT                            = True    # Plot stuff if True
    exploration_prob                = 1       # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.05    # exploration decay for exponential decreasing
    min_exploration_prob            = 0.001   # minimum of exploration probability
    BUFFER_CAPACITY                 = 1000    # buffer capacity
    MIN_BUFFER                      = 100     # Start sampling from buffer when have length > MIN_BUFFER
    BATCH_SIZE                      = 32      # batch size
    C_STEP                          = 4       # n° of step to update the target NN
    FLAG                            = True  # False = Load Model

    # initialize the Q network and Q_target network
    Q = get_critic(nx, nu)
    Q_target = get_critic(nx, nu)

    Q.summary()

    # Set initial weights of targets equal to those of the critic
    Q_target.set_weights(Q.get_weights())

    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
    
    print(critic_optimizer)

    ### --- Environment
    nd_u = 11                 # number of discretization steps for the joint torque u
    nd_x = 21
    env  = Pendulum_dci(1,nd_x,nd_x,nd_u) # enviroment with continuous state and discrete control input
    
    if (FLAG == True):
        Q, h_ctg = dqn_learning(env, DISCOUNT, Q, Q_target, NEPISODES, MAX_EPISODE_LENGTH, critic_optimizer, \
            exploration_prob, exploration_decreasing_decay, min_exploration_prob, \
            BUFFER_CAPACITY, MIN_BUFFER, BATCH_SIZE,C_STEP,compute_V_pi_from_Q, PLOT, NPRINT)
        
        print("\nTraining finished")
        Q.save('saved_model/my_model')
        print("\nSave NN weights to file (in HDF5)")
        Q.save_weights('saved_model/weight.h5')

    w = Q.get_weights()
    for i in range(len(w)):
        print("Shape Q weights layer", i, w[i].shape)
    for i in range(len(w)):
        print("Norm Q weights layer", i, np.linalg.norm(w[i]))

    if FLAG == False:
        Q = tf.keras.models.load_model('saved_model/my_model')
        assert(Q)
    
    if (nx == 2):
        V, pi, xgrid = compute_V_pi_from_Q(Q)
        env.plot_V_table(V, xgrid)
        env.plot_policy(pi, xgrid)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
    
    # print("Compute real Value function of greedy policy")
    # MAX_EVAL_ITERS    = 200     # Max number of iterations for policy evaluation
    # VALUE_THR         = 1e-3    # convergence threshold for policy evaluation
    # V_pi = policy_eval(env, DISCOUNT, pi, V, MAX_EVAL_ITERS, VALUE_THR, False)
    # env.plot_V_table(V_pi)
    # print("Average/min/max Value:", np.mean(V_pi), np.min(V_pi), np.max(V_pi)) 
        
    render_greedy_policy(env, Q, DISCOUNT)
    plt.figure()
    plt.plot( np.cumsum(h_ctg)/range(1,NEPISODES+1) )
    plt.title ("Average cost-to-go")

    plt.show()
