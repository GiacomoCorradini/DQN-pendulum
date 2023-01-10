import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
import time
from numpy.random import rand
import tensorflow as tf

from pendulum_dci import Pendulum_dci
from dqn_agent import DQNagent
from replay_buffer import ReplayBuffer
from plot_dqn import  plot_traj

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def render_greedy_policy(env, agent, exploration_prob, x0=None, maxiter=100):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    costToGo = 0.0
    gamma_i  = 1
    X_sim    = np.zeros([maxiter,env.pendulum.nx])   # store x
    U_sim    = np.zeros(maxiter)                     # store u
    Cost_sim = np.zeros(maxiter)                     # store u
    for i in range(maxiter):
        u = agent.get_action(0, env, x, True)
        if(env.njoint == 2):
            x,c      = env.step([u,env.c2du(0)])
        else:    x,c      = env.step([u])
        costToGo += gamma_i*c
        gamma_i  *= agent.DISCOUNT
        env.render()
        X_sim[i,:]  = np.concatenate(np.array([x]).T)
        U_sim[i]    = env.d2cu(u)
        Cost_sim[i] = c
    print("Real cost to go of state", x0, ":", costToGo)
    return X_sim, U_sim, Cost_sim

def compute_V_pi_from_Q(d2cu, agent, env, xstep=20):
    ''' Compute Value table and greedy policy pi from Q table. '''

    vMax   = env.vMax  
    nx     = env.pendulum.nx
    x      = np.empty(shape = (nx,xstep+1))
    DQ     = 2*np.pi/xstep
    DV     = 2*vMax/xstep
    x[0,:] = np.arange(-np.pi,np.pi+DQ, DQ)
    x[1,:] = np.arange(-vMax,vMax+DV, DV)

    pi     = np.empty(shape = (xstep+1,xstep+1))
    V      = np.empty(shape = (xstep+1,xstep+1))

    for i in range(np.shape(x)[1]):
        for j in range(np.shape(x)[1]):
            xu = np.reshape([x[0,i]*np.ones(agent.ndu),x[1,j]*np.ones(agent.ndu),np.arange(agent.ndu)],(nx+1,agent.ndu))
            V[i,j]   = np.min(agent.Q(xu.T))
            pi[i,j]  = d2cu(np.argmin(agent.Q(xu.T)))
    return V, pi, x


def dqn_learning(buffer, agent, env,\
                 gamma, nEpisodes, maxEpisodeLength, min_buffer, c_step,\
                 exploration_prob, exploration_decreasing_decay, min_exploration_prob, \
                 plot_traj, plot=False, nprint=1000):
    ''' 
        DQN learning algorithm:
        buffer: replay buffer
        agent: dqn agent
        env: environment 
        gamma: discount factor
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # Keep track of the cost-to-go history, trajectroy and control input (for plot)
    h_ctg    = []
    X_sim    = np.zeros([maxEpisodeLength,env.pendulum.nx])
    U_sim    = np.zeros(maxEpisodeLength)
    Cost_sim = np.zeros(maxEpisodeLength)
    
    # for every episode
    for i in range(nEpisodes):
        # reset the state
        env.reset()

        # initialize to zero the cost-to-go at the beginiing of each episode 
        J = 0
        gamma_to_the_i = 1

        # time for each episode
        start = time.time()
        
        # simulate the system for maxEpisodeLength steps
        for k in range(maxEpisodeLength):
            # state of the enviroment
            x = env.x

            # epsilon-greedy action selection
            u = agent.get_action(exploration_prob, env, x, True)

            # observe cost and next state (step = calculate dynamics)
            if (env.njoint == 2):
                x_next, cost = env.step([u,env.c2du(0.0)])
            else: x_next, cost = env.step([u])
        
            # store the experience (s,a,r,s',a') in the replay_buffer
            buffer.store_experience(x, u, cost, x_next)

            if buffer.get_length() > min_buffer:
                # Randomly sample minibatch (size of batch_size) of experience from replay_buffer
                # collect together state and control
                xu_batch, xu_next_batch, cost_batch = buffer.sample_batch(env, exploration_prob, agent)

                # convert numpy to tensorflow
                xu_batch      = agent.np2tf(xu_batch)
                cost_batch    = agent.np2tf(cost_batch)
                xu_next_batch = agent.np2tf(xu_next_batch)

                # optimizer with SGD
                agent.update(xu_batch, cost_batch, xu_next_batch)
                
                # Periodically update target network (period = c_step)
                if k % c_step == 0:
                    agent.update_Q_target()   
        
            # keep track of the cost to go
            J += gamma_to_the_i * cost
            gamma_to_the_i *= gamma
    
        h_ctg.append(J)

        # update the exploration probability with an exponential decay: 
        exploration_prob = max(np.exp(-exploration_decreasing_decay*i), min_exploration_prob)
        elapsed_time = round((time.time() - start),3)

        print("Episode", i, "completed in", elapsed_time, "s - eps =", round(100*exploration_prob,2), "- cost-to-go (J) =", round(J,2))
        
        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(i%nprint==0 and i>=nprint):
            X_sim, U_sim, Cost_sim = render_greedy_policy(env, agent, 0, None, maxEpisodeLength)
            if(plot):
                # if(env.njoint == 1):
                #     V, pi, xgrid = compute_V_pi_from_Q(env.d2cu,agent)
                #     env.plot_V_table(V, xgrid)
                #     env.plot_policy(pi, xgrid)
                time_vec = np.linspace(0.0,maxEpisodeLength*env.pendulum.DT,maxEpisodeLength)
                plot_traj(time_vec, X_sim, U_sim, Cost_sim, env)
                plt.show()

    return h_ctg

if __name__=="__main__":
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES                    = 500       # Number of training episodes
    NPRINT               = NEPISODES/5       # print something every NPRINT episodes
    MAX_EPISODE_LENGTH           = 100       # Max episode length
    QVALUE_LEARNING_RATE         = 1e-3      # alpha coefficient of Q learning algorithm
    DISCOUNT                     = 0.99      # Discount factor 
    PLOT                         = True      # Plot stuff if True
    PLOT_TRAJ                    = True      # Plot trajectory if True
    EXPLORATION_PROB             = 1         # initial exploration probability of eps-greedy policy
    EXPLORATION_DECREASING_DECAY = -np.log(5e-3)/NEPISODES     # exploration decay for exponential decreasing
    MIN_EXPLORATION_PROB         = 0.001     # minimum of exploration probability
    CAPACITY_BUFFER              = 500       # capacity buffer
    BATCH_SIZE                   = 32        # batch size 
    MIN_BUFFER                   = 100       # Start sampling from buffer when have length > MIN_BUFFER
    C_STEP                       = 4         # Every c step update w  
    # ----- Control/State
    njoint                       = 1         # number of joint
    nx                           = 2*njoint  # number of states
    nu                           = 1         # number of control input
    nd_u                         = 21        # number of discretization steps for the joint torque u
    nd_x                         = 21        # number of discretization steps for the joint state (for plot)
    # ----- FLAG to TRAIN/LOAD
    FLAG                         = True # False = Load Model

    ### --- Initialize agent, buffer and enviroment
    env = Pendulum_dci(njoint, nd_u)

    agent = DQNagent(nx, nu, env, DISCOUNT, QVALUE_LEARNING_RATE)
    # agent.Q.summary()
    # agent.Q_target.summary()

    buffer = ReplayBuffer(CAPACITY_BUFFER, BATCH_SIZE)

    if FLAG == True:

        h_ctg = dqn_learning(buffer, agent, env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH, MIN_BUFFER, C_STEP, EXPLORATION_PROB, EXPLORATION_DECREASING_DECAY, MIN_EXPLORATION_PROB, plot_traj, PLOT, NPRINT)
        plt.show()

        # save model and weights
        print("\nTraining finished")
        print("\nSave NN weights to file (in HDF5)")
        if (njoint == 1):
            agent.Q.save('saved_model/my_model1')
            agent.Q.save_weights('saved_model/weight1.h5')
        else:    
            agent.Q.save('saved_model/my_model2')
            agent.Q.save_weights('saved_model/weight2.h5')

        #plot cost
        plt.figure()
        plt.plot( np.cumsum(h_ctg)/range(1,NEPISODES+1) )
        plt.title ("Average cost-to-go")

    if FLAG == False: #load model
        if (njoint == 1):
            agent.Q = tf.keras.models.load_model('saved_model/my_model1')
        else:
            agent.Q = tf.keras.models.load_model('saved_model/my_model2')
        assert(agent.Q)
    
    if (njoint == 1): #plot V, pi for joint 1
        V, pi, xgrid = compute_V_pi_from_Q(env.d2cu,agent, env)
        env.plot_V_table(V, xgrid)
        env.plot_policy(pi, xgrid)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
        
    X_sim, U_sim, Cost_sim = render_greedy_policy(env, agent, EXPLORATION_PROB, None, MAX_EPISODE_LENGTH)

    if PLOT_TRAJ:
        time_vec = np.linspace(0.0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
        plot_traj(time_vec, X_sim, U_sim, Cost_sim, env)

    plt.figure.max_open_warning = 50
    plt.show()