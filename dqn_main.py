import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
import time
from numpy.random import rand
import tensorflow as tf

from pendulum_dci import Pendulum_dci
from dqn_agent import DQNagent
from replay_buffer import ReplayBuffer

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def render_greedy_policy(env, agent, exploration_prob, x0=None, maxiter=100):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    costToGo = 0.0
    gamma_i  = 1
    X_sim    = np.zeros([maxiter,env.pendulum.nx])   # store x
    U_sim    = np.zeros(maxiter)                     # store u
    for i in range(maxiter):
        u = agent.get_action(exploration_prob, env, x, False)
        if(env.njoint == 2):
            x,c      = env.step([u,env.c2du(0)])
        else:    x,c      = env.step([u])
        costToGo += gamma_i*c
        gamma_i  *= agent.DISCOUNT
        #env.render()
        X_sim[i,:] = np.concatenate(np.array([x]).T)
        U_sim[i]   = env.d2cu(u)
    print("Real cost to go of state", x0, ":", costToGo)
    return X_sim, U_sim

def compute_V_pi_from_Q(agent, vMax=5, xstep=20, nx=2):
    ''' Compute Value table and greedy policy pi from Q table. '''

    x      = np.empty(shape = (nx,xstep+1))
    DQ     = 2*np.pi/xstep
    DV     = 2*vMax/xstep
    x[0,:] = np.arange(-np.pi,np.pi+DQ, DQ)
    x[1,:] = np.arange(-vMax,vMax+DV, DV)

    pi     = np.empty(shape = (xstep+1,xstep+1))
    # pi2    = np.empty(shape = (xstep+1,xstep+1))
    V      = np.empty(shape = (xstep+1,xstep+1))
    # Q      = np.empty(shape = (xstep+1,xstep+1,agent.ndu))

    for i in range(np.shape(x)[1]):
        for j in range(np.shape(x)[1]):
            xu = np.reshape([x[0,i]*np.ones(agent.ndu),x[1,j]*np.ones(agent.ndu),np.arange(agent.ndu)],(nx+1,agent.ndu))
            V[i,j]   = np.min(agent.Q(xu.T))
            pi[i,j]  = np.argmin(agent.Q(xu.T))
            # Q[i,j,:] = agent.tf2np(agent.Q(xu))
            # u_best   = np.where(Q[i,j,:]==V[i,j])[0]
            # if(u_best[0]>env.c2du(0.0)):
            #     pi2[i,j] = u_best[-1]
            # elif(u_best[-1]<env.c2du(0.0)):
            #     pi2[i,j] = u_best[0]
            # else:
            #     pi2[i,j] = u_best[int(u_best.shape[0]/2)]
    return V, pi, x


def dqn_learning(buffer, agent, env,\
                 gamma, nEpisodes, maxEpisodeLength, min_buffer, c_step,\
                 exploration_prob, exploration_decreasing_decay, min_exploration_prob, \
                 compute_V_pi_from_Q, plot=False, nprint=1000):
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
    # Keep track of the cost-to-go history (for plot)
    h_ctg      = []
    i_fin      = np.zeros(int(nEpisodes/nprint))
    J_fin      = np.zeros(int(nEpisodes/nprint))
    eps_fin    = np.zeros(int(nEpisodes/nprint))
    # Make a copy of the initial Q table guess
    Q = tf.keras.models.clone_model(agent.Q)

    # count the n° of episodes
    ep       = 0
    X_sim    = np.zeros([maxEpisodeLength,env.pendulum.nx])   # store x
    U_sim    = np.zeros(maxEpisodeLength)                     # store u
    # for every episode
    for i in range(nEpisodes):
        # reset the state
        env.reset()
        J = 0
        ep += 1
        gamma_to_the_i = 1
        # simulate the system for maxEpisodeLength steps
        for k in range(maxEpisodeLength):
            # state of the enviroment
            x = env.x

            # epsilon-greedy action selection
            u = agent.get_action(exploration_prob, env, x, True)

            # observe cost and next state (step = calculate dynamics)
            if (env.njoint == 2):
                x_next, cost = env.step([u,env.c2du(0)])
            else: x_next, cost = env.step([u])
            print(u)

            # next control greedy
            u_next = agent.get_action(exploration_prob, env, x_next, False)
        
            # store the experience (s,a,r,s',a') in the replay_buffer
            buffer.store_experience(x, u, cost, x_next, u_next)
            X_sim[k,:] = x
            U_sim[k] = env.d2cu(u)

            if buffer.get_length() > 0 and k % c_step == 0:
                # Randomly sample minibatch (size of batch_size) of experience from replay_buffer
                # collect together state and control
                xu_batch, xu_next_batch, cost_batch = buffer.sample_batch(env)

                # convert numpy to tensorflow
                xu_batch      = agent.np2tf(xu_batch)
                cost_batch    = agent.np2tf(cost_batch)
                xu_next_batch = agent.np2tf(xu_next_batch)

                # optimizer with SGD
                agent.update(xu_batch, cost_batch, xu_next_batch)
                
                # Periodically update target network (period = c_step)
                agent.update_Q_target()
        
            # keep track of the cost to go
            J += gamma_to_the_i * cost
            gamma_to_the_i *= gamma
        
        J_avg = J / ep
        h_ctg.append(J_avg)

        # update the exploration probability with an exponential decay: 
        exploration_prob = max(np.exp(-exploration_decreasing_decay*i), min_exploration_prob)
        
        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(i%nprint==0 and i>=nprint):
            print("DQN - Episode %d, J=%.1f, eps=%.1f"%(i,J,100*exploration_prob))
            iaux = int(i/nprint)
            i_fin[iaux]   = i
            J_fin[iaux]   = J
            eps_fin[iaux] = exploration_prob
            if(plot):
                if(env.njoint == 1):
                    V, pi, xgrid = compute_V_pi_from_Q(agent)
                    env.plot_V_table(V, xgrid, iaux)
                    env.plot_policy(pi, xgrid, iaux)
                # env.plot_policy(pi2, xgrid, iaux)
                time_vec = np.linspace(0.0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
                plt.figure()
                plt.plot(time_vec, U_sim[:], "b")
                if env.uMax:
                    plt.plot(time_vec, env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
                    plt.plot(time_vec, -env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
                plt.gca().set_xlabel('Time [s]')
                plt.gca().set_ylabel('[Nm]')
                plt.title ("Torque input")
            
                plt.figure()
                plt.plot(time_vec, X_sim[:,0],'b')
                if env.njoint == 2:
                    plt.plot(time_vec, X_sim[:,1],'r')
                    plt.legend(["1st joint position","2nd joint position"],loc='upper right')
                plt.gca().set_xlabel('Time [s]')
                plt.gca().set_ylabel('[rad]')
                plt.title ("Joint position")
                
                plt.figure()
                if env.njoint == 1:
                    plt.plot(time_vec, X_sim[:,1],'b')
                else:
                    plt.plot(time_vec, X_sim[:,2],'b')
                    plt.plot(time_vec, X_sim[:,3],'r')
                    plt.legend(["1st joint velocity","2nd joint velocity"],loc='upper right')
                plt.gca().set_xlabel('Time [s]')
                plt.gca().set_ylabel('[rad/s]')
                plt.title ("Joint velocity")

    for i in range(int(nEpisodes/nprint)):
        print("Q learning - Iter %d, J=%.1f, eps=%.1f"%(i_fin[i],J_fin[i],100*eps_fin[i]))

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
    EXPLORATION_DECREASING_DECAY = 0.001     # exploration decay for exponential decreasing
    MIN_EXPLORATION_PROB         = 0.001     # minimum of exploration proba
    CAPACITY_BUFFER              = 1000      # capacity buffer
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
    agent.Q.summary()
    agent.Q_target.summary()

    buffer = ReplayBuffer(CAPACITY_BUFFER, BATCH_SIZE)

    if FLAG == True:

        h_ctg = dqn_learning(buffer, agent, env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH, MIN_BUFFER, C_STEP, EXPLORATION_PROB, EXPLORATION_DECREASING_DECAY, MIN_EXPLORATION_PROB, compute_V_pi_from_Q, PLOT, NPRINT)
        
        # save model and weights
        print("\nTraining finished")
        agent.Q.save('saved_model/my_model')
        print("\nSave NN weights to file (in HDF5)")
        agent.Q.save_weights('saved_model/weight.h5')

        #plot cost
        plt.figure()
        plt.plot( np.cumsum(h_ctg)/range(1,NEPISODES+1) )
        plt.title ("Average cost-to-go")

    if FLAG == False: #load model
        agent.Q = tf.keras.models.load_model('saved_model/my_model')
        assert(agent.Q)
    
    if (njoint == 1): #plot V, pi for joint 1
        V, pi, xgrid = compute_V_pi_from_Q(agent)
        env.plot_V_table(V, xgrid)
        env.plot_policy(pi, xgrid)
        # env.plot_policy(pi2, xgrid)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
        
    X_sim, U_sim = render_greedy_policy(env, agent, EXPLORATION_PROB, None, MAX_EPISODE_LENGTH)

    if PLOT_TRAJ:
        time_vec = np.linspace(0.0,MAX_EPISODE_LENGTH*env.pendulum.DT,MAX_EPISODE_LENGTH)
        plt.figure()
        plt.plot(time_vec, U_sim[:], "b")
        if env.uMax:
            plt.plot(time_vec, env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
            plt.plot(time_vec, -env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('[Nm]')
        plt.title ("Torque input")
    
        plt.figure()
        plt.plot(time_vec, X_sim[:,0],'b')
        if env.njoint == 2:
            plt.plot(time_vec, X_sim[:,1],'r')
            plt.legend(["1st joint position","2nd joint position"],loc='upper right')
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('[rad]')
        plt.title ("Joint position")
        
        plt.figure()
        if env.njoint == 1:
            plt.plot(time_vec, X_sim[:,1],'b')
        else:
            plt.plot(time_vec, X_sim[:,2],'b')
            plt.plot(time_vec, X_sim[:,3],'r')
            plt.legend(["1st joint velocity","2nd joint velocity"],loc='upper right')
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('[rad/s]')
        plt.title ("Joint velocity")

    plt.figure.max_open_warning = 50
    
    plt.show()