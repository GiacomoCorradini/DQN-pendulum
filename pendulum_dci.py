from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
    

class Pendulum_dci:
    ''' 
        Describe continuous state pendulum environment with discrete control input. Torque is discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, n_joint = 1, ndu=11, vMax=8, uMax=5, dt=5e-2, ndt=1, noise_stddev=0):
        self.njoint       = n_joint
        self.pendulum     = Pendulum(n_joint,noise_stddev, vMax, uMax)
        self.pendulum.DT  = dt         # Time step length
        self.pendulum.NDT = ndt        # Number of Euler steps per integration (internal)
        self.ndu          = ndu         # Number of discretization steps for joint torque
                                       # the value above must be odd
        self.vMax         = vMax       # Max velocity (v in [-vmax,vmax])
        self.uMax         = uMax       # Max torque (u in [-umax,umax])
        self.dt           = dt         # time step
        self.DU           = 2*uMax/ndu  # discretization resolution for joint torque


    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))

    def d2cu(self, iu):
        iu = np.clip(iu,0,self.ndu-1) - (self.ndu-1)/2
        return iu*self.DU

    # use the continuous time reset
    def reset(self,x=None):
        self.x = self.pendulum.reset(x)
        return self.x

    def step(self,iu):
        ''' Simulate one time step '''
        u   = self.d2cu(iu)
        self.x, cost = self.pendulum.step(u)
        return self.x, cost

    def render(self):
        self.pendulum.render()
        time.sleep(self.pendulum.DT)
    
    def plot_V_table(self, V, x, i=0):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(x[0], x[1], V, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title("V table %d" %i)
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show(block=False)
        
    def plot_policy(self, pi, x, i=0):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(x[0], x[1], pi, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title("Policy %d" %i)
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show(block=False)
            
if __name__=="__main__":

    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = Pendulum_dci(1)   

    x0 = x = env.reset()
    u = np.zeros(env.pendulum.nu)
    uaux = np.zeros(env.pendulum.nu)

    cost = []
    X = []
    V = []
    U = []

    for i in range(100):
        u[0] += 0.01
        if env.pendulum.nu == 2:
            u[1] = 0
            U.append([u[0],u[1]])
            uaux[1] = u[1]
        else:   
            U.append(u[0])
        uaux[0] = env.c2du(u[0])
        x,c = env.step(uaux)
        cost.append(c)
        X.append(x[:env.pendulum.nq])
        V.append(x[env.pendulum.nq:])
        #env.render()
    print(x)
    print(env.d2cu(0), env.d2cu(10), env.d2cu(5))
        
    plt.figure()
    plt.plot( np.cumsum(cost)/range(1,100+1) )
    plt.title("cost")
    plt.figure()
    plt.plot(np.reshape(X,(100,env.pendulum.nq)))
    plt.title("pos")
    plt.figure()
    plt.plot(np.reshape(V,(100,env.pendulum.nq)))
    plt.title("vel")
    plt.figure()
    plt.plot(np.reshape(U,(100,env.pendulum.nu)))
    plt.title("torque")
    plt.show()
