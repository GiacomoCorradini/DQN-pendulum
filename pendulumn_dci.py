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
    def __init__(self, n_joint = 1, nu=11, vMax=5, uMax=2, dt=5e-2, ndt=1, noise_stddev=0):
        self.njoint       = n_joint
        self.pendulum     = Pendulum(n_joint,noise_stddev)
        self.pendulum.DT  = dt         # Time step length
        self.pendulum.NDT = ndt        # Number of Euler steps per integration (internal)
        self.nu           = nu         # Number of discretization steps for joint torque
                                       # the value above must be odd
        self.vMax         = vMax       # Max velocity (v in [-vmax,vmax])
        self.uMax         = uMax       # Max torque (u in [-umax,umax])
        self.dt           = dt         # time step
        self.DU           = 2*uMax/nu  # discretization resolution for joint torque


    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))

    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
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
        self.pendulum.display(np.array([self.x[0],]))
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

    env = Pendulum_dci()   

    x0 = x = env.reset(np.asarray([[0.],[0.]]))
    print(x)
    u = env.c2du(0)
    #print(u)
    #print(env.d2cu(u))
    cost = []
    X = []
    V = []
    for i in range(100):
        u += 0.01
        x,c = env.step([u])
        cost.append(c)
        X.append(x[:env.pendulum.nq])
        V.append(x[env.pendulum.nq:])
        env.render()
        #print(c)
    print(x)
        
    plt.figure()
    plt.plot( np.cumsum(cost)/range(1,100+1) )
    plt.title("cost")
    plt.figure()
    plt.plot(np.concatenate(X))
    plt.title("pos")
    plt.figure()
    plt.plot(np.concatenate(V))
    plt.title("vel")
    plt.show()