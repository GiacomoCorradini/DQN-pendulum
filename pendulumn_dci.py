from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
    

class Pendulum_dci:
    ''' 
        Describe continuous state pendulum environment with discrete control input. Torque is discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, joint = 1, nu=11, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0.0):
        self.pendulum = Pendulum(joint,noise_stddev)
        self.pendulum.DT  = dt    # Time step length
        self.pendulum.NDT = ndt   # Number of Euler steps per integration (internal)
        self.nu = nu              # Number of discretization steps for joint torque
                                  # the value above must be odd

        self.vMax = vMax          # Max velocity (v in [-vmax,vmax])
        self.uMax = uMax          # Max torque (u in [-umax,umax])
        self.dt = dt              # time step
        self.DU = 2*uMax/nu       # discretization resolution for joint torque

    # @property
    # def nqv(self): return [self.nq,self.nv]
    # @property
    # def nx(self): return self.nq*self.nv
    
    # vertical position
    @property
    def goal(self): return 0.
    
    # Continuous to discrete c2d..
    # def c2dq(self, q):
    #     q = (q+pi)%(2*pi) # q is between 0 and 2pi
    #     return int(np.floor(q/self.DQ))  % self.nq

    # def c2dv(self, v):
    #     v = np.clip(v,-self.vMax+1e-3,self.vMax-1e-3)
    #     return int(np.floor((v+self.vMax)/self.DV))

    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))

    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU

    # def c2d(self, qv):
    #     '''
    #         From continuous to 2d discrete.
    #         input: qv is a general vector
    #         output: return discrete q and v
    #     '''
    #     return np.array([self.c2dq(qv[0]), self.c2dv(qv[1])])

    # Discrete to continuous
    # def d2cq(self, iq):
    #     iq = np.clip(iq,0,self.nq-1)
    #     return iq*self.DQ - pi + 0.5*self.DQ
    
    # def d2cv(self, iv):
    #     iv = np.clip(iv,0,self.nv-1) - (self.nv-1)/2
    #     return iv*self.DV
    
    # def d2c(self, iqv):
    #     '''From 2d discrete to continuous'''
    #     return np.array([self.d2cq(iqv[0]), self.d2cv(iqv[1])])
    
    # in renforcement learning works with one single value so we have to convert into  a single value

    # ''' From 2d discrete to 1d discrete '''
    # def x2i(self, x): return x[0]+x[1]*self.nq
    
    # ''' From 1d discrete to 2d discrete '''
    # def i2x(self, i): return [ i%self.nq, int(np.floor(i/self.nq)) ]

    # use the continuous time reset
    def reset(self,x=None):
        # if x is None:
        #     self.x = np.random.randint(0,self.nx)
        # else: 
        #     self.x = x
        #return self.x
        return self.pendulum.reset(x)

    def step(self,iu):
        ''' Simulate one time step '''
        cost     = -1 if self.x==self.goal else 0
        self.x   = self.dynamics(self.x,iu)
        return self.x, cost

    def render(self):
        self.pendulum.render()
        #q = self.d2cq(self.i2x(self.x)[0])
        #self.pendulum.display(np.array([self.x[0],]))
        #time.sleep(self.pendulum.DT)

    def dynamics(self,iu):
        #x   = self.d2c(self.i2x(ix))
        u   = self.d2cu(iu)
        self.xc,_ = self.pendulum.dynamics(self.x,u)
        return self.xc
    
    def plot_V_table(self, V):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt
        Q,DQ = np.meshgrid(range(V.shape[0]),range(V.shape[0]))
        plt.pcolormesh(Q, DQ, V.T, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('V table')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_policy(self, pi):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt
        Q,DQ = np.meshgrid(range(pi.shape[0]),range(pi.shape[0]))
        plt.pcolormesh(Q, DQ, pi.T, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title('Policy')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_Q_table(self, Q):
        ''' Plot the given Q table '''
        import matplotlib.pyplot as plt
        X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
        plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('Q table')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.show()