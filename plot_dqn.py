import numpy as np
import matplotlib.pyplot as plt

def plot_traj(time_vec, X_sim, U_sim, Cost_sim, env):
    plt.figure()
    plt.plot(time_vec, U_sim[:], "b")
    if env.uMax:
        plt.plot(time_vec, env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
        plt.plot(time_vec, -env.uMax*np.ones(len(time_vec)), "k--", alpha=0.8, linewidth=1.5)
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[Nm]')
    plt.title ("Torque input")

    plt.figure()
    plt.plot(time_vec, Cost_sim[:], "b")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Cost')
    plt.title ("Cost")

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