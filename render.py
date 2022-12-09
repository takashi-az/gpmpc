import matplotlib.pyplot as plt
import numpy as np

def rov_setdata(ax,x,y,theta,arm_len=0.2):
        x_arr = [x+arm_len*(np.cos(theta+np.pi/2)-np.cos(theta)),
                x+arm_len*(np.cos(theta+np.pi/2)+np.cos(theta)),
                x+arm_len*(np.cos(theta+np.pi/2)+np.cos(theta))-arm_len*np.cos(theta+np.pi/2),
                x+arm_len*np.cos(theta)+arm_len*np.cos(theta),
                x+arm_len*(np.cos(theta+np.pi/2)+np.cos(theta))-arm_len*np.cos(theta+np.pi/2),
                x+arm_len*(-np.cos(theta+np.pi/2)+np.cos(theta)),
                x+arm_len*(-np.cos(theta+np.pi/2)-np.cos(theta)),
                x+arm_len*(np.cos(theta+np.pi/2)-np.cos(theta))]
        y_arr = [y+arm_len*(np.sin(theta+np.pi/2)-np.sin(theta)),
                y+arm_len*(np.sin(theta+np.pi/2)+np.sin(theta)),
                y+arm_len*(np.sin(theta+np.pi/2)+np.sin(theta))-arm_len*np.sin(theta+np.pi/2),
                y+arm_len*np.sin(theta)+arm_len*np.sin(theta),
                y+arm_len*(np.sin(theta+np.pi/2)+np.sin(theta))-arm_len*np.sin(theta+np.pi/2),
                y+arm_len*(-np.sin(theta+np.pi/2)+np.sin(theta)),
                y+arm_len*(-np.sin(theta+np.pi/2)-np.sin(theta)),
                y+arm_len*(np.sin(theta+np.pi/2)-np.sin(theta))]
        rov, = ax.plot(x_arr,y_arr,color='b')
        return rov