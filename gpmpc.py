import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
from env import env
from gp import GP
from cem import CEM
from render import rov_setdata

def RMSE(x,y):
    rmse = [0.0 for i in range(len(x[0]))]
    for i in range(len(x[0])):
        sum_ = 0.0
        for j in range(len(x)):
            sum_ += (x[j][i] - y[j][i])**2
        sum_ = np.sqrt(sum_/len(x))
        rmse[i] = sum_
    return rmse

def log_rmse(rmse,dir):
    row = 2
    column = 3
    title = ['x','y','yaw','vx','vy','r']
    path = dir + '/rmses.png'
    fig = plt.figure(figsize=(6*column,5*row))
    for i in range(len(rmse[0])):
        ax = fig.add_subplot(row,column,i+1)
        x = []
        y = []
        for j in range(len(rmse)):
            x.append(j+1)
            y.append(rmse[j][i])
        ax.plot(x,y)
        ax.set_xlabel('episode')
        ax.set_ylabel('RMSE')
        ax.grid()
        ax.set_title(title[i])
    plt.savefig(path)
    path = dir + '/rmse.png'
    fig = plt.figure(figsize=(5,5))
    sum_ = []
    for i in range(len(rmse)):
        sum_.append(sum(rmse[i]))
    plt.plot(sum_)
    plt.title('RMSE')
    plt.xlabel('episode')
    plt.ylabel('RMSE')
    plt.grid()
    plt.savefig(path)
    

def log_traj(buffer,ref,dir,episode):
    row = 2
    column = 3
    title = ['x','y','yaw','vx','vy','r']
    path = dir+'/episode_{}'.format(episode)
    os.mkdir(path)
    fig = plt.figure(figsize=(4*column,4*row))
    for i in range(len(buffer[0])):
        ax = fig.add_subplot(row,column,i+1)
        x = []
        y = []
        for j in range(len(buffer)):
            x.append(j+1)
            y.append(buffer[j][i])
        ax.plot(x,y,label='traj')
        ax.plot(ref[:,i],label='ref')
        ax.grid()
        ax.set_title(title[i])
        ax.legend()
    png = path+'/traj.png'
    plt.savefig(png)

def log_input(buffer,dir,episode):
    row = 2
    column = 2
    title = ['u1','u2','u3','u4']
    path = dir+'/episode_{}'.format(episode)
    fig = plt.figure(figsize=(4*column,4*row))
    for i in range(len(buffer[0])):
        ax = fig.add_subplot(row,column,i+1)
        x = []
        y = []
        for j in range(len(buffer)):
            x.append(j+1)
            y.append(buffer[j][i])
        ax.plot(x,y)
        ax.grid()
        ax.set_title(title[i])
    png = path+'/input.png'
    plt.savefig(png)

def log_traj_img(buffer,dir,episode):
    fig = plt.figure(figsize=(8,8))
    plt.xlabel('x',fontsize='15')
    plt.ylabel('y',fontsize='15')
    plt.xlim([-0.5,4.5])
    plt.ylim([-2.5,2.5])
    x = np.linspace(0,1.2*np.pi,1000)
    ref = 0.5*np.sin(x/1.2*2)
    plt.plot(x,ref,linestyle = "dashed",color='black',label='ref')
    plt.plot(buffer[0],buffer[1],color='red',label='trajectory')
    plt.legend(fontsize='15')
    png = dir + '/episode_{}'.format(episode) + '/img.png'
    plt.savefig(png)

env = env()
gp = GP()
mpc = CEM(gp)

date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = './log/' + date

if not os.path.isdir('./log'):
    os.mkdir('./log')
os.mkdir(path)

fig,ax = plt.subplots(figsize=(6,6))
ax.set_xlim([-0.5,4.5])
ax.set_ylim([-2.5,2.5])
x = np.linspace(0,1.2*np.pi,1000)
ref = 0.5*np.sin(x/1.2*2)
ax.plot(x,ref,linestyle = "dashed",color='black',label='ref')
ax.set_xlabel('x',fontsize='15')
ax.set_ylabel('y',fontsize='15')
ax.legend(fontsize='20')

episode = 0
t = 1
s = env.reset()
done = False
state_x = []
state_y = []
state = []
action = []
rmse = []
ref = mpc.create_ref(task_horizon=100)

while 1:
    state_x.append(s[0])
    state_y.append(s[1])
    state.append(s.copy())
    pos = rov_setdata(ax,env.x[0],env.x[1],env.x[2])
    tra, = ax.plot(state_x,state_y,color='red')
    # ref_p = ax.scatter(ref[t-1][0],ref[t-1][1],color='blue')
    plt.pause(0.0001)
    pos.remove()
    tra.remove()
    # ref_p.remove()
    if done:
        ref = mpc.create_ref(task_horizon=100)
        rmse_ = RMSE(state[:100],ref)
        rmse.append(rmse_)
        log_rmse(rmse,path)
        log_traj(state,ref,path,episode)
        log_input(action,path,episode)
        log_traj_img([state_x,state_y],path,episode)
        # png = dir+'/episode_{}'.format(episode)+'/traj.png'
        # plt.savefig(png)
        state_x = []
        state_y = []
        state = []
        action = []
        s = env.reset()
        t = 0
        # mpc.min_idx = 0
        gp.log()
        gp.create_gram_matrix()
        episode += 1
        mpc.done = False
    start = time.time()
    if episode==0:
        u = np.random.rand(4)*6-3
    else:
        u = mpc.act(s)
    action.append(u)
    gp.memory(env.get_state(),u)
    print("action time:",round(time.time()-start,5))
    print("step:",t+1)
    next_s = env.step(u)
    s = next_s
    t += 1
    # done = (t==100)
    done = mpc.done
    if episode==0:
        done = (t==100) 