import numpy as np

class PID():
    def __init__(self):
        self.DT = 0.2
        self.Kp = 0.5
        self.Kd = 0.2
        self.Ki = 0.2
        self.e1 = np.zeros(4)
        self.e2 = np.zeros(4)
        self.pred_u = np.zeros(4)
        self.task_horizon = 100
        self.n_ahead = 1
        self.g_traj = self.create_ref(self.task_horizon)
    
    def create_ref(self,task_horizon):
        x = np.linspace(0,2*np.pi,task_horizon).reshape(-1,1)
        y = np.sin(x).reshape(-1,1)
        w = np.arctan(np.cos(x)).reshape(-1,1)
        vx = (np.sqrt((np.append(x[1:],2*np.pi).reshape(-1,1)-x)**2+(np.append(y[1:],0).reshape(-1,1)-y)**2))/self.DT
        vx[-1] = vx[-2]
        vy = np.zeros(len(x)).reshape(-1,1)
        r = ((np.append(w[1:],w[-1]).reshape(-1,1)-w)/self.DT)
        return np.concatenate([x,y,w,vx,vy,r],axis=1)
    
    def act(self,curr_x):
        x,y,w,vx,vy,r = curr_x
        goal = self.plan(curr_x,self.g_traj)
        error = (goal-curr_x)/self.DT
        max_ = 10000
        out = np.zeros(0)
        for i in range(100):
            u = np.random.rand(4)*6-3
            xdot = self.calc_xdot(curr_x,u)
            rmse_ = self.rmse(error,xdot)
            if rmse_ < max_:
                out = u.copy()
                max_ = rmse_
        # self.e1 = self.e1.clip(-1,1)
        # self.e2 = self.e2.clip(-1,1)
        u = self.pred_u + self.Kp*(self.e1) + self.Ki*(out-self.pred_u) + self.Kd*((out-self.pred_u)-self.e1*2+self.e2)
        u = u.clip(-3,3)
        self.e2 = self.e1.copy()
        self.e1 = out-self.pred_u.copy()
        self.pred_u = u.copy()
        return u
    
    def rmse(self,x,y):
        return np.sqrt(np.mean((x-y)**2))

    def calc_xdot(self,curr_x,inp):
        x,y,psi,u,v,r = curr_x
        m = 11.5;Iz = 0.16
        Xudot = -5.5;Yvdot = -12.7;Nrdot = -0.12
        Xu = -4.03;Yv = -6.22;Nr = -0.07
        Xuu = -18.18;Yvv = -21.66;Nrr = -1.55
        Mrb = np.array([[ m , 0 , 0 ],
                      [ 0 , m , 0 ],
                      [ 0 , 0 , Iz]])
        Ma = -np.array([[Xudot,0,0],
                        [0,Yvdot,0],
                        [0,0,Nrdot]])
        g = np.array([[0],
                      [0],
                      [0]])
        Crb = np.array([[0,0,-m*v],
                        [0,0,m*u],
                        [m*v,-m*u,0]])
        Ca = np.array([[0,0,Yvdot*v],
                       [0,0,-Xudot*u],
                       [-Yvdot*v,Xudot*u,0]])
        D = -np.array([[Xu+Xuu*np.abs(u),0,0],
                       [0,Yv+Yvv*np.abs(v),0],
                       [0,0,Nr+Nrr*np.abs(r)]])
        T = np.array([[0.707,0.707,-0.707,-0.707],
                      [-0.707,0.707,-0.707,0.707],
                      [-0.1888,0.1888,0.1888,-0.1888]])
        J = np.array([[np.cos(psi),-np.sin(psi),0],
                      [np.sin(psi),np.cos(psi),0],
                      [0,0,1]])
        Minv = np.linalg.inv(Mrb+Ma)
        xdot = np.block([[np.zeros([3,3]),J],[np.zeros([3,3]),-Minv@(Crb+Ca+D)]])@curr_x.T+np.block([[np.zeros([3,3])],[Minv]])@T@inp

        return xdot

    def plan(self, curr_x, g_traj):
        min_idx = np.argmin(np.linalg.norm(curr_x[:-1] - g_traj[:, :-1],
                                           axis=1))

        end = (min_idx+self.n_ahead)
        if end > len(g_traj):
            end = len(g_traj)

        return g_traj[end]