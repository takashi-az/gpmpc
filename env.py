import numpy as np

class env():
    def __init__(self):
        self.x = self.init_pos()
        self.dt = 0.2
    
    def init_pos(self):
        task_horizon = 100
        DT = 0.2
        x = np.linspace(0,1.2*np.pi,task_horizon).reshape(-1,1)
        y = 0.5*np.sin(x/1.2*2).reshape(-1,1)
        w = np.arctan(np.cos(x/1.2*2)).reshape(-1,1)
        vx = (np.sqrt((np.append(x[1:],1.2*np.pi).reshape(-1,1)-x)**2+(np.append(y[1:],0).reshape(-1,1)-y)**2))/DT
        vx[-1] = vx[-2]
        vy = np.zeros(len(x)).reshape(-1,1)
        r = (np.append(w[1:],w[-1]).reshape(-1,1)-w)/DT
        return np.array([x[0],y[0],w[0],vx[0],vy[0],r[0]]).reshape(6,)
    
    def calc_xdot(self,input):
        x,y,psi,u,v,r = self.x
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
        xdot = np.block([[np.zeros([3,3]),J],[np.zeros([3,3]),-Minv@(Crb+Ca+D)]])@self.x.T+np.block([[np.zeros([3,3])],[Minv]])@T@input

        return xdot
    
    def get_state(self):
        return self.x
    
    def step(self,u):
        xdot = self.calc_xdot(u)
        self.x += xdot*self.dt
        self.x[3]+=np.random.randn()*np.sqrt(0.0004)
        self.x[4]+=np.random.randn()*np.sqrt(0.0004)
        return self.x
        # self.noise()

    def noise(self):
        self.x[0:2] += np.random.rand(2) 
        self.x[2] += np.random.rand()*np.pi/12
    
    def reset(self):
        self.x = self.init_pos()
        return self.x