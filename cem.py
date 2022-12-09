import numpy as np
# import scipy.stats as stats

class CEM():
    def __init__(self,model):
        self.model = model
        self.DT = 0.2
        self.popsize = 50
        self.pred_len = 3
        self.num_elites = 20
        self.bootstrap = 3
        self.n_ahead = 1
        self.action_size = 4
        self.state_size = 6
        self.task_horizon = 100
        self.pred_input = np.zeros((self.pred_len,self.action_size))
        # self.min_idx = 0
        self.done = False
        self.g_traj = self.create_ref(self.task_horizon)
    
    def create_ref(self,task_horizon):
        x = np.linspace(0,1.2*np.pi,task_horizon).reshape(-1,1)
        y = 0.5*np.sin(x/1.2*2).reshape(-1,1)
        w = np.arctan(np.cos(x/1.2*2)).reshape(-1,1)
        vx = (np.sqrt((np.append(x[1:],1.2*np.pi).reshape(-1,1)-x)**2+(np.append(y[1:],0).reshape(-1,1)-y)**2))/self.DT
        vx[-1]=vx[-2]
        vy = np.zeros(len(x)).reshape(-1,1)
        r = ((np.append(w[1:],w[-1]).reshape(-1,1)-w)/self.DT)
        return np.concatenate([x,y,w,vx,vy,r],axis=1)
    
    def plan(self, curr_x, g_traj):
        """
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size)
            g_traj (numpy.ndarray): goal state, shape(ref_size,state_size),
                this state should be obtained from env
        Returns:
            g_xs (numpy.ndarrya): goal state, shape(pred_len+1, state_size)
        """
        min_idx = np.argmin(np.linalg.norm(curr_x[:-1] - g_traj[:, :-1],
                                           axis=1))

        # min_idx = self.min_idx

        if min_idx==len(g_traj)-1:
            self.done = True

        start = (min_idx+self.n_ahead)
        if start > len(g_traj):
            start = len(g_traj)

        end = min_idx+self.n_ahead+self.pred_len+1

        if (min_idx+self.n_ahead+self.pred_len+1) > len(g_traj):
            end = len(g_traj)

        if abs(start - end) != self.pred_len+1:
            return np.tile(g_traj[-1], (self.pred_len+1, 1))

        return g_traj[start:end]
    
    def state_cost_fn(self,x, g_x):
        """ state cost function
        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, 1) or
                shape(pop_size, pred_len, 1)
        """
        pred_len,_ = x.shape
        state_cost = 0
        R = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,0.2,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0]])
        for i in range(pred_len):
            state_cost += (x[i]-g_x[i])@R@(x[i]-g_x[i]).T
        
        return state_cost

    def input_cost_fn(self,u):
        """ input cost functions
        Args:
            u (numpy.ndarray): input, shape(pred_len, input_size)
                or shape(pop_size, pred_len, input_size)
        Returns:
            cost (numpy.ndarray): cost of input, shape(pred_len, input_size) or
                shape(pop_size, pred_len, input_size)
        """
        pred_len,_ = u.shape
        input_cost = 0
        Q = np.array([[0.0001,0,0,0],
                      [0,0.0001,0,0],
                      [0,0,0.0001,0],
                      [0,0,0,0.0001]])
        for i in range(pred_len):
            input_cost += u[i]@Q@u[i].T
            
        return input_cost
    
    def terminal_state_cost_fn(self,terminal_x, terminal_g_x):
        """
        Args:
            terminal_x (numpy.ndarray): terminal state,
                shape(state_size, ) or shape(pop_size, state_size)
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, ) or shape(pop_size, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, ) or
                shape(pop_size, pred_len)
        """
        terminal_state_cost = 0
        S = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0]])
        terminal_state_cost = 0.5*(terminal_x-terminal_g_x)@S@(terminal_x-terminal_g_x).T
        return terminal_state_cost
    
    def cost_fn(self,curr_x,samples,g_x):
        """
        calculate samples costs every particle

        Args curr_x(state_size)
            samples(popsize,pred_len,action_size)
            g_x (pred_len,state_size)

        return cost (popsize,)
        """
        costs = np.zeros(self.popsize)
        for i in range(self.popsize):
            state_traj = self.predict_traj(curr_x,samples[i])
            costs[i] = self.state_cost_fn(state_traj,g_x) + self.input_cost_fn(samples[i]) + self.terminal_state_cost_fn(state_traj[-1],g_x[-1])

        return costs

    def f_nom(self,curr_x,inp):
        x,y,psi,u,v,r = curr_x
        m = 11.5;Iz = 0.16
        Xudot = -5.5;Yvdot = -12.7;Nrdot = -0.12
        Mrb = np.array([[ m , 0 , 0 ],
                      [ 0 , m , 0 ],
                      [ 0 , 0 , Iz]])
        Ma = -np.array([[Xudot,0,0],
                        [0,Yvdot,0],
                        [0,0,Nrdot]])
        g = np.array([[0],
                      [0],
                      [0]])
        Minv = np.linalg.inv(Mrb+Ma)
        T = np.array([[0.707,0.707,-0.707,-0.707],
                      [-0.707,0.707,-0.707,0.707],
                      [-0.1888,0.1888,0.1888,-0.1888]])
        J = np.array([[np.cos(psi),-np.sin(psi),0],
                      [np.sin(psi),np.cos(psi),0],
                      [0,0,1]])
        xdot = np.block([[np.zeros([3,3]),J],[np.zeros([3,3]),np.zeros([3,3])]])@curr_x.T+np.block([[np.zeros([3,3])],[Minv]])@T@inp.T
        return xdot

    def f_dis(self,x,inp):
        u,v,r = x[3:]
        T = np.array([[0.707,0.707,-0.707,-0.707],
                      [-0.707,0.707,-0.707,0.707],
                      [-0.1888,0.1888,0.1888,-0.1888]])
        a = T@inp.T
        x_ = np.array([[u,v,r,a[0],a[1],a[2]]])
        xdot = self.model.predict(x_)

        return xdot
    
    def predict_next_state(self, curr_x, u):
        xdot = self.f_nom(curr_x,u) + self.f_dis(curr_x, u)
        next_x = curr_x + xdot*self.DT
        return next_x
    
    def predict_traj(self,curr_x,us):
        pred_len = us.shape[0]
        # initialze
        x = curr_x
        pred_xs = curr_x[np.newaxis, :]

        for t in range(pred_len):
            next_x = self.predict_next_state(x, us[t])
            pred_xs = np.concatenate((pred_xs, next_x[np.newaxis, :]), axis=0)
            x = next_x

        return pred_xs

    def act(self,curr_x,epsilon=0.1,alpha=0.3,episode_reset=False):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_mean.shape(pred_len,action_size)
            sigma (np.ndarray): The variance of the initial candidate distribution.
            sigma.shape(action_size,pred_len)
        """
        mean,t = self.pred_input, 0
        sigma = np.ones((self.pred_len,self.action_size))
        plan_traj = self.plan(curr_x,self.g_traj)

        while (t < self.bootstrap) and np.max(sigma) > epsilon:
            samples = np.random.randn(self.popsize,self.pred_len,self.action_size)*sigma + mean
            samples = np.clip(samples,-3,3)

            costs = self.cost_fn(curr_x,samples,plan_traj)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_sigma = np.std(elites, axis=0)

            mean = alpha * mean + (1 - alpha) * new_mean
            sigma = alpha * sigma + (1 - alpha) * new_sigma

            t += 1
        
        self.pred_input = mean
        # self.min_idx += 1

        return mean[0]