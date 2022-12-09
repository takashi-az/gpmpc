import numpy as np

class MPC():
    def __init__(self,model):
        self.pred_len = 10
        self.state_len = 6
        self.action_len = 4
        self.Dt = 0.2
        self.optimizer_mode = "conjugate"
        self.threshold = 0.01
        self.learning_rate = 0.01 
        self.model = model
        self.prev_sol = np.zeros((self.pred_len,self.action_len))
        self.traj = np.array([])

    def act(self,curr_x,g_xs):
        """ calculate the optimal inputs
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        sol = self.prev_sol.copy()
        count = 0
        # use for Conjugate method
        conjugate_d = None
        conjugate_prev_d = None
        conjugate_s = None
        conjugate_beta = None

        while True:
            # shape(pred_len+1, state_size)
            pred_xs = self.predict_traj(curr_x, sol)
            # shape(pred_len, state_size)
            pred_lams = self.predict_adjoint_traj(pred_xs, sol, g_xs)

            F_hat = self.gradient_hamiltonian_input(
                pred_xs, pred_lams, sol, g_xs)

            if np.linalg.norm(F_hat) < self.threshold:
                break

            if count > self.max_iters:
                # logger.debug(" break max iteartion at F : `{}".format(
                #     np.linalg.norm(F_hat)))
                print("break max iteartion at F")
                break

            if self.optimizer_mode == "conjugate":
                conjugate_d = F_hat.flatten()

                if conjugate_prev_d is None:  # initial
                    conjugate_s = conjugate_d
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)
                else:
                    prev_d = np.dot(conjugate_prev_d, conjugate_prev_d)
                    d = np.dot(conjugate_d, conjugate_d - conjugate_prev_d)
                    conjugate_beta = (d + 1e-6) / (prev_d + 1e-6)

                    conjugate_s = conjugate_d + conjugate_beta * conjugate_s
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)

            def compute_eval_val(u):
                pred_xs = self.predict_traj(curr_x, u)
                state_cost = self.state_cost_fn(
                    pred_xs[1:-1], g_xs[1:-1])
                input_cost = self.input_cost_fn(u)
                terminal_cost = self.config.terminal_state_cost_fn(pred_xs[-1], g_xs[-1])
                return state_cost + input_cost + terminal_cost

            alpha = self.line_search(F_hat, sol,
                                compute_eval_val, init_alpha=self.learning_rate)

            sol -= alpha * F_hat
            count += 1

        # update us for next optimization
        self.prev_sol = np.concatenate(
            (sol[1:], np.zeros((1, self.input_size))), axis=0)

        return sol[0]

    def f_nom(self,x,inp):
        x,y,psi,u,v,r = x
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
        xdot = np.block([[np.zeros([3,3]),J],[np.zeros([3,3]),np.zeros([3,3])]])@self.x.T+np.block([[np.zeros([3,3])],[Minv]])@T@inp.T
        return xdot

    def f_dis(self,x,inp):
        u,v,r = x[3:]
        T = np.array([[0.707,0.707,-0.707,-0.707],
                      [-0.707,0.707,-0.707,0.707],
                      [-0.1888,0.1888,0.1888,-0.1888]])
        a = T@inp.T
        x_ = np.array([[u,v,r,a[0],a[1],a[2]]])
        xdot = self.model.predict(x,y,x_)

        return xdot


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
                      [0,0,1,0,0,0],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
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
        Q = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
        for i in range(pred_len):
            input_cost += u@Q@u.T
            
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
                      [0,0,1,0,0,0],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
        terminal_state_cost = 0.5*(terminal_x-terminal_g_x)@S@(terminal_x-terminal_g_x).T
        return terminal_state_cost
    
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
    
    def predict_next_state(self, curr_x, u):
        xdot = self.f_nom(curr_x,u) + self.f_dis(curr_x, u)
        next_x = curr_x + xdot*self.DT
        return next_x

    def predict_adjoint_traj(self, xs, us, g_xs):
         """
        Args:
            xs (numpy.ndarray): states trajectory, shape(pred_len+1, state_size)
            us (numpy.ndarray): inputs, shape(pred_len, input_size)
            g_xs (numpy.ndarray): goal states, shape(pred_len+1, state_size)
        Returns:
            lams (numpy.ndarray): adjoint state, shape(pred_len, state_size),
                adjoint size is the same as state_size
        Notes:
            Adjoint trajectory be computed by backward path.
            Usually, we should -\dot{lam} but in backward path case, we can use \dot{lam} directry 
        """
        # get size
        (pred_len, input_size) = us.shape
        # pred final adjoint state
        lam = self.predict_terminal_adjoint_state(xs[-1],
                                                  terminal_g_x=g_xs[-1])
        lams = lam[np.newaxis, :]

        for t in range(pred_len-1, 0, -1):
            prev_lam = \
                self.predict_adjoint_state(lam, xs[t], us[t],
                                           g_x=g_xs[t])
            # update
            lams = np.concatenate((prev_lam[np.newaxis, :], lams), axis=0)
            lam = prev_lam

        return lams
    
    def predict_terminal_adjoint_state(self, terminal_x, terminal_g_x=None):
        """ predict terminal adjoint state
        Args:
            terminal_x (numpy.ndarray): terminal state, shape(state_size, )
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, )
        Returns:
            terminal_lam (numpy.ndarray): terminal adjoint state,
                shape(state_size, )
        """
        S = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
        terminal_lam = (terminal_x-terminal_g_x)@S
        return terminal_lam #[state_size]

    def predict_adjoint_state(self, lam, x, u, g_x=None):
        """ predict adjoint states
        Args:
            lam (numpy.ndarray): adjoint state, shape(state_size, )
            x (numpy.ndarray): state, shape(state_size, )
            u (numpy.ndarray): input, shape(input_size, )
            goal (numpy.ndarray): goal state, shape(state_size, )
        Returns:
            prev_lam (numpy.ndarrya): previous adjoint state,
                shape(state_size, )
        """
        if len(u.shape) == 1:
            delta_lam = self.DT * \
                self.gradient_hamiltonian_state(x, lam, u, g_x)
            prev_lam = lam + delta_lam
            return prev_lam

        elif len(u.shape) == 2:
            raise ValueError
    
    def gradient_hamiltonian_state(self,x, lam, u, g_x): 
        """
        Args:
            x (numpy.ndarray): shape(pred_len+1, state_size)
            lam (numpy.ndarray): shape(pred_len, state_size)
            u (numpy.ndarray): shape(pred_len, input_size)
            g_xs (numpy.ndarray): shape(pred_len, state_size)
        Returns:
            lam_dot (numpy.ndarray), shape(state_size, )
        """
        if len(lam.shape) == 1:
            state_size = lam.shape[0]
            lam_dot = np.zeros(state_size)
            lam_dot[0] = x[0] - (2. * x[0] * x[1] + 1.) * lam[1]
            lam_dot[1] = x[1] + lam[0] + \
                (-3. * (x[1]**2) - x[0]**2 + 1.) * lam[1]

            return lam_dot

        elif len(lam.shape) == 2:
            pred_len, state_size = lam.shape
            lam_dot = np.zeros((pred_len, state_size))

            for i in range(pred_len):
                lam_dot[i, 0] = x[i, 0] - \
                    (2. * x[i, 0] * x[i, 1] + 1.) * lam[i, 1]
                lam_dot[i, 1] = x[i, 1] + lam[i, 0] + \
                    (-3. * (x[i, 1]**2) - x[i, 0]**2 + 1.) * lam[i, 1]

            return lam_dot

        else:
            raise NotImplementedError

    def gradient_hamiltonian_input(self,x, lam, u, g_x):
        """
        Args:
            x (numpy.ndarray): shape(pred_len+1, state_size)
            lam (numpy.ndarray): shape(pred_len, state_size)
            u (numpy.ndarray): shape(pred_len, input_size)
            g_xs (numpy.ndarray): shape(pred_len, state_size)
        Returns:
            F (numpy.ndarray), shape(pred_len, input_size)
        """
        if len(x.shape) == 1:
            input_size = u.shape[0]
            F = np.zeros(input_size)
            F[0] = u[0] + lam[1]

            return F

        elif len(x.shape) == 2:
            pred_len, input_size = u.shape
            F = np.zeros((pred_len, input_size))

            for i in range(pred_len):
                F[i, 0] = u[i, 0] + lam[i, 1]

            return F

        else:
            raise NotImplementedError
    
    def line_search(grad, sol, compute_eval_val,
                init_alpha=0.001, max_iter=100, update_ratio=1.):
        """ line search
        Args:
            grad (numpy.ndarray): gradient
            sol (numpy.ndarray): sol
            compute_eval_val (numpy.ndarray): function to compute evaluation value
        Returns: 
            alpha (float): result of line search 
        """
        assert grad.shape == sol.shape
        base_val = np.inf
        alpha = init_alpha
        original_sol = sol.copy()

        for _ in range(max_iter):
            updated_sol = original_sol - alpha * grad
            eval_val = compute_eval_val(updated_sol)

            if eval_val < base_val:
                alpha += init_alpha * update_ratio
                base_val = eval_val
            else:
                break

        return alpha