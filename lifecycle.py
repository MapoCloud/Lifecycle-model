#Tring to build up an lifecycle model 

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import lognorm

#Define the model
class lifecycle():
    '''Implementation of stochastic lifecycle model with random income'''
    
    def __init__(self,Mbar=10,
                 ngrid=50,nquad=10,
                 interpolation='linear',
                 beta=0.9,R=1.05,sigma=1.,rho=0.9): 
        '''Objiect creator for stochastic lifecycle model'''
        self.beta = beta                      # Discount factor
        self.R = R                            # Gross interest rate
        self.sigma = sigma                    # Param for lognormal distribution in income shock
        self.rho = rho                        # Params for CRRA utility
        self.Mbar = Mbar                      # Upper bound for wealth 
        self.ngrid = ngrid                    # Numbers of grid pionts in state space
        self.nquad = nquad                    # Number of quadrature points
        self.interpolation = interpolation    # Type of interpolation, see below
        
    def __repr__(self):
        '''String representation for the model'''
        return 'lifecycle model with beta={:1.3f}, sigma={:1.3f}, gross return={:1.3f}, rho={:1.3f}\nGrids: state {} points up to {:1.1f}, quadrature {} points\nInterpolation: {}\nThe model is {}solved.'\
               .format(self.beta,self.sigma,self.R,self.rho,self.ngrid,self.Mbar,self.nquad,self.interpolation,'' if hasattr(self,'solution') else 'not ')
    # Property & Setting           
    @property
    def ngrid(self):
        '''Property getter for ngrid parameter'''
        return self.__ngrid
    
    @ngrid.setter
    def ngrid(self,ngrid):
        '''Property setter for the ngrid parameter'''
        self.__ngrid = ngrid
        epsilon = np.finfo(float).eps                    # smallest positive float number difference
        self.grid = np.linspace(epsilon,self.Mbar,ngrid) # grid for state space

    @property
    def sigma(self):
        '''Property getter for the sigma parameter'''
        return self.__sigma
    
    @sigma.setter
    def sigma(self,sigma):
        '''Property setter for the sigma parameter'''
        self.__sigma=sigma
        self.__quadrature_setup()
    
    @property
    def rho(self):
        '''Property getter for the rho parameter'''
        return self.__rho
    
    @rho.setter
    def rho(self,rho):
        '''Property setter for the rho parameter'''
        self.__rho = rho
        self.utility                   # Update rho if changed
        self.marginal_utility          # Update rho if changed
        self.inverse_marginal_utility  # Update rho if changed
    
    @property
    def nquad(self):
        '''Property getter for the number of quadrature points'''
        return self.__nquad
    
    @nquad.setter
    def nquad(self,nquad):
        '''Property setter for the number of quadrature points'''
        self.__nquad=nquad
        self.__quadrature_setup()
    
    # Interal funcs for solving the model 
    def __quadrature_setup(self):
        '''Internal function to set up quadrature points and weights,
        depends on sigma and nquad, therefore called from the property setters
        '''
        try:
            # Quadrature points and weights for log-normal distribution
            self.quadp,self.quadw=np.polynomial.legendre.leggauss(self.__nquad) # Get quadrature point and weight.Can be replaced with Gauss-Hermite quadrature
            self.quadp = (self.quadp+1)/2                                       # Rescale to [0,1],may be different for different qudrature methods
            self.quadp = lognorm.ppf(self.quadp,self.__sigma)                   # Inverse cdf
            self.quadw /=2                                                      # Rescale weights
        except(AttributeError):
            # When __nquad or __sigma is not yet set
            pass
    
    def utility(self,c):
        '''Utility function'''
        return c**(1-self.__rho)/(1-self.__rho)
    
    def marginal_utility(self,c):
        '''Marginal utility function'''
        return 1/(c**self.__rho)
    
    def inverse_marginal_utility(self,u):
        '''Inverse marginal utility function'''
        return 1/(u**(1/self.__rho))
    
    def next_period_wealth(self,M,c,y):
        '''Next period budget'''
        if self.nquad>1:
            return self.R*(M-c)+y                       # Next period wealth
        else :
            return self.R*(M-c)+np.zeros(shape=y.shape) # If income=0,cake-eating problem
    
    def interp_func(self,x,f):
        '''Returns the interpolation function for given data'''
        if self.interpolation=='linear':
            return interpolate.interp1d(x,f,kind='slinear',fill_value="extrapolate")
        elif self.interpolation=='quadratic':
            return interpolate.interp1d(x,f,kind='quadratic',fill_value="extrapolate")
        elif self.interpolation=='cubic':
            return interpolate.interp1d(x,f,kind='cubic',fill_value="extrapolate")
        elif self.interpolation=='polynomial':
            p = np.polynomial.polynomial.polyfit(x,f,self.ngrid_state-1)
            return lambda x: np.polynomial.polynomial.polyval(x,p)
        else:
            print('Unknown interpolation type')
            return None
    
    # Solver & Convergence check   
    def solve_egm(self,maxiter=500,tol=1e-4,callback=None):
        '''Solves the model using EGM (successive approximations of efficient Coleman-Reffet operator)
           Callback function is invoked at each iteration with keyword arguments.
        '''
        A=np.linspace(0,self.Mbar,self.ngrid) # Grids of cash in hands
        # Initial policy func and value func setup
        interp= lambda x,f:interpolate.interp1d(x,f,kind='slinear',fill_value="extrapolate")
        c0=interp([0,self.Mbar],[0,self.Mbar]) # initial policy func,a 45 degree line of c & M
        V0=self.utility(self.grid) # initial value func, just utility func
        for iter in range(maxiter):
            # EGM steps
            M1=self.next_period_wealth(A[:,np.newaxis],0,self.quadp[np.newaxis,:]) # matrix with A in axis=0 y in axis=1
            c1=np.maximum(c0(M1),self.grid[0]) # consumption in next period on the right hand side of the Euler equation:Why compair to grid[0]? Cuz borrowing restriction
            mu=self.marginal_utility(c1) # marginal utility of RHS,next we should compute the expectation
            RHS=self.beta*self.R*np.dot(mu,self.quadw) # Right hand site of Euler equation 
            # vertorize c & M
            c=np.empty(self.ngrid+1,dtype=float) 
            M=np.empty(self.ngrid+1,dtype=float)
            M[0]=c[0]=0
            c[1:]=self.inverse_marginal_utility(RHS) # Current period consumption
            M[1:]=A+c[1:]                            # endogous point of M
            c1=interp(M,c)                           # Updated policy func
            c1grid=c1(self.grid)                     # vector represent policy func
            # Now tring to get the value func
            M1=self.next_period_wealth(self.grid[:,np.newaxis],c1grid[:,np.newaxis],self.quadp[np.newaxis,:]) # if c1 is the optimal consumption, 
            # bellman equation will be satisfied ,given an initial value func will get the right value func
            interfunc=self.interp_func(self.grid,V0)
            V=interfunc(M1)
            EV=np.dot(V,self.quadw)
            V1=self.utility(c1grid)+self.beta*EV
            err=np.amax(np.abs(c1grid-c0(self.grid)))
            if callback: callback(iter=iter,model=self,value=V1,policy=c1grid,err=err) # callback for making plots
            if err<tol:
                break # converged
            c0,V0=c1,V1 #set c1 V1 as new initial func in next iteration
        else:
            raise RuntimeError('No convergence: maximum number of iterations achieved!')
        self.solution={'value':V1,'policy':c1grid,'solver':'egm'}  # save the model solution to the object
        return V1,c1grid
    
    def solve_plot(self,**kvarg):
        '''Illustrate solution
           Inputs: solver (string), and any inputs to the solver
        '''
        fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(14,8))
        ax1.grid(b=True, which='both', color='0.65', linestyle='-')
        ax2.grid(b=True, which='both', color='0.65', linestyle='-')
        ax1.set_title('Value function convergence with EGM')
        ax2.set_title('Policy function convergence with EGM')
        ax1.set_xlabel('Wealth, M')
        ax2.set_xlabel('Wealth, M')
        ax1.set_ylabel('Value function')
        ax2.set_ylabel('Policy function')
        def callback(**kwargs):
            print('|',end='')
            grid = kwargs['model'].grid
            v = kwargs['value']
            c = kwargs['policy']
            ax1.plot(grid[1:],v[1:],color='k',alpha=0.25)
            ax2.plot(grid,c,color='k',alpha=0.25)
        V,c = self.solve_egm(callback=callback,**kvarg)
        # add solutions
        ax1.plot(self.grid[1:],V[1:],color='r',linewidth=2.5)
        ax2.plot(self.grid,c,color='r',linewidth=2.5)
        plt.show()
        return V,c
    
    # Accuracy check
    def euler_residual(self,c,M,policy):
        '''Computes the Euler residuals for a given points (M,c), and
           given policy function that enters into the RHS
           Argument policy is interpolation function for the policy
        '''
        if isinstance(c,np.ndarray):
            c0,M0=c[:,np.newaxis],M[:,np.newaxis]
            y=self.quadp[np.newaxis,:]
        else:
            c0,M0=c,M
            y=self.quadp
        M1=self.next_period_wealth(M0,c0,y)
        c1=np.maximum(policy(M1),self.grid[0])
        mu=self.marginal_utility(c1)
        RHS=self.beta*self.R*np.dot(mu,self.quadw)
        LHS=self.marginal_utility(c)
        return LHS-RHS
    
    def accuracy(self,dense_grid_factor=10,verbose=False):
        '''Compute the average squared Euler residuals for the saved solution'''
        assert hasattr(self,'solution'), 'Need to solve the model to compute the accuracy measure!'
        grid = np.linspace(self.grid[0],self.Mbar,self.ngrid*dense_grid_factor) # dense grid for state space
        inter = self.interp_func(self.grid,self.solution['policy'])  # interpolation function for policy function
        c = inter(grid)  # consumption on the dense grid
        er = self.euler_residual(c=c,M=grid,policy=inter)
        er = er[np.logical_not(np.isclose(c,grid,atol=1e-10))]  # disregard corner solutions
        acc = np.mean(er**2)
        if verbose:
            print('Average squared Euler residuals ({}) using {} points is {}'.format(
                self.solution['solver'],self.ngrid*dense_grid_factor,acc))
        else:
            return acc
    
    # Simulation (need to rework)
    def simulator(self,init_wealth=1,T=10,seed=None,plot=True):
        '''Simulation of the model for given number of periods from given initial conditions'''
        assert hasattr(self,'solution'), 'Need to solve the model before simulating!'
        if seed!=None:
            np.random.seed(seed)  # fix the seed if needed
        init_wealth = np.asarray(init_wealth).ravel()  # flat np array of initial wealth
        N = init_wealth.size  # number of trajectories to simulate
        sim = {'M':np.empty((N,T+1)),'c':np.empty((N,T+1))}
        sim['M'][:,0] = init_wealth  # initial wealth in the first column
        inter = self.interp_func(self.grid,self.solution['policy'])  # interpolation function for policy function
        for t in range(T+1):
            sim['c'][:,t] = inter(sim['M'][:,t])  # optimal consumption in period t
            if t<T:
                y = lognorm.rvs(self.sigma,size=N) # draw random income
                sim['M'][:,t+1] = self.next_period_wealth(sim['M'][:,t],sim['c'][:,t],y) # next period wealth
        if plot:
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,6))
            ax1.set_title('Simulated wealth and consumption trajectories')
            ax1.set_ylabel('Wealth')
            ax2.set_ylabel('Consumption')
            ax2.set_xlabel('Time period in the simulation')
            for ax in (ax1,ax2):
                ax.grid(b=True, which='both', color='0.95', linestyle='-')
            for i in range(N):
                ax1.plot(sim['M'][i,:],alpha=0.75)
                ax2.plot(sim['c'][i,:],alpha=0.75)
            plt.show()
        return sim # return simulated data

# model example    
m=lifecycle(ngrid=100,sigma=.5,nquad=10,rho=0.95)
m
v,c = m.solve_plot()
m.accuracy(verbose=True)
sims = m.simulator(init_wealth=m.Mbar*np.arange(15)/15,T=25,seed=2022)
            
            
            
            
            
        
    
    
            
            
    
