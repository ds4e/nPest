import numpy as np
import matplotlib.pyplot as plt


class ecdf:
    """
    Empirical cumulative distribution function.

    Given a sample x, computes the proportion of observations
    less than or equal to each observed value.
    """
        
    def __init__(self, x):
        """
        Create an empirical CDF from a sample.

        Parameters
        ----------
        x : array-like
            Observed values.
        """
        self.name = getattr(x, 'name', None)
        self.x = np.array(x)
        self.n = len(self.x)
        self.grid = np.sort( np.unique(self.x) )
        self.I = ( self.x.reshape(1,-1) <= self.grid.reshape(-1,1) ).astype(int)
        self.F_hat = self.I.mean(axis=1)

    def plot(self, fill = False ):
        """
        Plot the empirical cumulative distribution function.
        """        
        plt.step(self.grid, self.F_hat, where='post')
        if fill:
            plt.fill_between( self.grid, self.F_hat, step='post')
        plt.title('Empirical CDF')
        plt.ylabel('Proportion')
        if self.name is not None:
            plt.xlabel(self.name)
        #plt.show()

    def quantile(self, u_value):
        """
        Return the empirical quantile associated with a proportion.

        Parameters
        ----------
        u_value : float
            A proportion between 0 and 1.

        Returns
        -------
        float
            The smallest observed value whose empirical CDF is
            greater than or equal to u_value.
        """        
        if u_value <= 0 or u_value > 1:
            raise ValueError('u_value must be between 0 and 1.')
        return self.grid[self.F_hat >= u_value][0]

    def prop(self, x_value):
        """
        Return the empirical proportion at a given value.

        Parameters
        ----------
        x_value : float
            Value at which to evaluate the empirical CDF.

        Returns
        -------
        float
            Proportion of observations less than or equal to x_value.
        """

        return (self.x <= x_value).mean()

    def __repr__(self):
        return f"ecdf(n={self.n})"
    


class kde:
    """
    Kernel density estimator.

    Estimates a probability density from observed data using
    either a uniform or Gaussian kernel.
    """

    def __init__(self, x, 
        kernel_type = 'uniform', 
        graph_steps = 100):
        """
        Create a kernel density estimator.

        Parameters
        ----------
        x : array-like
            Observed values.

        kernel_type : str, default='uniform'
            Kernel used to construct the density estimate.
            Options are 'uniform' and 'gaussian'.

        h : float or None, default=None
            Bandwidth of the kernel. If None, a rule-of-thumb
            bandwidth is computed when fit() is called.

        graph_steps : int, default=100
            Number of points used to evaluate and plot the density.
        """
        self.name = getattr(x, 'name', None)
        self.kernel_type = kernel_type
        self.h = None
        self.x = np.array(x)
        self.n = len(self.x)
        self.x_sd = self.x.std()
        self.graph_steps = graph_steps
        self.grid = None

    def fit(self, h=None):
        """
        Fit the kernel density estimator.

        If no bandwidth is supplied, a rule-of-thumb bandwidth
        is computed from the sample standard deviation and
        sample size.

        Parameters
        ----------
        h : float or None, default=None
            Bandwidth to use for the density estimate.

        Returns
        -------
        None
            Stores the evaluation grid and estimated density
            in self.grid and self.f_hat.
        """

        ## Compute bandwidth
        if h is None:
            if self.kernel_type == 'uniform':
                self.h = 1.84 * self.x_sd * self.n ** (-0.2)
            elif self.kernel_type == 'gaussian':
                self.h = 1.06 * self.x_sd * self.n ** (-0.2)
        else:
            self.h = h
        self.grid = np.linspace( self.x.min()-2*self.h, 
                                     self.x.max()+2*self.h, 
                                     self.graph_steps)
        
        ## Compute density estimate
        diff = self.grid.reshape(-1,1) - self.x.reshape(1,-1) 
        if self.kernel_type == 'uniform':
            I = np.abs( diff ) < self.h
            self.f_hat = np.mean(I,axis=1)/(2 * self.h)
        elif self.kernel_type == 'gaussian':
            K = np.exp( -(diff/self.h)** 2/2 )/np.sqrt( 2*np.pi )
            self.f_hat = np.mean(K,axis=1)/self.h

    def plot(self, fill = False):
        """
        Plot the fitted kernel density estimate.

        The model must be fitted before calling this method.
        """        
        if self.grid is None:
            print('Model not yet fitted.')
        else:
            if self.kernel_type == 'uniform':
                plt.step(self.grid, self.f_hat, where='mid')
                if fill:
                    plt.fill_between(self.grid,self.f_hat,step='mid')               
            elif self.kernel_type == 'gaussian':
                plt.plot(self.grid, self.f_hat)
                if fill:
                    plt.fill_between(self.grid, self.f_hat)

            plt.title('Kernel Density Estimator')
            plt.ylabel('Density')                
            if self.name is not None:
                        plt.xlabel(self.name)
            #plt.show()

    def __repr__(self):
        return f"kde(kernel_type='{self.kernel_type}', h={self.h})"


