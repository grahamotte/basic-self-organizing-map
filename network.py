
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable
from mpl_toolkits.axes_grid import AxesGrid
from progress import ProgressBar, Percentage, Bar


def fromdistance(fn, shape, center=None, dtype=float):
    '''Construct an array by executing a function over a normalized distance.
    sqrt((x-x0)^2+(y-y0)^2) at coordinate x,y where x,y in range [-1,+1]
    '''
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i]-center[i])/float(max(1,shape[i]-1)))**2
#            d += ((args[i]-center[i])/float(shape[i]))**2
        return np.sqrt(d)/np.sqrt(len(shape))
    if center == None:
        center = np.array(list(shape))//2
    return fn(np.fromfunction(distance,shape,dtype=dtype))

def gaussian(shape,center,sigma=0.5):
    ''' Return a two-dimensional gaussian with given shape.
       Shape of the output array
       Center of gaussian
       Width of gaussian
    '''
    def g(x): return np.exp(-x**2/sigma**2)
    return fromdistance(g,shape,center)


def identity(shape,center):
    ''' Return a two-dimensional gaussian with given shape.
       Shape of the output array
       Center of gaussian
       Width of gaussian
    '''
    def identity(x): return x
    return fromdistance(identity,shape,center)



# -----------------------------------------------------------------------------
class MAP(object):
    ''' Neural Map class '''

    def __init__(self, shape=(10,10,2),
                 sigma_i = 10.00, sigma_f = 0.010,
                 lrate_i = 0.500, lrate_f = 0.005,
                 lrate = 0.1, elasticity = 2.0, init_method='random'):

        # Build map
        # Fixed initialization
        if init_method == 'fixed':
            self.adj = np.ones(shape)*0.5

        # grid initialization
        elif init_method == 'regular':
            self.adj = np.zeros(shape)
            for i in range(shape[0]):
                self.adj[i,:,0] = np.linspace(0,1,shape[1])
                self.adj[:,i,1] = np.linspace(0,1,shape[1])
                
        # Random initialization
        else:
            self.adj = np.random.random(shape)

        self.max = 0
        self.elasticity = elasticity
        self.sigma_i = sigma_i # Initial neighborhood parameter
        self.sigma_f = sigma_f # Final neighborhood parameter
        self.lrate_i = lrate_i # Initial learning rate
        self.lrate_f = lrate_f # Final learning rate
        self.lrate   = lrate   # Constant learning rate
        self.entropy    = []
        self.distortion = []


    def learn(self, samples, epochs=25000, noise=0, test_samples=None, show_progress=True):
        ''' Learn given distribution using n data
                List of sample sets
                Number of epochs to be ran for each sample set
        '''

        # Check if samples is a list
        if type(samples) not in [tuple,list]:
            samples = (samples,)
            epochs = (epochs,)

        n = 0 # total number of epochs to be ran
        
        for j in range(len(samples)):
            n += epochs[j]
        
        self.entropy = []
        self.distortion = []

        if show_progress:
            bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=n).start()
        index = 0

        for j in range(len(samples)):
            
            self.samples = samples[j]
            I = np.random.randint(0,self.samples.shape[0],n)
            
            for i in range(epochs[j]):
                # Set sigma and learning rate via time
                t = index/float(n)
                lrate = self.lrate_i*(self.lrate_f/self.lrate_i)**t
                sigma = self.sigma_i*(self.sigma_f/self.sigma_i)**t
                C = self.adj.copy()

                # Learn something
                S = self.samples[I[i]] + noise*(2*np.random.random(len(self.samples[I[i]]))-1)
                S = np.minimum(np.maximum(S,0),1)
                self.learn_data(S,lrate,sigma)

                #self.learn_data(self.samples[I[i]],lrate,sigma)
                if i%100 == 0:
                    self.entropy.append(((self.adj-C)**2).sum())

                    if test_samples is not None:
                        distortion = self.compute_distortion(test_samples)
                    else:
                        distortion = self.compute_distortion(self.samples)

                    self.distortion.append(distortion)

                if show_progress:
                    bar.update(index+1)

                index = index+1

        if show_progress: bar.finish()


    def compute_distortion(self, samples):
        distortion = 0

        for i in range(samples.shape[0]):
            data = samples[i]
            D = ((self.adj-data)**2).sum(axis=-1)
            distortion += D.min()
        
        distortion /= float(samples.shape[0])

        return distortion


    def plot(self, axes):
        ''' Plot network on given axes
        '''

        classname = self.__class__.__name__
        fig = plt.gcf()
        divider = make_axes_locatable(axes)
        axes.axis([0,1,0,1])

        # Plot samples
        axes.scatter(self.samples[:,0], self.samples[:,1], s=1, color='g', alpha=0.5)
        C = self.adj
        Cx,Cy = C[...,0], C[...,1]

        if classname != 'SSk':
        
            for i in range(C.shape[0]):
                axes.plot (Cx[i,:], Cy[i,:], 'k', alpha=1, lw=1.5)
        
            for i in range(C.shape[1]):
                axes.plot (Cx[:,i], Cy[:,i], 'k', alpha=1, lw=1.5)

        # Y = self.distortion[::1]
        # X = np.arange(len(Y))/float(len(Y)-1)
        # axes.plot(X,Y)

    def plot_dist(self, axes):
        ''' Plot network on given axes
        '''

        classname = self.__class__.__name__
        fig = plt.gcf()
        divider = make_axes_locatable(axes)
        axes.axis([0,1,0,.5])


        Y = self.distortion[::1]
        X = np.arange(len(Y))/float(len(Y)-1)
        axes.plot(X,Y)


class SOM(MAP):
    ''' Self Organizing Map class '''

    def learn_data(self, data, lrate, sigma):
        ''' Learn a single data using lrate and sigma parameter
            Learning rate
            Neighborhood width
        '''

        # Compute distances to data 
        D = ((self.adj-data)**2).sum(axis=-1)

        # Get index of nearest node (minimum distance)
        winner = np.unravel_index(np.argmin(D), D.shape)

        # Generate a gaussian centered on winner
        G = gaussian(D.shape, winner, sigma)
        G = np.nan_to_num(G)

        # Move nodes towards data according to gaussian 
        delta = (self.adj - data)
        for i in range(self.adj.shape[-1]):
            self.adj[...,i] -= lrate * G * delta[...,i]


class DSOM(MAP):
    ''' Dynamic Self Organizing Map class '''

    def learn_data(self, data, lrate=0, sigma=0):
        ''' Learn a single datum 
            Data to be learned
        '''
        # Compute distances to data 
        D = ((self.adj-data)**2).sum(axis=-1)

        # Get index of nearest node (min dist)
        winner = np.unravel_index(np.argmin(D), D.shape)

        # Dynamics
        self.max = max(D.max(), self.max)
        d = np.sqrt(D/self.max)
        sigma = self.elasticity*d[winner]

        # Generate a gaussian by winner
        G = gaussian(D.shape, winner, sigma)
        G = np.nan_to_num(G)

        # Move nodes towards data according to gaussian 
        delta = (self.adj - data)
        for i in range(self.adj.shape[-1]):
            self.adj[...,i] -= self.lrate*d*G*delta[...,i]



