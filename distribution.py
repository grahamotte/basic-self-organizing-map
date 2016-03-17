
import numpy as np
from PIL import Image


def normal(center=(0.5,0.5), scale=(0.125,0.125), bounds=(0,1,0,1), n=1):
    ''' Return n points drawn from a 2 dimensional normal distribution.
        Center of disribution
        Scale of distribution
        Bounds of distribution as (xmin,xmax,ymin,ymax)
        Number of sample to generate
    '''
    Z = np.zeros((n,2))
    Z[:,0] = np.random.normal(center[0], scale[0], n)
    Z[:,0] = np.maximum(np.minimum(Z[:,0],bounds[1]),bounds[0])
    Z[:,1] = np.random.normal(center[1], scale[1], n)
    Z[:,1] = np.maximum(np.minimum(Z[:,1],bounds[3]),bounds[2])
    return Z

def uniform(center=(0.5,0.5), scale=(0.5,0.5), n=1):
    ''' Return n points drawn from a 2 dimensional normal distribution.
        Center of disribution
        Scale of distribution
        Number of sample to generate
    '''
    Z = np.zeros((n,2))
    Z[:,0] = np.random.uniform(center[0]-scale[0],center[0]+scale[0],n)
    Z[:,1] = np.random.uniform(center[1]-scale[1],center[1]+scale[1],n)
    return Z


def ring(center=(0.5,0.5), radius=(0.0,0.5), n=1):
    ''' Return n points drawn from a 2 dimensional normal distribution.
        Center of disribution
        Inner/Outer radius
        Number of sample to generate
    '''
    Z = np.zeros((n,2))
    rmin,rmax = radius
    xc,yc = center
    for i in range(Z.shape[0]):
        r = -1
        while r < rmin or r > rmax:
            x,y = np.random.random(), np.random.random()
            r = np.sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc))
        Z[i,:] = x,y
    return Z


def image(filename, shape=(8,8), n=1):
    image = np.array(Image.open(filename),dtype=float)/256.0
    Z = np.zeros((n,shape[0]*shape[1]))
    x = np.random.randint(0,image.shape[0]-shape[0]-1,n)
    y = np.random.randint(0,image.shape[1]-shape[1]-1,n)
    for i in range(n):
        Z[i] = image[x[i]:x[i]+shape[0],y[i]:y[i]+shape[1]].flatten()
    return Z
