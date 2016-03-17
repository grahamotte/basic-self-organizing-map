
if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from network import SOM, DSOM
    from distribution import uniform, normal, ring

    size = 8
    epochs = 10000
    n = 10000

    np.random.seed(12345)

    samples_1 = uniform(n=n, center=(0.25,0.25), scale=(0.2,0.2))
    samples_2 = uniform(n=n, center=(0.25,0.75), scale=(0.2,0.2))
    samples_3 = uniform(n=n, center=(0.75,0.25), scale=(0.2,0.2))
    samples_4 = uniform(n=n, center=(0.75,0.75), scale=(0.2,0.2))

    mag = 1000
    s1 = np.array([0,0])
    for x in range(0,mag):
        v = x/float(mag)
        b = np.array([v, np.sin(7*v)/5 + np.random.random_sample()/3+ 0.3])
        s1 = np.row_stack((s1,b))

    s2 = np.array([0,0])
    for x in range(0,mag):
        v = x/float(mag)
        b = np.array([v, np.sin(7*(v+.1))/5 + np.random.random_sample()/3+ 0.3])
        s2 = np.row_stack((s2,b))

    s3 = np.array([0,0])
    for x in range(0,mag):
        v = x/float(mag)
        b = np.array([v+.2, np.sin(7*(v+.2))/5 + np.random.random_sample()/3+ 0.3])
        s3 = np.row_stack((s3,b))

    samps = [s1,s2,s3]
    n = len(samps)
    epcs = [epochs//n for x in range(n)]

    print 'Self-Organizing Map'
    np.random.seed(12345)
    som = SOM((size,size,2))
    som.learn(samps,epcs)

    print 'Dynamic Self-Organizing Map'
    np.random.seed(12345)
    dsom = DSOM((size,size,2), elasticity=2)
    dsom.learn(samps,epcs)


    fig = plt.figure(figsize=(16,7))
    fig.patch.set_alpha(1.0)

    axes = plt.subplot(1,2,1)
    som.plot(axes)
    axes = plt.subplot(1,2,2)
    dsom.plot(axes)
    fig.savefig('dynamicA.png',dpi=150)
    fig.clf()

    axes = plt.subplot(1,2,1)
    som.plot_dist(axes)
    axes = plt.subplot(1,2,2)
    dsom.plot_dist(axes)
    fig.savefig('dynamicB.png',dpi=150)

    #plt.show()
    
