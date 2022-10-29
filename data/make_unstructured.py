import numpy as np
from scipy.interpolate import interpn as interpolate
import gif
import matplotlib.pyplot as plt

if __name__ == '__main__':

    domain = 'grid_ignition_center_cut'
    data_dir = f'data/{domain}/train.npy'

    data = np.load(data_dir)
    data = np.transpose(data, (0,2,1,3))

    time_steps = data.shape[0]
    x_n = data.shape[1]
    y_n = data.shape[2]
    channels = data.shape[3]

    #convert to grid
    x_lim = 50
    y_lim = 50

    points = (np.linspace(0, x_lim, x_n), np.linspace(0, y_lim, y_n))

    #determine sample points
    num_points = x_n*y_n

################################################################################

    sample_points = np.zeros((num_points, 2))

    # sample_points[:,0] = np.random.uniform(0, x_lim, num_points)
    # sample_points[:,1] = y_lim*np.random.beta(3, 3, size=num_points)

    x = np.arange(0.5, x_lim+0.5, 1)
    y = np.arange(0.5, y_lim+0.5, 1)

    for i in range(x_n):
        for j in range(y_n):
            sample_points[j+y_n*i,0] = x[i]
            sample_points[j+y_n*i,1] = y[j]

    sample_points[:,0] += np.random.uniform(-0.45, 0.45, num_points)
    sample_points[:,1] += np.random.uniform(-0.45, 0.45, num_points)

################################################################################

    #interpolate
    sample_features = np.zeros((time_steps, num_points, channels))

    for t in range(time_steps):
        sample_features[t,...] = interpolate(points, data[t,...], sample_points)

    #visualize
    plt.scatter(sample_points[:,0], sample_points[:,1], s=5, c=sample_features[10,:,1])
    plt.show()

    plt.hist2d(sample_points[:,0], sample_points[:,1], density=False, bins=50, weights=sample_features[10,:,1], vmin=-1, vmax=1)
    plt.show()

    #save data
    np.save('data/unstructured_ignition_center_cut_2/points', sample_points)
    np.save('data/unstructured_ignition_center_cut_2/features', sample_features)
