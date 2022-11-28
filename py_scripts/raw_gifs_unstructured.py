import numpy as np
import torch
import gif
import matplotlib.pyplot as plt

domain = 'unstructured_ignition_center_cut'
data_dir = f'data/{domain}/'

points = torch.from_numpy(np.float32(np.load(data_dir+'points.npy')))
features = torch.from_numpy(np.float32(np.load(data_dir+'features.npy')))[...,1]

#TODO: update
mean = torch.mean(features, dim=(0,1), keepdim=True)
stdv = torch.sqrt(torch.var(features, dim=(0,1), keepdim=True))
stdv = torch.max(stdv, torch.tensor(1e-3))

features = (features-mean)/stdv

features = features/(torch.max(torch.abs(features))+1e-4)

points = points.numpy()
features = features.numpy()

@gif.frame
def plot(i):
    # plt.scatter(points[:,0], points[:,1], s=5, c=features[i,:], vmin=-1, vmax=1)
    plt.hist2d(points[:,0], points[:,1], bins=50, weights=features[i,:], vmin=-1, vmax=1)
    plt.colorbar(location='top')

frames = [plot(i) for i in range(features.shape[0])]

gif.save(frames, f'{domain}.gif', duration=50)
