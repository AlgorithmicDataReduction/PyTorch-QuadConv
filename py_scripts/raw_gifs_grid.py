import numpy as np
import torch
import gif
import matplotlib.pyplot as plt

domain = 'ignition_center_cut'
data_dir = f'../data/{domain}/train.npy'

data = torch.from_numpy(np.float32(np.load(data_dir)))
data = data[...,[0,1]]

mean = torch.mean(data, dim=(0,1,2), keepdim=True)
stdv = torch.sqrt(torch.var(data, dim=(0,1,2), keepdim=True))
stdv = torch.max(stdv, torch.tensor(1e-3))

data = (data-mean)/stdv

data = data/(torch.max(np.abs(data))+1e-4)

@gif.frame
def plot(i, channel):
    plt.imshow(data[i,:,:, channel], vmin=-1, vmax=1, origin='lower')
    plt.colorbar(location='top')

for channel in [0,1]:
    frames = [plot(i, channel) for i in range(data.shape[0])]

gif.save(frames, f'{domain}_c{channel}.gif', duration=50)
