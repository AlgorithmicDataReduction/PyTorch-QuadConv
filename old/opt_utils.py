import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import imageio

from torch.utils.data import DataLoader

import platform
system_id = platform.system()



def sobolev_loss(pred, x, diff_order=1, lambda_r = (0.25, 0.0625)):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bs = pred.shape[0]

    sq_shape = np.sqrt(x.shape[2]).astype(int)

    numel = sq_shape * sq_shape

    temp_x = torch.reshape(x,(x.shape[0],x.shape[1],sq_shape,sq_shape))
    temp_pred = torch.reshape(pred,(pred.shape[0],pred.shape[1],sq_shape,sq_shape))

    #loss = torch.sum( torch.max( ((temp_pred - temp_x)**2) - 1e-2, torch.zeros(1).to(device) ) )

    loss = torch.sum(  (temp_pred - temp_x)**2 )

    stencil = np.array([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]]) * 1/4
    stencil = torch.FloatTensor(stencil).to(device)
    stencil = torch.reshape(stencil,(1,1,3,3)).repeat(1,x.shape[1],1,1)

    for i in range(diff_order):

        temp_x = torch.nn.functional.conv2d(temp_x,stencil)

        temp_pred = torch.nn.functional.conv2d(temp_pred,stencil)

        #loss += lambda_r[i] * torch.sum( torch.max( ((temp_pred - temp_x)**2) - 1e-2, torch.zeros(1).to(device) ) )

        loss += lambda_r[i] * torch.sum(  (temp_pred - temp_x)**2 )

    return loss / bs

def test(dataloader, model):

    mse_loss = torch.nn.MSELoss(reduction='none')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.eval()
    test_loss, this_loss, max_loss = 0 , 0 , 0

    #idx_max, idx_curr = 0 , 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X = data
            #pts = data[1]

            X = X.to(device)
            #pts = pts.to(device)

            pred, compressed = model(X)
            this_loss = torch.mean( mse_loss(pred, X) / mse_loss(X,torch.zeros_like(X)) )
            test_loss += this_loss
            max_loss = max(max_loss, this_loss)

            #if max_loss == this_loss:
            #    idx_max = idx_curr

            #idx_curr += 1

    test_loss /= size
    print(f"Test Error: \n Avg Relative Error: {test_loss:>8f} \n Max Relative Error: {max_loss:>8f}")
    return test_loss

def train(dataloader, model, loss_fn, optimizer, EPSILON = 0):

    size = len(dataloader.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    cumul_loss = 0

    for batch, data in enumerate(dataloader):
        X = data
        #pts = data[1]

        X = X.to(device)
        #pts = pts.to(device)

        X_t = X + EPSILON * torch.randn_like(X[0,:])

        # Compute prediction error
        pred, compressed = model(X_t)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumul_loss += loss.item()

        if (batch) % 10 == 0:
            bs = X.shape[0]
            numel = X.shape[-1]
            loss, current = loss.item(), (batch+1) * len(X)
            loss /= (numel)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return cumul_loss / (batch + 1)

def load_ignition_data(batch_size=16, time_chunk = 1, order = None, split = 0.8, size=25, stride=25, noise=False, normalize = True, dataloader = True, center_cut = False, tile = 1, use_all_channels = False):

    if system_id == 'Windows':
        data_path = 'C:/Users/Kevin/Google Drive/Research/CAE/data/Ignition/default.npy'
    elif system_id == 'Darwin':
        data_path = '/Users/kdoh/Google Drive/My Drive/Research/CAE/data/Ignition/default.npy'

    if use_all_channels:
        idx =  (0,1,2)
    else:
        idx = (0)


    mydata = np.load(data_path)
    ignition_data = np.float32(mydata[:,74:174,0:100,idx])

    if center_cut:
        ignition_data = np.float32(mydata[:,99:149,0:50,idx])

    if noise:
        ignition_data += 0.0001*np.random.randn(*ignition_data.shape)

    ignition_data = torch.from_numpy(ignition_data)

    if use_all_channels:
        ignition_data = torch.movedim(ignition_data, -1, 0)
        ignition_data = ignition_data.reshape(-1, ignition_data.shape[-2], ignition_data.shape[-1] )

    ignition_data = ignition_data.unfold(1,size,stride).unfold(2, size, stride)

    ignition_data = ignition_data.reshape(-1,size,size).reshape(-1,1,size*size)

    if normalize:

        mean_ignition =  torch.mean(ignition_data, dim=(0,1,2), keepdim=True)
        stddev_ignition = torch.sqrt(torch.var(ignition_data, dim=(0,1,2), keepdim=True))
        stddev_ignition = torch.max(stddev_ignition, torch.tensor(1e-3))

        ignition_data = (ignition_data - mean_ignition) / stddev_ignition

        max_val = torch.max(torch.abs(ignition_data))

        ignition_data = ignition_data / (max_val + 1e-4)

        #ignition_data /= torch.linalg.vector_norm(ignition_data, dim = (1,2), keepdim=True)


    if order == 'random':
        np.random.shuffle(ignition_data)

    s = ignition_data.shape

    #ignition_data = ignition_data.unsqueeze(-1)

    split_num = int(np.floor(split * s[0]))

    cutoff = int(np.floor((split_num/time_chunk)))


    if dataloader:
        train_dl = DataLoader(ignition_data[0:cutoff,:,:],batch_size=batch_size,shuffle=True)

        if cutoff+1 >= s[0]:
            test_dl = None
        else:
            test_dl = DataLoader(ignition_data[cutoff+1::,:,:],batch_size=batch_size,shuffle=True)

    else:
        train_dl = ignition_data[0:cutoff,:,:]
        test_dl = ignition_data[cutoff+1::,:,:]

    return train_dl, test_dl, s[1::]



def make_gif(model,save_path, data_inputs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    size = data_inputs['size']
    tile = data_inputs['tile']

    data_inputs['use_all_channels'] = False

    ignition_data_rs , test_dl , s = load_ignition_data(dataloader = False, **data_inputs)

    ignition_data_rs = ignition_data_rs.to(device)

    with torch.no_grad():
        processed, compressed = model(ignition_data_rs)

    processed_squares = processed.reshape(-1,size,size).reshape(-1,tile,tile,size,size)

    processed_full = torch.zeros(450,size*tile,size*tile)

    for i in range(450):
        for j in range(tile):
            for k in range(tile):
                processed_full[i, size*j:size*(j+1), size*k:size*(k+1)] = processed_squares[i,j,k,:,:]



    filenames = []
    fig1 , ax1 = plt.subplots()

    with torch.no_grad():
        for i in range(450):


            ax1.imshow(processed_full[i,:,:], vmin = -1, vmax = 1)
            filename = f'{i}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.cla()


        plt.close('all')
        image_list = []
        for filename in filenames:
            image = imageio.imread(filename)
            image_list.append(image)

        imageio.mimwrite(os.path.join(save_path , 'processed_quadconv_new.gif'), image_list)

        for filename in set(filenames):
            os.remove(filename)

    return 
