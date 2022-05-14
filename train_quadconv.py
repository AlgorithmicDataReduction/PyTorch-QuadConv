import torch 
from torch.utils.data import DataLoader
from AutoConvPoint_refactor import QuadConvLayer, QuadInterpModule, AutoQuadConv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torch.profiler

torch.set_default_dtype(torch.float)

if False:
    prof =  torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard/profile/quadconv'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)


# TRAIN DATA
device = "cuda" if torch.cuda.is_available() else "cpu"

f = torch.load('./2D Test Data/Round1/btrain_original.pt')
g_size = int(np.sqrt(f.shape[2]))

locs = torch.linspace(-1,1,g_size)

x,y = torch.meshgrid( locs, locs, indexing ='xy')

gx = x.reshape(-1,1)
gy = y.reshape(-1,1)

grid = torch.cat([gx,gy], -1)

#to_quad_nodes = QuadInterpModule(g_size)
#f_interp = to_quad_nodes(gx,gy,f)

low_pass_training = torch.load('./2D Test Data/Round1/btrain_lowpass.pt')
#lp_interp = to_quad_nodes(gx,gy,low_pass_training)

#to_grid_nodes = QuadInterpModule(g_size, mode='grid')

ds_train = []
for i in range(f.shape[0]):
    #ds_train.append([f_interp[i,:,:], lp_interp[i,:,:]])
    ds_train.append([low_pass_training[i,:,:]])

dl = DataLoader(ds_train, batch_size=16, shuffle=True)

# TEST DATA

if False:
    f_test = torch.load('./2D Test Data/Round1/test_original.pt')
    low_pass_training_test = torch.load('./2D Test Data/Round1/test_lowpass.pt')

    ds_test = []

    for i in range(f_test.shape[0]):
        ds_test.append([f_test[i,:,:], low_pass_training_test[i,:,:]])

    dl_test= DataLoader(ds_test, batch_size=16, shuffle=True)


#quadconv = QuadConvLayer(dimension = 2, channels_out = 1, weight_mlp_chan=(4,8,4), kernel_mode='MLP')

#quadconv.set_quad(g_size)

quadconv =  AutoQuadConv(
        feature_dim = (16, 1, g_size*g_size),
        dimension = 2,
        compressed_dim = 10,
        point_seq = (25, 10, 5),
        channel_sequence = [1, 8, 16],
        final_activation = nn.Identity()
        )

#mesh_nodes, _ =  quadconv.get_quad_mesh()
#mesh_nodes =  mesh_nodes.to(device)

epochs = 5

size = len(dl.dataset)

optimizer = torch.optim.Adam(quadconv.parameters(), lr = 1e-2)

quadconv = quadconv.to(device)
#grid = grid.to(device)

cumul_loss = 0

for t in range(epochs):

    print('------Epoch %d------' % (t))
    size = len(dl.dataset)

    for batch, data_element in enumerate(dl):


        X = data_element[0].to(device)
        #Y = data_element[1].to(device)

        if False and batch == 100 and t == 0:
            fig1, axs = plt.subplots(1,2)
            axs[0].pcolormesh(x,y,X[1,0,:].reshape(25,25))
            axs[1].pcolormesh(x,y,Y[1,0,:].reshape(25,25))
            plt.show()


        # Compute prediction error
        #pred = quadconv(X,mesh_nodes)
        pred, compressed = quadconv(X)
        loss = torch.nn.functional.mse_loss(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumul_loss += loss.item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch*16
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if False:
    with torch.no_grad():

            test_loss = 0
            size = len(dl_test.dataset)

            for batch, data_element in enumerate(dl_test):

                X = data_element[0].to(device)
                Y = data_element[1].to(device)

                # Compute prediction error
                pred = quadconv(X,grid)
                test_loss += torch.nn.functional.mse_loss(pred, Y)

            #print('Final test loss %1.4f' % test_loss.numpy()/size)

# plot weight function
if False:
    with torch.no_grad():

        olocs = torch.linspace(-1,1,g_size*4)

        ox,oy = torch.meshgrid( olocs, olocs, indexing ='xy')

        oxg = ox.reshape(-1,1)
        oyg = oy.reshape(-1,1)

        ogrid = torch.cat([oxg,oyg], -1).to(device)

        fig1, ax1 = plt.subplots()
        ax1.pcolormesh(ox,oy, quadconv.kernel_func(ogrid).reshape(g_size*4,g_size*4).detach().cpu(), vmin=-5, vmax=10)

    plt.show()

    fig1.savefig('Learned Weight Function Interp')


print('done')