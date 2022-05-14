import torch 
from AutoConvPoint_refactor import AutoQuadConv, QuadConvLayer, QuadConvBlock
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from torch.profiler import profile, record_function, ProfilerActivity

g_size = 25

f = torch.randn(1,g_size,g_size)

f[0,5:30,40:50] += 20

f = f.reshape(1,1,-1)

#f[1:1000,5:20,5:20] += 2

#f[1000:2000,80:90,5:30] += -1/2

#f[2000:3000,1:25,25:50] += 1/2

#f[3000:4000,80:95,80:95] += 1

#f = f.reshape(4000,1,-1)

#f = f / torch.linalg.norm(f,dim=2, keepdim=True)

#f =  torch.zeros(1,1,g_size*g_size)

#f[:,0,int(g_size/2)*g_size+int(g_size/2)] = 1
#f[:,0,int(g_size/4)*g_size+int(g_size/4)] = 1

#f = torch.from_numpy(np.tile(np.sin(4*np.linspace(-pi, pi, 100)), (1,1))).reshape(1,1,-1).float()

locs = torch.linspace(-1,1,g_size)

x,y = torch.meshgrid( locs, locs, indexing ='xy')

gx = x.reshape(-1,1)
gy = y.reshape(-1,1)

grid = torch.cat([gx,gy], -1)

#quadconv = QuadConvLayer(dimension = 2, channels_in = 1 ,channels_out = 1, kernel_mode='sinc', quad_type='newton_cotes')
#quadconv.set_quad(g_size)
#quadconv.set_output_locs(g_size)

#quadconv = QuadConvBlock(dimension= 2, channels_in=1, channels_out=1, N_in=g_size, N_out=g_size, quad_type='newton_cotes')

model_inputs = {'feature_dim' : (1,1,g_size*g_size),
                    'dimension' : 2, 
                    'compressed_dim' : 10, 
                    'point_seq' : (g_size, 20, 10),
                    'channel_sequence' : [1, 8, 32],    
                    'final_activation' : torch.nn.Tanh(),
                    'mlp_chan' : (4, 4, 4),
                    'quad_type' : 'newton_cotes',
                    'use_bias' : False, 
                    'mlp_mode' : 'single', 
                    'adjoint_activation' : torch.nn.Tanh(),
                    'forward_activation' : torch.nn.Tanh(),
                    'noise_eps' : 0.001
                    }

quadconv =  AutoQuadConv(**model_inputs)

#resamp_to_quad = QuadInterpModule(g_size)
#f = resamp_to_quad(gx,gy,f)



if True:

    with torch.no_grad():
        #int_f = quadconv(f, grid)

        int_f =  quadconv(f)


    if False:
        fig1, ax1 = plt.subplots()
        ax1.plot(locs,int_f[0,0,:])
        ax1.plot(locs,f[0,0,:])

        fig2, ax2 = plt.subplots()
        ax2.plot(locs, quadconv.weight_func(locs.unsqueeze(-1)))

        plt.show()

    if False:

        torch.save(f, './2D Test Data/Round1/100train_original.pt')
        torch.save(int_f, './2D Test Data/Round1/100train_lowpass.pt')

    if True:
        with torch.no_grad():
            fig1, ax1 = plt.subplots()
            ax1.pcolormesh(x, y, int_f[0,0,:].reshape(g_size,g_size))

            #fig15, ax15 = plt.subplots()
            #ax15.pcolormesh(x, y, int_f[0,1,:].reshape(g_size,g_size))

            fig2, ax2 = plt.subplots()
            ax2.pcolormesh(x, y, f[0,0,:].reshape(g_size,g_size))

            #fig3, ax3 = plt.subplots()
            #ax3.pcolormesh(x,y, quadconv.weight_func[0](grid).reshape(g_size,g_size))

            plt.show()
        

    print('forward pass complete')




