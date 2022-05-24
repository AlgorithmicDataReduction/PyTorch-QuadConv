from copy import deepcopy
import sys
import torch
import os

import numpy as np
from functools import partial

from datetime import datetime
now = datetime.now()


from torch.utils.tensorboard import SummaryWriter

import platform
system_id = platform.system()


if system_id == 'Windows':
    sys.path.append('C:\\Users\\kevin\\Google Drive\\Research\\CAE\\AutoPoint')
    sys.path.append('C:\\Users\\kevin\\Google Drive\\Research\\CAE\\data\\Ignition')
    sys.path.append('.\\utils')

    traj_dirs = ['C:\\Users\\Kevin\\Documents\\Case0 Extra Data']
    home_dir = 'C:\\Users\\kevin\\Google Drive\\Research\\CAE\\AutoPoint'
elif system_id == 'Darwin':
    sys.path.append('/Users/kdoh/Google Drive/My Drive/Research/CAE/AutoPoint/utils')
    sys.path.append('/Users/kdoh/Google Drive/My Drive/Research/CAE/data/Ignition')

    home_dir = '/Users/kdoh/Google Drive/My Drive/Research/CAE/AutoPoint'
    traj_dirs = ['/Users/kdoh/Desktop/Case0 Extra Data']

from AutoConvPoint_refactor import AutoQuadConv
from opt_utils import train, test, sobolev_loss, load_ignition_data, make_gif

#torch.set_default_dtype(torch.float)

import torch.profiler
if False:
    prof =  torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1),
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard/profile/quadconv/fix_device'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)


def main(parser):
    batch_size = 128
    lambda_r = [0.25, 0.0625]
    diff_order = 1

    args = parser.parse_args()
    experiment_name = 'quadconv_EOS_' + now.strftime("%m%d%Y_%H%M%S")

    #train_dl = train_dataloader(traj_dirs,batch_size=batch_size)

    data_inputs = {'batch_size' : batch_size,
                    'split' : 1,
                    'order' : 'none',
                    'noise' : False,
                    'normalize' : True,
                    'size' : 25,
                    'stride' : 25,
                    'center_cut' : False,
                    'tile' : 4,
                    'use_all_channels' : False
                    }

    other_inputs = {'train_eps': 0.001
                    }

    train_dl , test_dl , _ = load_ignition_data(**data_inputs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = os.path.join("./Tensorboard/runs/", experiment_name)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    model_inputs = {'feature_dim' : (batch_size,1,data_inputs['size']*data_inputs['size']),
                    'dimension' : 2,
                    'compressed_dim' : 25,
                    'point_seq' : (data_inputs['size'], 25, 20, 15, 10),
                    'channel_sequence' : [1, 4, 8, 16, 32],
                    'final_activation' : torch.nn.Tanh(),
                    'mlp_chan' : [8]*4,
                    'quad_type' : 'newton_cotes',
                    'use_bias' : False,
                    'mlp_mode' : 'single',
                    'adjoint_activation' : torch.nn.Tanh(),
                    'forward_activation' : torch.nn.Tanh(),
                    'middle_activation' : torch.nn.CELU(alpha=1/8),
                    'noise_eps' : 0.001
                    }

    model =  AutoQuadConv(**model_inputs)

    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1000, threshold=1e-4, factor=0.1)


    loss_fn = partial(sobolev_loss, diff_order = diff_order, lambda_r = lambda_r)

    #loss_fn = torch.nn.MSELoss()

    epochs = 1000
    numel  = model_inputs['point_seq'][0]**2
    max_test_error = 1e6

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        avg_loss = train(train_dl, model, loss_fn, optimizer, EPSILON = other_inputs['train_eps'])
        writer.add_scalar("Sobolev Norm/train", avg_loss/numel, t)

        #prof.step()
        #new_error = test(train_dl, model)
        #writer.add_scalar("MSE/train", new_error, t)

        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], t)

        scheduler.step(avg_loss)


        if optimizer.param_groups[0]["lr"] < 1e-7 or t == (epochs-1):

            filename = now.strftime("%m%d%Y_%H%M%S") + '_final' + '_epoch_{}'.format(t)

            torch.save({
            'epoch' : t,
            'model_state' : model.state_dict(),
            'optim_state'  : optimizer.state_dict(),
            'schedule_state' : scheduler.state_dict(),
            'model_inputs' : model_inputs,
            'data_inputs' : data_inputs,
            'other_inputs' : other_inputs
            },  os.path.join(save_path, filename) )

            break


        if max_test_error > avg_loss:
            max_test_error = avg_loss
            curr_time = t
            curr_model_to_save = deepcopy(model.state_dict())
            curr_opt_to_save = deepcopy(optimizer.state_dict())
            curr_schedule_to_save = deepcopy(scheduler.state_dict())
            filename = now.strftime("%m%d%Y_%H%M%S") + '_epoch_{}'.format(t)


        if (t+1) % 5 == 0:
            foldername = 'Models'
            netname = 'AutoPoint'

            save_path = os.path.join(home_dir, foldername, netname, experiment_name)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            make_gif(model, save_path, data_inputs )

            torch.save({
            'epoch' : t,
            'model_state' : curr_model_to_save,
            'optim_state'  : curr_opt_to_save,
            'schedule_state' : curr_schedule_to_save,
            'model_inputs' : model_inputs,
            'data_inputs' : data_inputs,
            'other_inputs' : other_inputs
            },  os.path.join(save_path, filename) )


    print("Done!")
    writer.flush()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--channels', type = int, nargs = '+', default = [32, 32, 32, 32, 32, 32])
    parser.add_argument('--layers', type = int, nargs = '+', default = [100, 10])
    parser.add_argument('--batch_size', type = int, default = 1)

    #with prof as p:
    main(parser)
