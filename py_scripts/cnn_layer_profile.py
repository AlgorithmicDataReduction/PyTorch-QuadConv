import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

N_in = 250
N_out = 250
point_dim = 2
channels_in = 1
channels_out = 4

loss_fn = nn.functional.mse_loss

data = torch.ones(1, 1, N_in, N_in).cuda()
ref = torch.ones(1, 4, N_in, N_in).cuda()

layer1 = torch.nn.Conv2d(channels_in,
                            channels_out,
                            kernel_size=3,
                            padding='same')

layer1.cuda()

loss_fn(layer1(data), ref)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                on_trace_ready=tth('../lightning_logs/profiles/quad_conv')) as prof:

    loss_fn(layer1(data), ref).backward()
