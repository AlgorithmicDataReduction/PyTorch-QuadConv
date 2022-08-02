import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

from core.quadconv import QuadConvLayer as QL

layer = QL(point_dim=2,
            channels_in=1,
            channels_out=4,
            mlp_channels=[4,4])
N = 250

layer.set_quad(N)
layer.set_output_locs(N)

layer.cuda()

#create data and warmup cuda
loss_fn = nn.functional.mse_loss
data = torch.ones(1, 1, N**2).cuda()
ref = torch.ones(1, 4, N**2).cuda()

loss = loss_fn(layer(data), ref)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=tth('./lightning_logs/profiles/quad_conv')) as prof:

    loss_fn(layer(data), ref).backward()
