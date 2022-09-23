import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

from core.quadconv import QuadConvLayer as QL1
from core.quadconv2 import QuadConvLayer as QL2

point_dim = 2
N_in = 150**point_dim
N_out = 150**point_dim
channels_in = 1
channels_out = 4
batch_size = 2
filter_size = [4, 4]

loss_fn = nn.functional.mse_loss

#create data
data = torch.ones(batch_size, channels_in, N_in).cuda()
ref = torch.ones(batch_size, channels_out, N_out).cuda()

record_shapes=False
profile_memory=True
with_stack=False

###############################################################################

# layer1 = QL1(point_dim,
#                 channels_in,
#                 channels_out,
#                 mlp_channels=[4,4])
#
# layer1.set_quad(N_in)
# layer1.set_output_locs(N_out)
#
# layer1.cuda()

###############################################################################

layer2 = QL2(point_dim,
            N_in,
            N_out,
            channels_in,
            channels_out,
            filter_size)

layer2.cuda()

###############################################################################

#dry run
# loss_fn(layer1(data), ref)
loss_fn(layer2(data), ref)

###############################################################################

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                 record_shapes=record_shapes,
#                 profile_memory=profile_memory,
#                 with_stack=with_stack,
#                 on_trace_ready=tth('./lightning_logs/profiles/quad_conv')) as prof:
#
#     loss_fn(layer1(data), ref).backward()

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                 record_shapes=record_shapes,
#                 profile_memory=profile_memory,
#                 with_stack=with_stack,
#                 on_trace_ready=tth('./lightning_logs/profiles/quad_conv')) as prof:
#
#     loss_fn(layer2(data), ref).backward()
