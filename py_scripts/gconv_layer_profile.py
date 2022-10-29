import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

from torch_geometric.utils import grid
from torch_geometric.nn import GCNConv
from torch_geometric.profile import count_parameters

#setup
point_dim = 2
size = 50
N_in = size**point_dim
N_out = size**point_dim
channels_in = 1
channels_out = 2
batch_size = 1

loss_fn = nn.functional.mse_loss
# loss_fn = SobolevLoss(spatial_dim=point_dim).cuda()

#create data
data = torch.ones(batch_size, N_in, channels_in).cuda()
ref = torch.ones(batch_size, N_out, channels_out).cuda()

_, edge_index = grid(height=size, width=size, dtype=torch.long, device="cuda")
edge_index = edge_index.transpose(0,1)

###############################################################################

#pofiling options
profile = False
record_shapes = False
profile_memory = False
with_stack = False

###############################################################################

layer = GCNConv(channels_in, channels_out, cached=True)

layer.cuda()

#print layer data
# summary(layer, input_size=(batch_size, N_in, channels_in))
print(count_parameters(layer))

###############################################################################

#dry run
loss_fn(layer(data, edge_index), ref)

###############################################################################

if profile:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    on_trace_ready=tth('../lightning_logs/profiles/quad_conv')) as prof:

        loss_fn(layer(data, edge_index), ref).backward()
