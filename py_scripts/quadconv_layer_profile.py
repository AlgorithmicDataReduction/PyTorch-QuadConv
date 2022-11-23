import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler as tth

from torchinfo import summary

from torch_quadconv import QuadConv, MeshHandler

#setup
spatial_dim = 2
in_points = 50**spatial_dim
out_points = 50**spatial_dim
in_channels = 1
out_channels = 2
batch_size = 1
filter_seq = [4, 4]

loss_fn = nn.functional.mse_loss
# loss_fn = SobolevLoss(spatial_dim=point_dim).cuda()

#create data
input = torch.ones(batch_size, in_channels, in_points).cuda()
output = torch.ones(batch_size, out_channels, out_points).cuda()

#mesh
mesh = MeshHandler(torch.rand(in_points, spatial_dim))
mesh = mesh.cache([in_points, out_points]).cuda()

###############################################################################

#pofiling options
compute_profile = True
record_shapes = False
profile_memory = True
with_stack = True

###############################################################################

layer = QuadConv(spatial_dim=spatial_dim,
                    in_points=in_points,
                    out_points=out_points,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    filter_seq=filter_seq,
                    cache=True).cuda()

#print layer data
# summary(layer, input_size=(batch_size, in_channels, in_points))

###############################################################################

#dry run
loss_fn(layer(mesh, input), output)

###############################################################################

if compute_profile:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    on_trace_ready=tth('./lightning_logs/profiles/quad_conv')) as prof:

        layer(mesh, input)
        # loss_fn(layer(mesh, input), output)
        # loss_fn(layer(mesh, input), output).backward()
