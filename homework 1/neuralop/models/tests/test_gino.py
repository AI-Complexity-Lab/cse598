import torch
from torch.autograd import grad
import pytest
from tensorly import tenalg

tenalg.set_backend("einsum")

from ..gino import GINO

# Fixed variables
in_channels = 3
out_channels = 2
projection_channels = 16
lifting_channels = 16
fno_n_modes = (8,8,8)

# data parameters
n_in = 100
n_out = 100
latent_density = 8


@pytest.mark.parametrize("batch_size", [1,4])
@pytest.mark.parametrize("gno_coord_dim", [2,3])
@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
def test_gino(gno_transform_type, gno_coord_dim, batch_size):
    if torch.backends.cuda.is_built():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")
    
    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_radius=0.3,# make this large to ensure neighborhoods fit
        projection_channels=projection_channels,
        gno_coord_dim=gno_coord_dim,
        in_gno_mlp_hidden_layers=[16,16],
        out_gno_mlp_hidden_layers=[16,16],
        in_gno_transform_type=gno_transform_type,
        out_gno_transform_type=gno_transform_type,
        fno_n_modes=fno_n_modes[:gno_coord_dim],
        # keep the FNO model small for runtime
        fno_lifting_channels=lifting_channels,
        fno_projection_channels=projection_channels,
        fno_norm="ada_in", #TODO: Parametrize this
    ).to(device)

    # create grid of latent queries on the unit cube
    latent_geom = torch.stack(torch.meshgrid([torch.linspace(0,1,latent_density)] * gno_coord_dim, indexing='xy'))
    latent_geom = latent_geom.permute(*list(range(1,gno_coord_dim+1)),0).to(device)

    # create input geometry and output queries
    input_geom_shape = [n_in, gno_coord_dim]
    input_geom = torch.randn(*input_geom_shape, device=device)
    output_queries_shape = [n_out, gno_coord_dim]
    output_queries = torch.randn(*output_queries_shape, device=device)

    # create data and features
    x_shape = [batch_size, n_in, in_channels]
    x = torch.randn(*x_shape, device=device)
    # require and retain grad to check for backprop
    x.requires_grad_(True)

    ada_in = torch.randn(1, device=device)

    # Test forward pass
    out = model(x=x,
                input_geom=input_geom,
                latent_queries=latent_geom,
                output_queries=output_queries,
                ada_in=ada_in)

    # Check output size
    assert list(out.shape) == [batch_size, n_out, out_channels]

    # Check backward pass
    assert out.isfinite().all()
    if batch_size > 1:
        loss = out[0].sum()
    else:
        loss = out.sum()
    loss.backward()
    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
    if batch_size > 1:
        # assert f[1:] accumulates no grad
        assert not x.grad[1:].nonzero().any()
