import firedrake as fire
from mpi4py import MPI
import numpy  as np
from firedrake import dx, inner, grad, div, exp, sin, cos, dot
from firedrake.assemble import create_assembly_callable
from firedrake.function import Function
from ufl.formoperators import rhs
from ufl.tensors import as_vector
import spyro

## Variables
nx = 100
degree = 2
L = 1.0
mesh = fire.RectangleMesh(nx,nx,L,L)
#mesh = fire.UnitSquareMesh(nx,nx)
x, y = fire.SpatialCoordinate(mesh)
c = fire.Constant(1.0)
params = {"ksp_type": "cg", "pc_type": "jacobi"}
dt = fire.Constant(0.0005)
dt_float = float(dt)
final_time = 0.1
nspool = 10

## Source information
VecFS = fire.VectorFunctionSpace(mesh,"CG", degree)
ScaFS = fire.FunctionSpace(mesh, "CG", degree)
V = VecFS * ScaFS
wavelet = spyro.full_ricker_wavelet(dt=dt_float, tf=final_time, freq=5.0)
model = {}
model['opts'] = {
    'method':'CG',
    'degree':degree,
    'dimension':2,
}
model['parallelism'] = {
    'type':'spatial',
}
model['acquisition'] = {
    'source_type': 'Ricker',
    'num_sources': 1,
    'source_pos': [(0.5,0.5)],
    'frequency': 5.0,
    'delay':1.0,
    'num_receivers': 50,
    "receiver_locations": spyro.create_transect((0.2, 0.5), (0.8, 0.5), 50),
}

comm = spyro.utils.mpi_init(model)
source = spyro.Sources(model, mesh, ScaFS, comm)
source.current_source = 0
receivers = spyro.Receivers(model, mesh, ScaFS, comm)

