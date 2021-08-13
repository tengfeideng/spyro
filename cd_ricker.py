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

## Setting up output
outfile = fire.File("CG_CD_MMS.pvd")
exact_file = fire.File('exact_solution.pvd')
output = True


## Space
VecFS = fire.VectorFunctionSpace(mesh,"CG", degree)
ScaFS = fire.FunctionSpace(mesh, "CG", degree)
V = VecFS * ScaFS

## Declaring functions
UP = fire.Function(V)
u, p = UP.split()

g = fire.Function(V.sub(0))

UPn = fire.Function(V)
un, pn = UPn.split()

ss = fire.Function(V)
g, f = ss.split()

(q_vec, q) = fire.TestFunctions(V)

dudt_trial, dpdt_trial = fire.TrialFunctions(V)

## Setting up equations
LHS = dpdt_trial * q * dx + dot(dudt_trial, q_vec) * dx
RHS = dot(u, grad(q)) * dx + p * div(q_vec) * dx

bcs = [fire.DirichletBC(V.sub(1), 0.0, "on_boundary")]

## Assembling matrices
A = fire.assemble(LHS, bcs=bcs)
B = fire.Function(V)
#assembly_callable = create_assembly_callable(RHS, tensor=B)
solv = fire.LinearSolver(A, solver_parameters=params)
assembly_callable = create_assembly_callable(RHS, tensor=B)

## Integrating in time
nt = int(final_time/ dt_float)  # number of timesteps
dUP = fire.Function(V)

rhs_forcing = fire.Function(V.sub(1))
g_value = 0.0
t = 0.0

for step in range(nt):
    t = (step) * float(dt)
    if output and step % nspool == 0:
        outfile.write(p, time=t, name="Pressure")

        print(f"Simulation time is: {t:{10}.{4}} seconds", flush=True)
        print(g_value, flush = True)
    
    # Applying source
    rhs_forcing.assign(0.0)
    assembly_callable()
    # integrating source since it is int(fdt)
    g_value += dt_float*wavelet[step]
    g = source.apply_source(rhs_forcing,g_value)

    B1 = B.sub(1)
    B1 += g

    solv.solve(dUP, B)  # Solve for du and dp
    dU, dP = dUP.split()

    # Updating answer
    un.assign(u + dt * dU)
    pn.assign(p + dt * dP)

    u.assign(un)
    p.assign(pn)

    






