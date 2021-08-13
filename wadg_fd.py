import firedrake as fire
from firedrake import dx, inner, grad, div, sin
from firedrake.assemble import create_assembly_callable
import numpy as np
from spyro.solvers import helpers
import spyro

## Variables
nx = 100
degree = 2
mesh = fire.UnitSquareMesh(nx,nx)
c = fire.Constant(1.0)
params = {"ksp_type": "cg", "pc_type": "jacobi"}
dt = 0.0001
final_time = 1.0
frequency = 5.0
nspool = 100
outfile = fire.File("CGcomparison.pvd")
output = True

## Setting up output

## Space
VecFS = fire.VectorFunctionSpace(mesh,"CG", degree)
ScaFS = fire.FunctionSpace(mesh, "CG", degree)
V = VecFS * ScaFS

bc0 = fire.DirichletBC(V.sub(0), fire.as_vector([0.0, 0.0]), "on_boundary")
bc1 = fire.DirichletBC(V.sub(1), 0.0, "on_boundary")

## Creating sources
model = {}
model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV
    "degree": degree,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}
model["acquisition"] = {
    "source_type": "Ricker",
    "frequency": frequency,
    "delay": 1.0,
    "num_sources": 1,
    "source_pos": [(0.5, 0.5)],
}
comm = spyro.utils.mpi_init(model)
sources = spyro.Sources(model, mesh, ScaFS, comm)
sources.current_source = 0
wavelet = spyro.full_ricker_wavelet(
    dt=dt,
    tf=final_time,
    freq=model["acquisition"]["frequency"],
)

## Declaring functions
UP = fire.Function(V)
u, p = UP.split()
UP0 = fire.Function(V)
u0, p0 = UP0.split()

(q_vec, q) = fire.TestFunctions(V)

dudt_trial, dpdt_trial = fire.TrialFunctions(V)

## Setting up equations
LHS = (1/c**2) * dpdt_trial * q * dx + inner(dudt_trial, q_vec) * dx
RHS = inner(u, grad(q)) * dx + p * div(q_vec) * dx 

## Assembling matrices
A = fire.assemble(LHS, bcs = [bc0,bc1])
B = fire.Function(V)
assembly_callable = create_assembly_callable(RHS, tensor=B)
solv = fire.LinearSolver(A, solver_parameters=params)

## Integrating in time
nt = int(final_time/ dt)  # number of timesteps
dUP = fire.Function(V)
rhs_forcing = fire.Function(ScaFS)

for step in range(nt):
    t = step * float(dt)

    # Apply source
    rhs_forcing.assign(0.0)
    assembly_callable()
    f = sources.apply_source(rhs_forcing, wavelet[step])
    B0 = B.sub(1)
    B0 += f

    solv.solve(dUP, B)  # Solve for du and dp
    du, dp = dUP.split()

    u.assign(u0 + dt * du)
    p.assign(p0 + dt * dp)

    u0.assign(u)
    p0.assign(p)

    if step % nspool == 0:
        # assert (
        #     fire.norm(p) < 1
        # ), "Numerical instability. Try reducing dt or building the mesh differently"
        if output:
            outfile.write(p, time=t, name="Pressure")
        if t > 0:
            helpers.display_progress(comm, t)
    





