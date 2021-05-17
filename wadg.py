import firedrake as fire
from firedrake import dx, inner, grad, div
from firedrake.assemble import create_assembly_callable
import numpy as np
import spyro

## Variables
nx = 100
degree = 2
mesh = fire.UnitSquareMesh(nx,nx)
c = fire.Constant(1.0)
params = {"ksp_type": "cg", "pc_type": "jacobi"}
dt = 0.001
final_time = 1.0
frequency = 5.0
outfile = fire.File("CGcomparison.pvd")

## Setting up output

## Space
VecFS = fire.VectorFunctionSpace(mesh,"CG", degree)
ScaFS = fire.FunctionSpace(mesh, "CG", degree)
V = VecFS * ScaFS

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
wavelet = spyro.full_ricker_wavelet(dt=dt, tf=final_time, freq=frequency)

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
A = fire.assemble(LHS)
B = fire.Function(V)
assembly_callable = create_assembly_callable(RHS, tensor=B)
solv = fire.LinearSolver(A, solver_parameters=params)

## Integrating in time
dUP = fire.Function(V)
K1  = fire.Function(V)
K2  = fire.Function(V)
K3  = fire.Function(V)
rhs_forcing = fire.Function(ScaFS)
time = 0.0
step = 0
while time < final_time:
    time += dt
    # Apply source
    rhs_forcing.assign(0.0)
    assembly_callable()
    f = sources.apply_source(rhs_forcing, spyro.sources.ricker_wavelet(time, frequency))
    B0 = B.sub(1)
    B0 += f

    solv.solve(dUP, B)  # Solve for du and dp
    K1.assign(dUP)
    k1U, k1P = K1.split()

    # Second step
    u.assign(u0 + dt * k1U)
    p.assign(p0 + dt * k1P)
    assembly_callable()

    solv.solve(dUP, B)  # Solve for du and dp
    K2.assign(dUP)
    k2U, k2P = K2.split()

    # Third step
    u.assign(0.75 * u0 + 0.25 * (u + dt * k2U))
    p.assign(0.75 * p0 + 0.25 * (p + dt * k2P))
    assembly_callable()


    solv.solve(dUP, B)  # Solve for du and dp
    K3.assign(dUP)
    k3U, k3P = K3.split()

    # Updating answer
    u.assign((1.0 / 3.0) * u0 + (2.0 / 3.0) * (u + dt * k3U))
    p.assign((1.0 / 3.0) * p0 + (2.0 / 3.0) * (p + dt * k3P))

    u0.assign(u)
    p0.assign(p)
    if step %
    outfile.write(p)
    step+=1





