import firedrake as fire
import firedrake as fire
from firedrake import dx, inner, grad, div
from firedrake.adjoint import assembly
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
sources = spyro.Sources(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(dt=dt, tf=final_time, freq=frequency)

## Declaring functions
UP = fire.Function(V)
u, p = UP.split()

(q_vec, q) = fire.TestFunctions(V)

dudt_trial, dpdt_trial = fire.TrialFunctions(V)

## Setting up equations
LHS = (1/c**2) * dpdt_trial * q * dx + inner(dudt_trial, q_vec) * dx
RHS = inner(u, grad(q)) * dx + p * div(q_vec) * dx 

## Assembling matrices
A = fire.assemble(LHS)
B = fire.Function(V)
assembly_callable = fire.create_assembly_callable(RHS, tensor=B)
solv = fire.LinearSolver(A, solver_parameters=params)

## Integrating in time
rhs_forcing = fire.Function(ScaFS)
time = 0.0
step = 0
while time < final_time:
    time += dt
    # Apply source
    rhs_forcing.assign(0.0)
    assembly_callable()
    f = sources.apply_source(rhs_forcing, wavelet[step])
    B0 = B.sub(0)
    B0 += f

    b1 = fire.assemble(RHS)
    solv.solve(dUP, b1)  # Solve for du and dp
    K.assign(dUP)
    solv.solve(dUP, b2)
    K.assign(K+dUP)
    K1.assign(K)
    k1U, k1P = K1.split()

    # Second step
    u.assign(u0 + dt * k1U)
    p.assign(p0 + dt * k1P)

    # solv.solve() #Solve for du and dp
    b1 = fire.assemble(RHS_1, bcs = bcp)

    solv.solve(dUP, b1)  # Solve for du and dp
    K.assign(dUP)
    solv.solve(dUP, b2)
    K.assign(K+dUP)
    K2.assign(K)
    k2U, k2P = K2.split()

    # Third step
    u.assign(0.75 * u0 + 0.25 * (u + dt * k2U))
    p.assign(0.75 * p0 + 0.25 * (p + dt * k2P))

    # solve.solve() #Solve for du and dp
    b1 = fire.assemble(RHS_1, bcs = bcp)
    if IT < dstep:
        ricker.assign(3./4.*timedependentSource(model, t, freq) + 1./4.*timedependentSource(model, t+2*float(dt), freq) )
        f.assign(expr)
        b2 = fire.assemble(RHS_2, bcs = bcp)
    solv.solve(dUP, b1)  # Solve for du and dp
    K.assign(dUP)
    solv.solve(dUP, b2)
    K.assign(K+dUP)
    K3.assign(K)
    k3U, k3P = K3.split()

    # Updating answer
    u.assign((1.0 / 3.0) * u0 + (2.0 / 3.0) * (u + dt * k3U))
    p.assign((1.0 / 3.0) * p0 + (2.0 / 3.0) * (p + dt * k3P))

    u0.assign(u)
    p0.assign(p)
    step+=1





