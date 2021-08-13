import firedrake as fire
from mpi4py import MPI
import numpy  as np
from firedrake import dx, inner, grad, div, exp, sin, cos, dot
from firedrake.assemble import create_assembly_callable
from firedrake.function import Function
from ufl.tensors import as_vector

## Variables
nx = 100
degree = 2
L = 1.0
mesh = fire.RectangleMesh(nx,nx,L,L)
#mesh = fire.UnitSquareMesh(nx,nx)
x, y = fire.SpatialCoordinate(mesh)
c = fire.Constant(1.0)
params = {"ksp_type": "cg", "pc_type": "jacobi"}
dt = fire.Constant(0.001)
dt_float = float(dt)
final_time = 0.1

nspool = 100
outfile = fire.File("CG_CD_MMS.pvd")
exact_file = fire.File('exact_solution.pvd')
output = True

## Setting up output

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

(q_vec, q) = fire.TestFunctions(V)

dudt_trial, dpdt_trial = fire.TrialFunctions(V)

## Setting up equations
LHS = dpdt_trial * q * dx + dot(dudt_trial, q_vec) * dx
RHS = dot(u, grad(q)) * dx + p * div(q_vec) * dx
#p.interpolate(exp(-50*((x-L/2)**2+(y-L/2)**2))-exp(-100*((x-L/2)**2+(y-L/2)**2)))

bcs = [fire.DirichletBC(V.sub(1), 0.0, "on_boundary")]

## Assembling matrices
A = fire.assemble(LHS, bcs=bcs)
#B = fire.Function(V)
#assembly_callable = create_assembly_callable(RHS, tensor=B)
solv = fire.LinearSolver(A, solver_parameters=params)

## Integrating in time
nt = int(final_time/ dt_float)  # number of timesteps
dUP = fire.Function(V)
p_exact = fire.Function(V.sub(1))
rhs_forcing = fire.Function(V.sub(1))
t = 0.0

for step in range(nt):
    if output:
        outfile.write(p, time=t, name="Pressure")
        exact_file.write(p_exact, time=t, name="Exact pressure")
    if t > 0:
        print(f"Simulation time is: {t:{10}.{4}} seconds", flush=True)
    
    t = (step+1) * float(dt)
    RHS = dot(u, grad(q)) * dx + p * div(q_vec) * dx
    # 1st case: with pressure source
    RHS += 2 * t * sin(np.pi*x) * sin(np.pi*y) * q * dx
    RHS += t**3/3 * np.pi**2 * (sin(np.pi*x)+sin(np.pi*y)) * q * dx
    # # 2nd case: with force source
    # expr_x = t/np.pi * cos(np.pi*x)*sin(np.pi*y) + t**2 * np.pi * cos(np.pi*x)
    # expr_y = t/np.pi * sin(np.pi*x)*cos(np.pi*y) + t**2 * np.pi * cos(np.pi*y)
    # source_value = fire.project(fire.as_vector([expr_x,expr_y]), V.sub(0))
    # g.assign(source_value)
    # RHS += dot(g,q_vec)*dx

    B = fire.assemble(RHS)
    p_exact.interpolate(t**2*sin(np.pi*x)*sin(np.pi*y))

    # Apply source
    #assembly_callable()

    solv.solve(dUP, B)  # Solve for du and dp
    dU, dP = dUP.split()

    # Updating answer
    un.assign(u + dt * dU)
    pn.assign(p + dt * dP)

    u.assign(un)
    p.assign(pn)

    


error = fire.errornorm(p_exact, p)
print(f"Error of {error}", flush = True)






