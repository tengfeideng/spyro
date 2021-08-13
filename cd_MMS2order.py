import firedrake as fire
from mpi4py import MPI
import numpy  as np
from firedrake import dx, inner, grad, div, exp, sin, dot
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
outfile = fire.File("CG_CD_MMS2order.pvd")
exact_file = fire.File('exact_solution2order.pvd')
output = True

## Setting up output

## Space
ScaFS = fire.FunctionSpace(mesh, "CG", degree)
V = ScaFS

## Declaring functions
pm = fire.Function(V)
p = fire.Function(V)
pn = fire.Function(V)

temp = fire.Function(V)

q = fire.TestFunction(V)

d2pdt2_trial = fire.TrialFunction(V)

## Setting up equations
LHS = d2pdt2_trial * q * dx
RHS = -dot(grad(p),grad(q))*dx 
#p.interpolate(exp(-50*((x-L/2)**2+(y-L/2)**2))-exp(-100*((x-L/2)**2+(y-L/2)**2)))

bcs = [fire.DirichletBC(V, 0.0, "on_boundary")]

## Assembling matrices
A = fire.assemble(LHS, bcs=bcs)
#B = fire.Function(V)
#assembly_callable = create_assembly_callable(RHS, tensor=B)
solv = fire.LinearSolver(A, solver_parameters=params)

## Integrating in time
nt = int(final_time/ dt_float)  # number of timesteps
d2pdt2 = fire.Function(V)
p_exact = fire.Function(V)

for step in range(nt):
    t = step * float(dt)
    RHS = -dot(grad(p),grad(q))*dx  + 2*sin(np.pi*x)*sin(np.pi*y)*(1+t**2*np.pi**2)*q*dx
    B = fire.assemble(RHS)
    p_exact.interpolate(t**2*sin(np.pi*x)*sin(np.pi*y))

    # Apply source
    #assembly_callable()

    solv.solve(d2pdt2, B)  # Solve for du and dp

    # Updating answer
    temp.assign(2*p -pm + dt**2 * d2pdt2)
    p.assign(pn)
    pm.assign(p)
    pn.assign(temp)

    if output:
        outfile.write(p, time=t, name="Pressure")
        
        exact_file.write(p_exact, time=t, name="Exact pressure")
    if t > 0:

        print(f"Simulation time is: {t:{10}.{4}} seconds", flush=True)


error = fire.errornorm(p_exact, p)
print(f"Error of {error}", flush = True)






