from firedrake import *
from scipy.optimize import *
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
import SeismicMesh
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
#from ..domains import quadrature, space

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV", # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.5,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(0.75, 0.75)],
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 1,
    "receiver_locations": spyro.create_transect(
       (0.9, 0.75), (0.9, 0.75), 1
    ),
}
model["Aut_Dif"] = {
    "status": True, 
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.001*800,  # Final time for event (for test 7)
    "dt": 0.0010,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}

comm = spyro.utils.mpi_init(model)

mesh = RectangleMesh(100, 100, 1.5, 1.5, diagonal="crossed") # to test FWI, mesh aligned with interface
# mesh.coordinates.dat.data[:, 0] -= 1.5
# mesh.coordinates.dat.data[:, 1] -= 0.0

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)

V = FunctionSpace(mesh, element)

lamb_exact = Constant(1.)  # exact
lamb_guess = Constant(0.9) # guess
rho = Constant(1.)  
vp_exact = Function(V).interpolate( (lamb_exact / rho) ** 0.5 )
vp_guess = Function(V).interpolate( (lamb_guess / rho) ** 0.5 )
AD = model["Aut_Dif"]["status"] 
if AD:
    spyro.tools.gradient_test_acoustic_ad(model, mesh, V, comm, vp_exact, vp_guess)

else:
    spyro.tools.gradient_test_acoustic(model, mesh, V, comm, vp_exact, vp_guess)
