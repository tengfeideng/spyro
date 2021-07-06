from firedrake.utility_meshes import UnitSquareMesh
import numpy as np
import firedrake as fire
from pyop2.mpi import COMM_WORLD
import psutil
import os
import spyro
import meshio
import copy
import SeismicMesh
import time
from generate_dictionary import generate_model

def generate_mesh2D(model, comm):

    print('Entering mesh generation', flush = True)
    M = 5
    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx


    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx
    edge_length = lbda/M

    bbox = (-Real_Lz, 0.0, 0.0, Real_Lx)
    rec = SeismicMesh.Rectangle(bbox)

    if comm.comm.rank == 0:
        points, cells = SeismicMesh.generate_mesh(
        domain=rec, 
        edge_length=edge_length, 
        mesh_improvement = False,
        comm = comm.ensemble_comm,
        verbose = 0
        )
        print('entering spatial rank 0 after mesh generation')
        
        points, cells = SeismicMesh.geometry.delete_boundary_entities(points, cells, min_qual= 0.6)
        a=np.amin(SeismicMesh.geometry.simp_qual(points, cells))

        meshio.write_points_cells("meshes/homogeneous.msh",
            points,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("meshes/homogeneous.vtk",
            points,[("triangle", cells)],
            file_format="vtk"
            )

    comm.comm.barrier()
    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/homogeneous.msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    print('Finishing mesh generation', flush = True)
    return mesh

def generate_mesh2D_quad(model, comm = COMM_WORLD):
    print('Entering mesh generation', flush = True)
    degree = model['opts']['degree']
    if model['opts']['degree']   == 2:
        M = 4
    elif model['opts']['degree'] == 3:
        M = 4
    elif model['opts']['degree'] == 4:
        M = 4
    elif model['opts']['degree'] == 5:
        M = 4

    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx


    minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
    frequency = model["acquisition"]['frequency']
    lbda = minimum_mesh_velocity/frequency

    Lz = model["mesh"]['Lz']
    lz = model['BCs']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['BCs']['lx']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx
    edge_length = lbda/M
    nz = int(Real_Lz/edge_length)
    nx = int(Real_Lx/edge_length)

    mesh = fire.RectangleMesh(nx, nz, Real_Lz, Real_Lx, quadrilateral=True)

    coordinates = copy.deepcopy(mesh.coordinates.dat.data)
    mesh.coordinates.dat.data[:,0]=-coordinates[:,0]
    mesh.coordinates.dat.data[:,1]= coordinates[:,1] - lx

    return mesh

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

method = 'spectral'
if method == 'spectral':
    quadrilateral = True
elif method == 'KMV':
    quadrilateral = False

degree = 5
output = False
model = generate_model(method, degree)

comm = spyro.utils.mpi_init(model)
#mesh = generate_mesh2D(model,comm)
Lz = model["mesh"]['Lz']
lz = model['BCs']['lz']
Lx = model["mesh"]['Lx']
lx = model['BCs']['lx']

Real_Lz = Lz + lz
Real_Lx = Lx + 2*lx

comm.comm.barrier()
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Reading mesh", flush=True)

if method == 'spectral':
    mesh = generate_mesh2D_quad(model)

elif method == 'KMV':
    mesh = fire.Mesh(
            "meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
comm.comm.barrier()

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print(f"Setting up {method} element", flush=True)

if method == 'spectral':
    element = fire.FiniteElement("CG", mesh.ufl_cell(), degree=degree, variant="spectral")
elif method == 'KMV':
    element = fire.FiniteElement(method, mesh.ufl_cell(), degree=degree, variant="KMV")

V = fire.FunctionSpace(mesh, element)

vp = fire.Constant(1.429)

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Finding sources and receivers", flush=True)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)

if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    print("Starting simulation", flush=True)

wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
t1 = time.time()
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output = output)
print(time.time() - t1, flush=True)
if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
    mem = get_memory_usage()
    print(mem, flush=True)

#spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3, show = True)
#spyro.io.save_shots(model, comm, p_r)
