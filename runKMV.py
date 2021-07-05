from firedrake.utility_meshes import UnitSquareMesh
import numpy as np
import firedrake as fire
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

method = 'KMV'
if method == 'spectral':
    quadrilateral = True
elif method == 'KMV':
    quadrilateral = False

degree = 3
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
lbda = 1.429/5.0 
num_elementsz= int(3.08*Real_Lz/(2*lbda))
print(num_elementsz, flush=True)
num_elementsx= int(3.08*Real_Lx/(2*lbda))
print(num_elementsx, flush=True)
mesh = fire.RectangleMesh(num_elementsz,num_elementsx,Real_Lz,Real_Lx, quadrilateral=quadrilateral)
coordinates = copy.deepcopy(mesh.coordinates.dat.data)
mesh.coordinates.dat.data[:,0]=-coordinates[:,0]

if method == 'spectral':
    element = fire.FiniteElement("CG", mesh.ufl_cell(), degree=degree, variant="spectral")
elif method == 'KMV':
    element = fire.FiniteElement(method, mesh.ufl_cell(), degree=degree, variant="KMV")

V = fire.FunctionSpace(mesh, element)

vp = fire.Constant(1.429)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
t1 = time.time()
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output = output)
print(time.time() - t1, flush=True)

#spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3, show = True)
#spyro.io.save_shots(model, comm, p_r)
