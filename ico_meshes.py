from firedrake.utility_meshes import UnitSquareMesh
import numpy as np
import firedrake as fire
import spyro
import meshio
import copy
import SeismicMesh
import time
from generate_dictionary import generate_model

def generate_mesh2D_tri(model, comm):

    print('Entering mesh generation', flush = True)
    degree = model['opts']['degree']
    if model['opts']['degree']   == 2:
        M = 7.020
    elif model['opts']['degree'] == 3:
        M = 3.696
    elif model['opts']['degree'] == 4:
        M = 2.664
    elif model['opts']['degree'] == 5:
        M = 2.028

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

    bbox = (-Real_Lz, 0.0, -lx, Real_Lx-lx)
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

        meshio.write_points_cells("meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".msh",
            points,[("triangle", cells)],
            file_format="gmsh22", 
            binary = False
            )
        meshio.write_points_cells("meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".vtk",
            points,[("triangle", cells)],
            file_format="vtk"
            )

    comm.comm.barrier()
    if method == "CG" or method == "KMV":
        mesh = fire.Mesh(
            "meshes/ICOSAHOM_KMVtri_homogeneous_P"+str(degree)+".msh",
            distribution_parameters={
                "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
            },
        )
    print('Finishing mesh generation', flush = True)
    return mesh

def generate_mesh2D_quad(model, comm):
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
    fire.File("meshes/meshQuadTest.pvd").write(mesh)

    return mesh

def generate_mesh3D_tri(model, comm):
    
    print('Entering mesh generation', flush = True)
    if model['opts']['degree']   == 2:
        M = 7.020
    elif model['opts']['degree'] == 3:
        M = 3.696
    elif model['opts']['degree'] == 4:
        M = 2.664
    elif model['opts']['degree'] == 5:
        M = 2.028

    method = model["opts"]["method"]

    Lz = model["mesh"]['Lz']
    lz = model['PML']['lz']
    Lx = model["mesh"]['Lx']
    lx = model['PML']['lx']
    Ly = model["mesh"]['Ly']
    ly= model['PML']['ly']

    Real_Lz = Lz + lz
    Real_Lx = Lx + 2*lx
    Real_Ly = Ly + 2*ly

    if model['testing_parameters']['experiment_type']== 'homogeneous':
        minimum_mesh_velocity = model['testing_parameters']['minimum_mesh_velocity']
        frequency = model["acquisition"]['frequency']
        lbda = minimum_mesh_velocity/frequency

        Lz = model["mesh"]['Lz']
        lz = model['PML']['lz']
        Lx = model["mesh"]['Lx']
        lx = model['PML']['lx']

        Real_Lz = Lz + lz
        Real_Lx = Lx + 2*lx
        edge_length = lbda/M
        #print(Real_Lz)

        bbox = (-Real_Lz, 0.0, 0.0, Real_Lx, 0.0, Real_Ly)
        cube = SeismicMesh.Cube(bbox)

        if comm.comm.rank == 0:
            # Creating rectangular mesh
            points, cells = SeismicMesh.generate_mesh(
            domain=cube, 
            edge_length=edge_length, 
            mesh_improvement = False,
            max_iter = 75,
            comm = comm.ensemble_comm,
            verbose = 0
            )
            print('entering spatial rank 0 after mesh generation')
            
            points, cells = SeismicMesh.geometry.delete_boundary_entities(points, cells, min_qual= 0.6)
            a=np.amin(SeismicMesh.geometry.simp_qual(points, cells))

            meshio.write_points_cells("meshes/ICOSAHOM_KMVtetra_homogeneous_P"+str(degree)+".msh",
                points,[("triangle", cells)],
                file_format="gmsh22", 
                binary = False
                )
            meshio.write_points_cells("meshes/ICOSAHOM_KMVtetra_homogeneous_P"+str(degree)+".vtk",
                points,[("triangle", cells)],
                file_format="vtk"
                )

        comm.comm.barrier()
        if method == "CG" or method == "KMV":
            mesh = fire.Mesh(
                "meshes/ICOSAHOM_KMVtetra_homogeneous_P"+str(degree)+".msh",
                distribution_parameters={
                    "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
                },
            )

def generate_mesh(model, comm):
    if   model['opts']['dimension'] == 2:
        return generate_mesh2D(model, comm)
    elif model['opts']['dimension'] == 3:
        return generate_mesh3D(model, comm)

def generate_mesh2D(model, comm):
    if   model['opts']['method'] == 'KMV':
        return generate_mesh2D_tri(model, comm)
    elif model['opts']['method'] == 'CG':
        return generate_mesh2D_quad(model, comm)

def generate_mesh3D(model, comm):
    if   model['opts']['method'] == 'KMV':
        return generate_mesh3D_tri(model, comm)
    elif model['opts']['method'] == 'CG':
        raise ValueError("3D quad mesh not yet implemented")
        #return generate_mesh3D_quad(model, comm)



dimension = 2
method = 'spectral'
if method == 'spectral':
    quadrilateral = True
elif method == 'KMV':
    quadrilateral = False

degrees = [2, 3, 4, 5]
output = False

for degree in degrees:
    model = generate_model(method, degree)
    comm = spyro.utils.mpi_init(model)
    mesh = generate_mesh(model,comm)


