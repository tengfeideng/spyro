import numpy as np
import spyro

def generate_model(method, degree):
    if method == "KMV":
        quadrature = "KMV"
    elif method == 'spectral':
        method = 'CG'
        quadrature = 'GLL'

    lbda = 1.429/5.0
    model = {}
    model["opts"] = {
        "method": method,  # either CG or KMV
        "quadrature": quadrature,  # Equi or KMV
        "degree": degree,  # p order
        "dimension": 2,  # dimension
    }
    model["parallelism"] = {
        "type": "off",
    }
    model["mesh"] = {
        "Lz": 40*lbda,  # depth in km - always positive
        "Lx": 30*lbda,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
    }
    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 2,  # damping layer has a exponent variation
        "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
        "R": 1e-6,  # theoretical reflection coefficient
        "lz": lbda,  # thickness of the PML in the z-direction (km) - always positive
        "lx": lbda,  # thickness of the PML in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
    }
    model["acquisition"] = {
        "source_type": "Ricker",
        "num_sources": 1,
        "source_pos": [(-20*lbda, 15*lbda)],
        "frequency": 5.0,
        "delay": 1.0,
        "num_receivers": 15,
        "receiver_locations": spyro.create_transect((-20*lbda,20*lbda), (-20*lbda, 25*lbda), 15),
    }
    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": 4.0,  # Final time for event
        "dt": 0.0005,
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 400,  # how frequently to output solution to pvds
        "fspool": 99999,  # how frequently to save solution to RAM
    }
    model['testing_parameters'] = {
            'minimum_mesh_velocity': 1.429,
        }

    return model