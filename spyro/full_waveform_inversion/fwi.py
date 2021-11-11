import os
from numpy.lib.shape_base import vsplit
from firedrake import *
import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from mpi4py import MPI

import meshio
import SeismicMesh
from SeismicMesh import write_velocity_model

import spyro
from spyro import receivers
from spyro.utils.mesh_utils import build_mesh

class FWI():
    """Runs a standart FWI gradient based optimization.
    """
    def __init__(self, model, comm = None, iteration_limit = 100, params = None):
        inner_product = 'L2'
        self.current_iteration = 0 
        self.mesh_iteration = 0
        self.iteration_limit = iteration_limit
        self.model = model
        self.dimension = model["opts"]["dimension"]
        self.method = model["opts"]["method"]
        self.degree = model["opts"]["degree"]
        self.comm = spyro.utils.mpi_init(model)
        self.shot_record = spyro.io.load_shots(model, self.comm)
        self.output_directory = "results/full_waveform_inversion/"

        if params == None:
            params = {
                "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
                "Step": {
                    "Type": "Augmented Lagrangian",
                    "Augmented Lagrangian": {
                        "Subproblem Step Type": "Line Search",
                        "Subproblem Iteration Limit": 5.0,
                    },
                    "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
                },
                "Status Test": {
                    "Gradient Tolerance": 1e-16,
                    "Iteration Limit": iteration_limit,
                    "Step Tolerance": 1.0e-16,
                },
            }
        
        if comm == None:
            self.comm = spyro.utils.mpi_init(model)
        else:
            self.comm = comm

        self.parameters = params
        if model["mesh"]["meshfile"] != None:
            mesh, V = spyro.io.read_mesh(model, self.comm)
            self.mesh = mesh
            self.space = V
        else:
            mesh, V = self._build_initial_mesh()
            self.mesh = mesh
            self.space = V

        quad_rule = finat.quadrature.make_quadrature(
            V.finat_element.cell, V.ufl_element().degree(), "KMV"
        )
        dxlump = dx(rule=quad_rule)
        self.quadrule = dxlump

        self.sources, self.receivers, self.wavelet = self._get_acquisition_geometry()
        
        self.inner = self.Inner(inner_product = inner_product)
        
        self.vp = spyro.io.interpolate(model, mesh, V, guess=False)

        self.vp = self.run_FWI()
    
    def _build_inital_mesh(self):
        vp_filename, vp_filetype = os.path.splitext(self.model["inversion"]["initial_guess"])
        if vp_filetype == '.segy':
            write_velocity_model(self.model["inversion"]["initial_guess"], ofname = vp_filename)
            new_vpfile = vp_filename+'.hdf5'
            self.model["inversion"]["initial_guess"] = new_vpfile
        
        print('Entering mesh generation', flush = True)
        mesh_filename = "fwi_mesh_"+str(self.mesh_iteration)
        self.model["mesh"]["meshfile"] = mesh_filename +".msh"
        mesh = build_mesh(self.model, self.comm, mesh_filename, vp_filename+'.segy' )
        element = spyro.domains.space.FE_method(mesh, self.model["opts"]["method"], self.model["opts"]["degree"])
        space = FunctionSpace(mesh, element)
        return mesh, space

    def _get_acquisition_geometry(self):
        comm = self.comm
        mesh = self.mesh
        V = self.space
        vp = self.vp
        sources = spyro.Sources(self.model, mesh, V, comm)
        receivers = spyro.Receivers(self.model, mesh, V, comm)
        wavelet = spyro.full_ricker_wavelet(
            dt=self.model["timeaxis"]["dt"],
            tf=self.model["timeaxis"]["tf"],
            freq=self.model["acquisition"]["frequency"],
        )
        return sources, receivers, wavelet

    def run_FWI(self, continuation = False, iterations = None):

        V = self.space
        dxlump = self.quadrule
        model = self.model
        mesh = self.mesh
        comm = self.comm
        vp = self.vp
        sources = self.sources
        wavelet = self.wavelet
        receivers = self.receivers
        
        def regularize_gradient(vp, dJ):
            """Tikhonov regularization"""
            m_u = TrialFunction(V)
            m_v = TestFunction(V)
            mgrad = m_u * m_v * dxlump
            ffG = dot(grad(vp), grad(m_v)) * dxlump
            G = mgrad - ffG
            lhsG, rhsG = lhs(G), rhs(G)
            gradreg = Function(V)
            grad_prob = LinearVariationalProblem(lhsG, rhsG, gradreg)
            grad_solver = LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                    "mat_type": "matfree",
                },
            )
            grad_solver.solve()
            dJ += gradreg
            return dJ

        class Objective(ROL.Objective):
            def __init__(self, inner_product):
                ROL.Objective.__init__(self)
                self.inner_product = inner_product
                self.p_guess = None
                self.misfit = 0.0
                self.p_exact_recv = spyro.io.load_shots(model, comm)

            def value(self, x, tol):
                """Compute the functional"""
                J_total = np.zeros((1))
                self.p_guess, p_guess_recv = spyro.solvers.forward(
                    model,
                    mesh,
                    comm,
                    vp,
                    sources,
                    wavelet,
                    receivers,
                )
                self.misfit = spyro.utils.evaluate_misfit(
                    model, p_guess_recv, self.p_exact_recv
                )
                J_total[0] += spyro.utils.compute_functional(model, self.misfit, velocity=vp)
                J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
                J_total[0] /= comm.ensemble_comm.size
                if comm.comm.size > 1:
                    J_total[0] /= comm.comm.size
                return J_total[0]

            def gradient(self, g, x, tol):
                """Compute the gradient of the functional"""
                dJ = Function(V, name="gradient")
                dJ_local = spyro.solvers.gradient(
                    model,
                    mesh,
                    comm,
                    vp,
                    receivers,
                    self.p_guess,
                    self.misfit,
                )
                if comm.ensemble_comm.size > 1:
                    comm.allreduce(dJ_local, dJ)
                else:
                    dJ = dJ_local
                dJ /= comm.ensemble_comm.size
                if comm.comm.size > 1:
                    dJ /= comm.comm.size
                # regularize the gradient if asked.
                if model['opts']['regularization']:
                    dJ = regularize_gradient(vp, dJ)
                # mask the water layer
                dJ.dat.data[water] = 0.0
                # Visualize
                if comm.ensemble_comm.rank == 0:
                    grad_file.write(dJ)
                g.scale(0)
                g.vec += dJ

            def update(self, x, flag, iteration):
                vp.assign(Function(V, x.vec, name="velocity"))
                # If iteration reduces functional, save it.
                if iteration >= 0:
                    if comm.ensemble_comm.rank == 0:
                        control_file.write(vp)
        class L2Inner(object):
            def __init__(self):
                self.A = assemble(
                    TrialFunction(V) * TestFunction(V) * dxlump, mat_type="matfree"
                )
                self.Ap = as_backend_type(self.A).mat()

            def eval(self, _u, _v):
                upet = as_backend_type(_u).vec()
                vpet = as_backend_type(_v).vec()
                A_u = self.Ap.createVecLeft()
                self.Ap.mult(upet, A_u)
                return vpet.dot(A_u)

        if continuation == True:
            self.iteration_limit = iterations
            self.parameters['Status Test']['Iteration Limit'] = iterations
            if iterations == None:
                raise ValueError('If you are continuing a FWI please specify iteration number.')

        params = ROL.ParameterList(self.parameters, "Parameters")
        comm = self.comm
        vp_output = self.output_directory+"velocity_"
        ctrl_output = self.output_directory+"control_"
        gradient_output = self.output_directory+"gradient_"

        if comm.ensemble_comm.rank == 0 and continuation == False:
            File(vp_output+"initial_guess.pvd", comm=comm.comm).write(vp)
        
        if comm.ensemble_comm.rank == 0:
            control_file = File(self.output_directory + "control.pvd", comm=comm.comm)
            grad_file = File(self.output_directory + "grad.pvd", comm=comm.comm)

        water = np.where(vp.dat.data[:] < 1.51)

        cont = 0

        inner_product = L2Inner()
        obj = Objective(inner_product)

        u = Function(V, name="velocity").assign(vp)
        opt = FeVector(u.vector(), inner_product)
        # Add control bounds to the problem (uses more RAM)
        xlo = Function(V)
        xlo.interpolate(Constant(1.0))
        x_lo = FeVector(xlo.vector(), inner_product)

        xup = Function(V)
        xup.interpolate(Constant(5.0))
        x_up = FeVector(xup.vector(), inner_product)

        bnd = ROL.Bounds(x_lo, x_up, 1.0)

        # Set up the line search
        algo = ROL.Algorithm("Line Search", params)

        algo.run(opt, obj, bnd)

        if comm.ensemble_comm.rank == 0:
            File("res.pvd", comm=comm.comm).write(obj.vp)
        
        self.vp = vp


class syntheticFWI(FWI):
    def __init__(self, model, comm = None, iteration_limit = 100, params = None):
        self.inner_product = 'L2'
        self.current_iteration = 0 
        self.mesh_iteration = 0
        self.iteration_limit = iteration_limit
        self.model = model
        self.dimension = model["opts"]["dimension"]
        self.method = model["opts"]["method"]
        self.degree = model["opts"]["degree"]
        self.comm = spyro.utils.mpi_init(model)
        try:
            self.shot_record = spyro.io.load_shots(model, self.comm)
        except:
            self.shot_record = spyro.synthetic.create_shot_record(model, self.comm)
        self.output_directory = "results/full_waveform_inversion/"

        if params == None:
            params = {
                "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
                "Step": {
                    "Type": "Augmented Lagrangian",
                    "Augmented Lagrangian": {
                        "Subproblem Step Type": "Line Search",
                        "Subproblem Iteration Limit": 5.0,
                    },
                    "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
                },
                "Status Test": {
                    "Gradient Tolerance": 1e-16,
                    "Iteration Limit": iteration_limit,
                    "Step Tolerance": 1.0e-16,
                },
            }
        
        if comm == None:
            self.comm = spyro.utils.mpi_init(model)
        else:
            self.comm = comm

        self.parameters = params
        if model['inversion']['initial_guess'] == None:
            self._smooth_and_save() 
        if model["mesh"]["meshfile"] != None:
            mesh, V = spyro.io.read_mesh(model, self.comm)
            self.mesh = mesh
            self.space = V
        else:
            mesh, V = self._build_inital_mesh()
            self.mesh = mesh
            self.space = V

        quad_rule = finat.quadrature.make_quadrature(
            V.finat_element.cell, V.ufl_element().degree(), "KMV"
        )
        dxlump = dx(rule=quad_rule)
        self.quadrule = dxlump

        self.vp = spyro.io.interpolate(model, mesh, V, guess=True)
        self.sources, self.receivers, self.wavelet = self._get_acquisition_geometry()
        
        # if model['inversion']['shot_record'] == False:
        #     self._generate_shot_record() 
        
        self.vp = self.run_FWI()

    def _generate_shot_record(self):
        model = self.model
        mesh = self.mesh
        V = self.space
        comm = self.comm 
        vp = spyro.io.interpolate(model, mesh, V, guess=False)

        p, p_r = spyro.solvers.forward(model, mesh, comm, vp, self.sources, self.wavelet, self.receivers)

        spyro.io.save_shots(model, comm)

    def _smooth_and_save(self):
        true_model = self.model['inversion']['true_model']
        filename, filetype = os.path.splitext(true_model)
        guess_model = filename+'_smooth_guess' + filetype

        spyro.synthetic.smooth_field(true_model, guess_model)
        self.model["inversion"]['initial_guess'] = guess_model
        
        



    


        

