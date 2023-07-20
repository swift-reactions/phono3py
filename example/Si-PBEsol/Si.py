"""Example to run thermal conductivity of Si."""
import numpy as np
from phonopy.interface.vasp import read_vasp

from contextlib import redirect_stdout

from mpi4py import MPI
import phono3py
from phono3py import Phono3py

def run_thermal_conductivity(log_level):
    """Run RTA thermal conductivity calculation from input files."""
    ph3 = phono3py.load("phono3py_disp.yaml", log_level=log_level)
    ph3.mesh_numbers = [11, 11, 11]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=range(0, 101, 10), 
        write_kappa=True,
        pinv_solver=2,
        is_LBTE=True)
    # Conductivity_RTA object
    print(ph3.thermal_conductivity.kappa)


def create_supercells_with_displacements():
    """Create supercells with displacements."""
    cell = read_vasp("POSCAR-unitcell")
    ph3 = Phono3py(cell, np.diag([2, 2, 2]), primitive_matrix="F")
    ph3.generate_displacements(distance=0.03)
    print(ph3.supercells_with_displacements)  # List of PhonopyAtoms
    print(ph3.displacements.shape)  # (supercells, atoms, xyz)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    with open(f'log_rank{rank}.txt', 'w') as f:
        with redirect_stdout(f):
            create_supercells_with_displacements()
            run_thermal_conductivity(log_level = 2*(rank==0))
