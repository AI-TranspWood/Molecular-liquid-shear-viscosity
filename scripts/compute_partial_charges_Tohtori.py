"""Simple ``aiida-shell`` script to get some values.

Just requires a configured AiiDA profile to be installed.
"""
import io
import os
import re
import tempfile
import time

from aiida import orm
from aiida.orm import Computer, SinglefileData, load_code, load_computer, load_node
from aiida_shell import launch_shell_job


def submit_veloxchem(self):
    print('Submitting the aiida-shell subprocess ', flush=True)
    xyzfile = load_node(self.ctx.xyz)
#    xyzfile = SinglefileData(file=os.path.join(os.getcwd(), 'mol.xyz')) # This for test purposes

    # Your Python script as a file
    script_content = """import sys
import veloxchem as vlx
infile = sys.argv[1]
molecule = vlx.Molecule.read_xyz(infile)
mol_xyz = molecule.get_xyz_string()
basis = vlx.MolecularBasis.read(molecule, "6-31G*")
resp_drv = vlx.RespChargesDriver()
resp_charges = resp_drv.compute(molecule, basis, "resp")
    """
    script_file = SinglefileData(file=io.StringIO(script_content), filename='resp.py').store()
    code = load_code('veloxchem@Tohtori')

    # Inputs to launch_shell_job
    results_veloxchem, node_veloxchem = launch_shell_job(
        'python',
        arguments = '{script_file} {xyzfile}',
        nodes={
            'script_file': script_file,
            'xyzfile': xyzfile
        },
        metadata = {
            'options': {
                'computer': load_computer('Tohtori'),
                'withmpi': False,
                'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                'max_wallclock_seconds': 1200,
                'custom_scheduler_commands': ('#SBATCH --cpus-per-task=20\n'),
                'prepend_text': (
                    'export OMP_NUM_THREADS=20\n'
                    'eval "$(conda shell.bash hook)"\n'
                    'conda activate echem')
            }
        },
        outputs = ['*.pdb']  # <- Tell AiiDA to fetch this file
    )

    print('...in veloxchem...')
    print('Outputs:')
    print(results_veloxchem.keys())

    nodelist=[]
    self.ctx.pdb = None # This deletes the pdb file from veloxchem but it is no longer needed
    for key, node in results_veloxchem.items():
        print(f'{key}: {node.__class__.__name__}<{node.pk}>')
        nodelist.append(int({node.pk}.pop()))
    self.ctx.pdb = load_node(nodelist[0])
