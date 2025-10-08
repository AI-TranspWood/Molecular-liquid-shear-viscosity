"""
Runs system minimization in preparation for equilibration run
"""
import os
import re
import time

from aiida import orm
from aiida.orm import SinglefileData, load_code, load_computer, load_node
from aiida_shell import launch_shell_job


def submit_minimization(self):
    print('Running grompp... ', flush=True)
    code = load_code('gromacs2024@Tohtori')
    grofile = self.ctx.system_gro
    topfile = self.ctx.top_updated
    itpfile = self.ctx.itp_with_resp
    mdpfile = SinglefileData(file=os.path.join(os.getcwd(), 'minimize.mdp'))
    results_grompp, node_grompp = launch_shell_job('gmx_mpi',
                                arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o minimize.tpr',
                                nodes={
                                    'mdpfile': mdpfile,
                                    'grofile': grofile,
                                    'topfile': topfile,
                                    'itpfile': itpfile
                                },
                                metadata={
                                  'options': {
                                      'computer': load_computer('Tohtori'),
                                      'withmpi': False,
                                      'resources': {
                                          'num_machines': 1,
                                          'num_mpiprocs_per_machine': 1,
                                      }
                                  }
                              },
                              outputs=['mdout.mdp','minimize.tpr']
    )

    print('...in grompp...')
    print(f'Calculation terminated: {node_grompp.process_state}')

    nodelist=[]
    for key, node in results_grompp.items():
        nodelist.append(node.pk)
    self.ctx.tpr = load_node(nodelist[1])

# gromacs command
# gmx_mpi mdrun -v -deffnm minimize

    print('Running mdrun... ', flush=True)
    tprfile = self.ctx.tpr

    custom_scheduler_commands = '\n'.join([
        '#SBATCH -p gen02_ivybridge',
        '#SBATCH --exclude=c[1003-1006,2001]'
    ])

    prepend_text = '\n'.join([
        'export OMP_NUM_THREADS=1',
        'module load gcc/12.2.0'
    ])

    results_mdrun, node_mdrun = launch_shell_job('gmx_mpi',
                                arguments='mdrun -v -s {tprfile} -deffnm minimize',
                                nodes={
                                    'tprfile': tprfile
                                },
                                metadata={
                                  'options': {
                                      'computer': load_computer('Tohtori'),
                                      'withmpi': True,
                                      'resources': {
                                          'num_machines': 1,
                                          'num_mpiprocs_per_machine': 20},
                                      'custom_scheduler_commands': custom_scheduler_commands,
                                      'prepend_text': prepend_text
                                  }
                                },
                                outputs=['minimize.gro']
    )

    print('...in mdrun...')
    print(f'Calculation terminated: {node_mdrun.process_state}')

    nodelist=[]
    for key, node in results_mdrun.items():
        nodelist.append(node.pk)
    self.ctx.minimized_gro = load_node(nodelist[0])
