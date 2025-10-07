"""
Puts molecules into box, yields the .gro for molecular system
"""
from aiida import orm
from aiida.plugins import DataFactory
from aiida.orm import Computer, load_computer, load_code, load_node, Int
from aiida_shell import launch_shell_job
import time, os, re

def build_gro(self):
    print("Building system... ", flush=True)
    code = load_code('gromacs2024@Tohtori')
    grofile = self.ctx.gro
    nmols = Int(self.ctx.nmols)
    box_vector = self.ctx.box_size
    results_insert, node_insert = launch_shell_job('gmx_mpi',
                                arguments='insert-molecules -ci {grofile} -o system.gro -nmol {nmols} -try 1000 -box {box_vector} {box_vector} {box_vector}',
                                nodes={
                                    'grofile': grofile,
                                    'nmols': nmols,
                                    'box_vector': box_vector
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
                              outputs=['system.gro']
    )

    nodelist=[]
    for key, node in results_insert.items():
#        print(f'{key}: {node.__class__.__name__}<{node.pk}>')
        nodelist.append(int({node.pk}.pop()))
    self.ctx.system_gro = load_node(nodelist[0])



