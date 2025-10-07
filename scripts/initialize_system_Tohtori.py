"""
Sets up the .gro, .itp and .top files for a single molecule using ACPYPE
Converts the molecule into .xyz format (using OpenBabel) for RESP charge calculation
"""
from aiida import orm
from aiida.orm import Computer, load_computer, load_code, FolderData, SinglefileData, load_node
from aiida_shell import launch_shell_job
import time, os, re, io

def wrap_file(folder, filename):
    with folder.open(filename, 'rb') as f:
        return SinglefileData(file=io.BytesIO(f.read()), filename=filename).store()

def submit_acpype(self):
    print("Submitting the aiida-shell subprocess ", flush=True)
    mol = self.ctx.mol
    smiles_string = self.ctx.smiles_string
    ff = self.ctx.ff
    basename = self.ctx.basename
    code=load_code('acpype@Tohtori')
    results_acpype, node_acpype = launch_shell_job(
                              'acpype',
                              arguments = '-i '+ smiles_string +' -n 0 -c bcc -q sqm -b '+ basename +' -a '+ ff +' -s 108000',
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
                              outputs=[f"{basename}.acpype"]
    )

    print('...in acpype...')
    print(f'Calculation terminated: {node_acpype.process_state}')
    print('Outputs:')
    print(results_acpype.keys())

    # Expected filename patterns
    target_suffixes = {
        'gro': 'GMX.gro',
        'itp': 'GMX.itp',
        'top': 'GMX.top',
        'pdb': 'NEW.pdb',
    }

    # Initialize ctx fields
    self.ctx.gro = None
    self.ctx.itp = None
    self.ctx.top = None
    self.ctx.pdb = None
    
    for key, node in results_acpype.items():
#        print(f'{key}: {node.__class__.__name__}<{node.pk}>')
        if isinstance(node, FolderData):
            for filename in node.list_object_names():
                # Match based on suffix
                if filename.endswith(target_suffixes['gro']):
                    self.ctx.gro = wrap_file(node, filename)
                elif filename.endswith(target_suffixes['itp']):
                    self.ctx.itp = wrap_file(node, filename)
                elif filename.endswith(target_suffixes['top']):
                    self.ctx.top = wrap_file(node, filename)
                elif filename.endswith(target_suffixes['pdb']):
                    self.ctx.pdb = wrap_file(node, filename)

    # Optionally verify what was found
    print(f"GRO: {self.ctx.gro.filename if self.ctx.gro else 'Not found'}")
    print(f"ITP: {self.ctx.itp.filename if self.ctx.itp else 'Not found'}")
    print(f"TOP: {self.ctx.top.filename if self.ctx.top else 'Not found'}")
    print(f"PDB: {self.ctx.pdb.filename if self.ctx.pdb else 'Not found'}")

    # Now convert .pdb to .xyz using OpenBabel

    code=load_code('obabel@Tohtori')
    results_obabel, node_obabel = launch_shell_job(
                               'obabel',
                               arguments = '{pdbfile} -O mol.xyz',
                               nodes={
                                    'pdbfile': self.ctx.pdb
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
                               outputs=["mol.xyz"]
    )

    print('...in obabel...')
    print(f'Calculation terminated: {node_acpype.process_state}')
    print('Outputs:')
    print(results_obabel.keys())
 
    nodelist=[]
    self.ctx.xyz = None
    for key, node in results_obabel.items():
        print(f'{key}: {node.__class__.__name__}<{node.pk}>')
        nodelist.append(int({node.pk}.pop()))
    self.ctx.xyz = nodelist[0]

