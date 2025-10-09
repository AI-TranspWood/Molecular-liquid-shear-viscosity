"""Implementation of the WorkChain for AITW viscosity calculation."""
from glob import glob
import os
import re
from tempfile import NamedTemporaryFile

from aiida import orm
from aiida.engine import ToContext, WorkChain
from aiida.orm import load_node
from aiida_shell import launch_shell_job
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from . import functions as fnc
from .NemdParallelWorkChain import NemdParallelWorkChain
from .PostprocessPressureWorkChain import PostprocessPressureWorkChain


def wrap_file(folder, filename):
    with folder.open(filename, 'rb') as f:
        return orm.SinglefileData(f, filename=filename).store()
        # return orm.SinglefileData(file=io.BytesIO(f.read()), filename=filename).store()

class MonomerWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input('smiles_string', valid_type=orm.Str, help='The SMILES string of the molecule to simulate.')
        spec.input(
            'force_field', valid_type=orm.Str,
            default=lambda: orm.Str('gaff2'),
            help='The SMILES string of the molecule to simulate.'
        )
        spec.input(
            'nmols', valid_type=orm.Int,
            default=lambda: orm.Int(1000),
            help='The number of molecules to insert into the simulation box.'
        )
        # spec.input('computer_label', valid_type=orm.Str, help='The computer to run the workflow on.')
        spec.input('acpype_code', valid_type=orm.AbstractCode, help='Code for running the `acpype` program.')
        spec.input('obabel_code', valid_type=orm.AbstractCode, help='Code for running the `obabel` program.')
        spec.input('veloxchem_code', valid_type=orm.AbstractCode, help='Code for python with `veloxchem` installed.')
        spec.input('gmx_code', valid_type=orm.AbstractCode, help='Code for running `gmx` or `gmx_mpi`.')

        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            cls.submit_acpype,
            cls.submit_veloxchem,
            cls.run_resp_injection,
            cls.update_top_file,
            cls.get_box_size,
            cls.build_gro,
            cls.submit_minimization,
            cls.submit_equilibration,
            cls.set_nemd_inputs,
            cls.submit_parallel_nemd,
            cls.collect_outputs,
            cls.submit_postprocessing,
            cls.finalize
        )

        # OUTPUTS ############################################################################
        spec.outputs.dynamic = True

    def setup(self):
        """Setup context variables."""
        self.ctx.smiles_string = self.inputs.smiles_string.value
        self.ctx.ff = self.inputs.force_field.value
        self.ctx.nmols = self.inputs.nmols.value

    def submit_acpype(self):
        self.report('Submitting the aiida-shell subprocess ')
        basename = 'aiida'
        # code=load_code('acpype@Tohtori')
        results_acpype, node_acpype = launch_shell_job(
            self.inputs.acpype_code,
            arguments=f'-i {{smiles}} -n 0 -c bcc -q sqm -b {basename} -a {{ff}} -s 108000',
            nodes={
                'smiles': self.inputs.smiles_string,
                'ff': self.inputs.force_field
            },
            metadata={
                'options': {
                    'withmpi': False,
                }
            },
            outputs=[f'{basename}.acpype']
        )

        self.report('...in acpype...')
        self.report(f'Calculation terminated: {node_acpype.process_state}')
        self.report('Outputs:')
        self.report(results_acpype.keys())
        self.report(results_acpype)

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
            # self.report(f'{key}: {node.__class__.__name__}<{node.pk}>')
            if isinstance(node, orm.FolderData):
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
        self.report(f"GRO: {self.ctx.gro.filename if self.ctx.gro else 'Not found'}")
        self.report(f"ITP: {self.ctx.itp.filename if self.ctx.itp else 'Not found'}")
        self.report(f"TOP: {self.ctx.top.filename if self.ctx.top else 'Not found'}")
        self.report(f"PDB: {self.ctx.pdb.filename if self.ctx.pdb else 'Not found'}")

        results_obabel, node_obabel = launch_shell_job(
            self.inputs.obabel_code,
            arguments = '{pdbfile} -O mol.xyz',
            nodes={
                'pdbfile': self.ctx.pdb
            },
            metadata={
                'options': {
                    'withmpi': False,
                    'redirect_stderr': True,
                }
            },
            outputs=['mol.xyz']
        )

        self.report('...in obabel...')
        self.report(f'Calculation terminated: {node_acpype.process_state}')
        self.report('Outputs:')
        self.report(results_obabel)

        self.ctx.xyz = results_obabel['mol_xyz']

    def submit_veloxchem(self):
        self.report('Submitting the aiida-shell subprocess ')
        xyzfile = self.ctx.xyz
        # xyzfile = orm.SinglefileData(file=os.path.join(os.getcwd(), 'mol.xyz')) # This for test purposes

        # TODO: this is creating a new node it should be done in a calcfunction also probably generalizing the inputs
        # EG is 6-31G* what should always beb used? Should this be user settable?
        script_content = '\n'.join([
            'import sys',
            'import veloxchem as vlx',
            'infile = sys.argv[1]',
            'molecule = vlx.Molecule.read_xyz(infile)',
            'mol_xyz = molecule.get_xyz_string()',
            'basis = vlx.MolecularBasis.read(molecule, "6-31G*")',
            'resp_drv = vlx.RespChargesDriver()',
            'resp_charges = resp_drv.compute(molecule, basis, "resp")',
        ])
        script_file = orm.SinglefileData.from_string(script_content, filename='resp.py').store()

        # Inputs to launch_shell_job
        results_veloxchem, node_veloxchem = launch_shell_job(
            self.inputs.veloxchem_code,
            arguments = '{script_file} {xyzfile}',
            nodes={
                'script_file': script_file,
                'xyzfile': xyzfile
            },
            metadata = {
                'options': {
                    'withmpi': False,
                }
            },
        )

        self.report('...in veloxchem...')
        self.report('Outputs:')
        self.report(results_veloxchem.keys())

        self.ctx.pdb = results_veloxchem['stdout']

    def run_resp_injection(self):
        """
        Inject RESP charges from a PDB file into an ITP file and store the result in self.ctx.
        Assumes self.ctx.pdb and self.ctx.itp are SinglefileData nodes.
        """
        self.ctx.itp_with_resp = fnc.run_resp_injection(
            pdb_file=self.ctx.pdb,
            itp_file=self.ctx.itp
        )

        self.report(f'Updated ITP file with RESP charges stored as node <{self.ctx.itp_with_resp.pk}>')

    def update_top_file(self):
        """Update the .top file to reference the new .itp file and correct molecule count."""
        self.ctx.top_updated = fnc.update_top_file(
            nmols=self.inputs.nmols,
            top_file=self.ctx.top,
            itp_file=self.ctx.itp_with_resp,
        )

        self.report(f"Updated .top file stored: {self.ctx.top_updated.filename} (pk={self.ctx.top_updated.pk})")

    def get_box_size(self):
        """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
        self.ctx.box_size = fnc.get_box_size(
            nmols=self.inputs.nmols,
            smiles_string=self.inputs.smiles_string
        )

        self.report(f"Calculated box edge length: {self.ctx.box_size.value:.2f} nm")

    def build_gro(self):
        self.report('Building system... ')
        grofile = self.ctx.gro
        nmols = orm.Int(self.ctx.nmols)
        box_vector = self.ctx.box_size

        results_insert, node_insert = launch_shell_job(
            self.inputs.gmx_code,
            arguments=(
                'insert-molecules -ci {grofile} -o system.gro -nmol {nmols} '
                '-try 1000 -box {box_vector} {box_vector} {box_vector}'
            ),
            nodes={
                'grofile': grofile,
                'nmols': nmols,
                'box_vector': box_vector
            },
            metadata={
                'options': {
                    'withmpi': False,
                }
            },
            outputs=['system.gro']
        )

        nodelist=[]
        for key, node in results_insert.items():
            nodelist.append(int({node.pk}.pop()))
        self.ctx.system_gro = load_node(nodelist[0])

    def submit_minimization(self):
        print('Running grompp... ')
        grofile = self.ctx.system_gro
        topfile = self.ctx.top_updated
        itpfile = self.ctx.itp_with_resp
        mdpfile = orm.SinglefileData(file=os.path.join(os.getcwd(), 'minimize.mdp'))

        # code = load_code('gromacs2024@Tohtori')
        results_grompp, node_grompp = launch_shell_job(
            self.inputs.gmx_code,
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o minimize.tpr',
            nodes={
                'mdpfile': mdpfile,
                'grofile': grofile,
                'topfile': topfile,
                'itpfile': itpfile
            },
            metadata={
                'options': {
                    'withmpi': False,
                }
            },
            outputs=['mdout.mdp','minimize.tpr']
        )

        self.report('...in grompp...')
        self.report(f'Calculation terminated: {node_grompp.process_state}')

        nodelist=[]
        for key, node in results_grompp.items():
            nodelist.append(node.pk)
        self.ctx.tpr = load_node(nodelist[1])

        # gromacs command
        # gmx_mpi mdrun -v -deffnm minimize
        self.report('Running mdrun... ')
        tprfile = self.ctx.tpr

        results_mdrun, node_mdrun = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm minimize',
            nodes={
                'tprfile': tprfile
            },
            metadata={
                'options': {
                    'withmpi': True,
                }
            },
            outputs=['minimize.gro']
        )

        self.report('...in mdrun...')
        self.report(f'Calculation terminated: {node_mdrun.process_state}')

        nodelist=[]
        for key, node in results_mdrun.items():
            nodelist.append(node.pk)
        self.ctx.minimized_gro = load_node(nodelist[0])

    def submit_equilibration(self):
        self.report('Running grompp... ')
        grofile = self.ctx.minimized_gro
        topfile = self.ctx.top_updated
        itpfile = self.ctx.itp_with_resp
        mdpfile = orm.SinglefileData(file=os.path.join(os.getcwd(), 'equilibrate.mdp'))

        # code = load_code('gromacs2024@Tohtori')
        results_grompp, node_grompp = launch_shell_job(
            self.inputs.gmx_code,
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o equilibrate.tpr',
            nodes={
                'mdpfile': mdpfile,
                'grofile': grofile,
                'topfile': topfile,
                'itpfile': itpfile
            },
            metadata={
                'options': {
                    'withmpi': False,
                }
            },
            outputs=['mdout.mdp','equilibrate.tpr']
        )

        self.report('...in grompp...')
        self.report(f'Calculation terminated: {node_grompp.process_state}')

        nodelist=[]
        for key, node in results_grompp.items():
            nodelist.append(node.pk)
        self.ctx.tpr = load_node(nodelist[1])

        # gromacs command
        # gmx_mpi mdrun -v -deffnm minimize

        self.report('Running mdrun... ')
        tprfile = self.ctx.tpr

        results_mdrun, node_mdrun = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm equilibrate',
            nodes={
                'tprfile': tprfile
            },
            metadata={
                'options': {
                    'withmpi': True,
                }
            },
            outputs=['equilibrate.gro']
        )

        self.report('...in mdrun...')
        self.report(f'Calculation terminated: {node_mdrun.process_state}')

        nodelist=[]
        for key, node in results_mdrun.items():
            nodelist.append(node.pk)
        self.ctx.equilibrated_gro = load_node(nodelist[0])

    def set_nemd_inputs(self):
        """Load prepared input files for NEMD step."""
        mdp_files = sorted(glob('eta_*.mdp'))
        self.ctx.mdp_files = orm.List(list=[os.path.abspath(f) for f in mdp_files])

    def submit_parallel_nemd(self):
        """Submit the parallel NEMD WorkChain."""
        inputs = {
            'grofile': self.ctx.equilibrated_gro,
            'topfile': self.ctx.top_updated,
            'itpfile': self.ctx.itp_with_resp,
            'mdp_files': self.ctx.mdp_files
        }
        future = self.submit(NemdParallelWorkChain, **inputs)
        return ToContext(nemd=future)

    def collect_outputs(self):
        """Collect .edr files from the NEMD parallel run."""
        nemd_wc = self.ctx.nemd

        if not nemd_wc.is_finished_ok:
            self.report('NemdParallelWorkChain did not finish successfully.')
            return

        self.ctx.edr_outputs = list(nemd_wc.outputs.edr_outputs.values())

        self.report('All done! Collected outputs:')
        for i, node in enumerate(self.ctx.edr_outputs):
            self.report(f"edr_output {i}: {node.filename}")
            self.out(f'edr_output_{i}', node)

    def submit_postprocessing(self):
        edr_inputs = {f'edr_{i}': node for i, node in enumerate(self.ctx.edr_outputs)}
        future = self.submit(
            PostprocessPressureWorkChain,
            edr_files=edr_inputs,
            grofile=self.ctx.equilibrated_gro,
            mdp_files=self.ctx.mdp_files
        )
        return ToContext(postprocess=future)

    def finalize(self):
        """Expose outputs from postprocessing."""
        for key in self.ctx.postprocess.outputs:
            value = self.ctx.postprocess.outputs[key]
            self.out(key, value)
