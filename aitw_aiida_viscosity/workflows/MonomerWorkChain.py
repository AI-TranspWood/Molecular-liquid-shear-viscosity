"""Implementation of the WorkChain for AITW viscosity calculation."""
from glob import glob
import os

from aiida import orm
from aiida.engine import ToContext, WorkChain
from aiida.orm import load_node
from aiida_shell import launch_shell_job

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
            'reference_temperature', valid_type=orm.Float,
            help='The reference temperature in Kelvin for the simulation.'
        )
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

        spec.input(
            'veloxchem_basis', valid_type=orm.Str,
            default=lambda: orm.Str('6-31G*'),
            help=(
                'The basis set to use in the VeloxChem calculation. This should be 6-31G* for RESP partial charges '
                'with the GAFF and GAFF2 force fields.'
            )
        )
        spec.input(
            'gromacs_minimization_steps', valid_type=orm.Int,
            default=lambda: orm.Int(5000),
            help='The number of steps to use in the GROMACS minimization.'
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
            cls.submit_obabel,
            cls.make_veloxchem_input,
            cls.submit_veloxchem,
            cls.run_resp_injection,
            cls.update_top_file,
            cls.get_box_size,
            cls.build_gro,
            cls.make_gromacs_minimization_input,
            cls.submit_minimization_init,
            cls.submit_minimization_run,
            # cls.submit_equilibration,
            # cls.set_nemd_inputs,
            # cls.submit_parallel_nemd,
            # cls.collect_outputs,
            # cls.submit_postprocessing,
            # cls.finalize
        )

        # OUTPUTS ############################################################################
        spec.outputs.dynamic = True

        # ERRORS ############################################################################
        spec.exit_code(
            300, 'ERROR_ACPYPE_MISSING_OUTPUT',
            message='ACPYPE did not produce all expected output files.'
        )

    def setup(self):
        """Setup context variables."""
        self.ctx.smiles_string = self.inputs.smiles_string.value
        self.ctx.ff = self.inputs.force_field.value
        self.ctx.nmols = self.inputs.nmols.value

    def submit_acpype(self):
        """Submit acpype and obabel calculations to generate initial structure and parameters."""
        basename = 'aiida'

        self.report('Running acpype through aiida-shell...')
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
        self.report(f'Submitted job: {node_acpype}')
        self.report(f'Outputs: {results_acpype}')

        # Expected filename patterns
        target_suffixes = {
            'gro': 'GMX.gro',
            'itp': 'GMX.itp',
            'top': 'GMX.top',
            'pdb': 'NEW.pdb',
        }

        # TODO: probably should use a CalcFunction here to extract and store these files
        missing = set(target_suffixes.keys())
        out_node: orm.FolderData = results_acpype[f'{basename}_acpype']
        for file_obj in out_node.list_objects():
            filename = file_obj.name
            for key in missing.copy():
                suffix = target_suffixes[key]
                if filename.endswith(suffix):
                    setattr(self.ctx, key, wrap_file(out_node, filename))
                    missing.remove(key)
                    break
        if missing:
            self.report(f'Missing expected output files from ACPYPE: {missing}')
            return self.exit_codes.ERROR_ACPYPE_MISSING_OUTPUT

    def submit_obabel(self):
        """Convert PDB file to XYZ using Open Babel."""
        self.report('Running obabel through aiida-shell...')
        out_filename = 'mol.xyz'
        results_obabel, node_obabel = launch_shell_job(
            self.inputs.obabel_code,
            arguments = f'{{pdbfile}} -O {out_filename}',
            nodes={
                'pdbfile': self.ctx.pdb
            },
            metadata={
                'options': {
                    'withmpi': False,
                    'redirect_stderr': True,
                }
            },
            outputs=[out_filename]
        )

        self.report(f'Submitted job: {node_obabel}')
        self.report(f'Outputs: {results_obabel.keys()}')

        self.ctx.xyz = results_obabel[out_filename.replace('.', '_')]

    def make_veloxchem_input(self):
        """Prepare input files for VeloxChem calculation."""
        self.ctx.veloxchem_input = fnc.generate_veloxchem_input(self.inputs.veloxchem_basis)

    def submit_veloxchem(self):
        """Submit a VeloxChem calculation to compute RESP charges and store the resulting PDB file"""
        self.report('Running veloxchem through aiida-shell...')
        results_veloxchem, node_veloxchem = launch_shell_job(
            self.inputs.veloxchem_code,
            arguments = '{script_file} {xyzfile}',
            nodes={
                'script_file': self.ctx.veloxchem_input,
                'xyzfile': self.ctx.xyz
            },
            metadata = {
                'options': {
                    'withmpi': False,
                }
            },
        )

        self.report(f'Submitted job: {node_veloxchem}')
        self.report(f'Outputs: {results_veloxchem}')

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

        self.report(f'Updated ITP file with RESP charges stored as node {self.ctx.itp_with_resp}')

    def update_top_file(self):
        """Update the .top file to reference the new .itp file and correct molecule count."""
        self.ctx.top_updated = fnc.update_top_file(
            nmols=self.inputs.nmols,
            top_file=self.ctx.top,
            itp_file=self.ctx.itp_with_resp,
        )

        self.report(f"Updated .top file stored: {self.ctx.top_updated.filename} {self.ctx.top_updated}")

    def get_box_size(self):
        """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
        self.ctx.box_size = fnc.get_box_size(
            nmols=self.inputs.nmols,
            smiles_string=self.inputs.smiles_string
        )

        self.report(f"Calculated box edge length: {self.ctx.box_size.value:.2f} nm")

    def build_gro(self):
        self.report(f'Running GROMACS insert-molecules to create a box of {self.inputs.nmols.value} molecules... ')
        filename = 'system.gro'
        results_insert, node_insert = launch_shell_job(
            self.inputs.gmx_code,
            arguments=(
                f'insert-molecules -ci {{grofile}} -o {filename} -nmol {{nmols}} ' +
                '-try 1000 -box {box_vector} {box_vector} {box_vector}'
            ),
            nodes={
                'grofile': self.ctx.gro,
                'nmols': self.inputs.nmols,
                'box_vector': self.ctx.box_size
            },
            metadata={
                'options': {
                    'withmpi': False,
                }
            },
            outputs=[filename]
        )
        self.report(f'Submitted job: {node_insert}')
        self.report(f'Outputs: {results_insert}')

        self.ctx.system_gro = results_insert[filename.replace('.', '_')]

    def make_gromacs_minimization_input(self):
        """Generate a basic GROMACS minimization input file."""
        self.ctx.minimize_mdp = fnc.generate_gromacs_minimization_input(
            minimization_steps=self.inputs.gromacs_minimization_steps
        )
        # self.report(f'Generated minimization.mdp file: {self.ctx.minimize_mdp}')

    def submit_minimization_init(self):
        self.report('Running GROMACS minimization initialization...')
        results_grompp, node_grompp = launch_shell_job(
            self.inputs.gmx_code,
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o minimize.tpr',
            nodes={
                'mdpfile': self.ctx.minimize_mdp,
                'grofile': self.ctx.system_gro,
                'topfile': self.ctx.top_updated,
                'itpfile': self.ctx.itp_with_resp
            },
            metadata={
                'options': {
                    'withmpi': False,
                    'redirect_stderr': True,
                }
            },
            outputs=['mdout.mdp', 'minimize.tpr']
        )

        self.report(f'Submitted job: {node_grompp}')
        self.report(f'Outputs: {results_grompp}')

        self.ctx.tpr = results_grompp['minimize_tpr']

    def submit_minimization_run(self):
        self.report('Running GROMACS minimization mdrun...')
        # gmx_mpi mdrun -v -deffnm minimize
        results_mdrun, node_mdrun = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm minimize',
            nodes={
                'tprfile': self.ctx.tpr
            },
            metadata={
                'options': {
                    'withmpi': True,
                    'redirect_stderr': True,
                }
            },
            outputs=['minimize.gro']
        )

        self.report(f'Submitted job: {node_mdrun}')
        self.report(f'Outputs: {results_mdrun}')

        self.ctx.minimized_gro = results_mdrun['minimize_gro']

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
