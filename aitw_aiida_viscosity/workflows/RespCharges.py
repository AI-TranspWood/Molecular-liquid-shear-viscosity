"""Compute the Restrained Electrostatic Potential (RESP) starting from a smiles string"""
from aiida import orm
from aiida.common import exceptions as exc
from aiida.engine import ToContext, WorkChain
from aiida.plugins import CalculationFactory
from aiida_shell import launch_shell_job

from . import functions as fnc
from .utils import clean_workchain_calcs

BASENAME = 'aiida'
ShellJob = CalculationFactory('core.shell')

class RespChargesWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input('num_steps', valid_type=orm.Int, help='The number of MD steps to run in the NEMD simulation.')
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
            'veloxchem_basis', valid_type=orm.Str,
            default=lambda: orm.Str('6-31G*'),
            help=(
                'The basis set to use in the VeloxChem calculation. This should be 6-31G* for RESP partial charges '
                'with the GAFF and GAFF2 force fields. '
                'See https://veloxchem.org/docs/basis_sets.html for details and available basis sets.'
            )
        )

        spec.input('acpype_code', valid_type=orm.AbstractCode, help='Code for running the `acpype` program.')
        spec.input('obabel_code', valid_type=orm.AbstractCode, help='Code for running the `obabel` program.')
        spec.input('veloxchem_code', valid_type=orm.AbstractCode, help='Code for python with `veloxchem` installed.')

        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,

            cls.submit_acpype,
            cls.inspect_acpype,

            cls.submit_obabel,
            cls.inspect_obabel,

            cls.make_veloxchem_input,
            cls.submit_veloxchem,
            cls.inspect_veloxchem,
        )

        # OUTPUTS ############################################################################
        spec.output(
            'xyz', valid_type=orm.SinglefileData,
            help='The XYZ file of the molecule generated from ACPYPE bi the SMILES code.'
        )
        spec.output(
            'pdb', valid_type=orm.SinglefileData,
            help='The PDB file of the molecule with RESP charges computed by VeloxChem.'
        )

        spec.output_namespace(
            'acpype',
            valid_type=orm.SinglefileData,
            dynamic=True,
            help='ACPYPE output files.'
        )

        # ERRORS ############################################################################
        spec.exit_code(
            300, 'ERROR_ACPYPE_FAILED',
            message='The ACPYPE calculation did not finish successfully.'
        )
        spec.exit_code(
            305, 'ERROR_ACPYPE_MISSING_OUTPUT',
            message='ACPYPE did not produce all expected output files.'
        )
        spec.exit_code(
            310, 'ERROR_OBABEL_FAILED',
            message='The Open Babel calculation did not finish successfully.'
        )
        spec.exit_code(
            315, 'ERROR_OBELAB_MISSING_OUTPUT',
            message='Open Babel did not produce the expected output file.'
        )
        spec.exit_code(
            320, 'ERROR_VELOXCHEM_FAILED',
            message='The VeloxChem calculation did not finish successfully.'
        )
        spec.exit_code(
            325, 'ERROR_VELOXCHEM_MISSING_OUTPUT',
            message='VeloxChem did not produce the expected output file.'
        )

    def setup(self):
        """Setup context variables."""
        # Use remote code if local code not provided
        self.ctx.smiles_string = self.inputs.smiles_string.value
        self.ctx.ff = self.inputs.force_field.value

    def submit_acpype(self):
        """Submit acpype and obabel calculations to generate initial structure and parameters."""
        self.report('Running acpype through aiida-shell...')
        _, node = launch_shell_job(
            self.inputs.acpype_code,
            arguments=f'-i {{smiles}} -n 0 -c bcc -q sqm -b {BASENAME} -a {{ff}} -s 108000',
            nodes={
                'smiles': self.inputs.smiles_string,
                'ff': self.inputs.force_field
            },
            metadata={
                'call_link_label': 'acpype',
                'options': {
                    'withmpi': False,
                },
            },
            outputs=[f'{BASENAME}.acpype'],
            submit=True
        )
        self.report(f'Submitted job: {node}')
        # self.report(f'Outputs: {results_acpype}')
        return ToContext(acpype_calc=node)

    def inspect_acpype(self):
        """Inspect the output of the ACPYPE calculation and store relevant files in the context."""
        # Expected filename patterns
        target_suffixes = {
            'gro': 'GMX.gro',
            'itp': 'GMX.itp',
            'top': 'GMX.top',
            'pdb': 'NEW.pdb',
        }

        calc = self.ctx.acpype_calc
        if not calc.is_finished_ok:
            self.report('ACPYPE calculation failed.')
            return self.exit_codes.ERROR_ACPYPE_FAILED
        res = calc.outputs

        missing = []
        files = {}
        folder = res[f'{BASENAME}_acpype']
        for key, suffix in target_suffixes.items():
            try:
                file = fnc.extract_files_suffix(folder, orm.Str(suffix).store())
            except ValueError:
                missing.append(key)
                continue
            files[key] = file
            setattr(self.ctx, key, file)
        if missing:
            self.report(f'Missing expected output files from ACPYPE: {missing}')
            return self.exit_codes.ERROR_ACPYPE_MISSING_OUTPUT
        self.out('acpype', files)

    def submit_obabel(self):
        """Convert PDB file to XYZ using Open Babel."""
        self.report('Running obabel through aiida-shell...')
        out_filename = f'{BASENAME}.xyz'
        _, node = launch_shell_job(
            self.inputs.obabel_code,
            arguments = f'{{pdbfile}} -O {out_filename}',
            nodes={
                'pdbfile': self.ctx.pdb
            },
            metadata={
                'call_link_label': 'obabel',
                'options': {
                    'withmpi': False,
                    'redirect_stderr': True,
                }
            },
            outputs=[out_filename],
            submit=True
        )

        self.report(f'Submitted job: {node}')

        return ToContext(obabel_calc=node)

    def inspect_obabel(self):
        """Inspect the output of the Open Babel calculation."""
        calc = self.ctx.obabel_calc
        if not calc.is_finished_ok:
            self.report('Open Babel calculation failed.')
            return self.exit_codes.ERROR_OBABEL_FAILED

        try:
            xyz_file = calc.outputs[f'{BASENAME}_xyz']
        except exc.NotExistentKeyError:
            self.report('Open Babel did not produce the expected XYZ output file.')
            return self.exit_codes.ERROR_OBELAB_MISSING_OUTPUT
        self.ctx.xyz = xyz_file
        self.out('xyz', xyz_file)

    def make_veloxchem_input(self):
        """Prepare input files for VeloxChem calculation."""
        self.ctx.veloxchem_input = fnc.generate_veloxchem_input(self.inputs.veloxchem_basis)

    def submit_veloxchem(self):
        """Submit a VeloxChem calculation to compute RESP charges and store the resulting PDB file"""
        self.report('Running veloxchem through aiida-shell...')
        _, node = launch_shell_job(
            self.inputs.veloxchem_code,
            arguments = '{script_file} {xyzfile}',
            nodes={
                'script_file': self.ctx.veloxchem_input,
                'xyzfile': self.ctx.xyz
            },
            metadata = {
                'call_link_label': 'veloxchem',
                'options': {
                    'withmpi': False,
                }
            },
            submit=True,
        )

        self.report(f'Submitted job: {node}')
        return ToContext(veloxchem_calc=node)

    def inspect_veloxchem(self):
        """Extract the PDB file with RESP charges from the VeloxChem calculation."""
        calc = self.ctx.veloxchem_calc
        if not calc.is_finished_ok:
            self.report('VeloxChem calculation failed.')
            return self.exit_codes.ERROR_VELOXCHEM_FAILED

        try:
            pdb_file = calc.outputs['stdout']
        except exc.NotExistentKeyError:
            self.report('VeloxChem did not produce the expected PDB output file.')
            return self.exit_codes.ERROR_VELOXCHEM_MISSING_OUTPUT
        self.out('pdb', pdb_file)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
