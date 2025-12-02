"""WorkChain to perform a GROMACS box-creation, minimization and equilibration runs"""
import copy

from aiida import orm
from aiida.common import exceptions as exc
from aiida.engine import ToContext
from aiida.plugins import CalculationFactory
from aiida_shell import launch_shell_job

from . import functions as fnc
from .GromacsBase import GromacsBaseWorkChain

BASENAME = 'aiida'
ShellJob = CalculationFactory('core.shell')

class GromacsEquilibrationWorkChain(GromacsBaseWorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input(
            'nmols', valid_type=orm.Int,
            default=lambda: orm.Int(1000),
            help='The number of molecules to insert into the simulation box.'
        )
        spec.input(
            'box_size', valid_type=orm.Float,
            help='The edge length of the box in nanometers.'
        )

        spec.input(
            'gromacs_minimization_steps', valid_type=orm.Int,
            default=lambda: orm.Int(5000),
            help='The number of steps to use in the GROMACS minimization.'
        )
        spec.input(
            'gromacs_equilibration_steps', valid_type=orm.Int,
            default=lambda: orm.Int(500000),
            help='The number of steps to use in the GROMACS equilibration.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,

            cls.submit_insertmol,
            cls.inspect_insertmol,

            cls.make_gromacs_minimization_input,
            cls.submit_minimization_init,
            cls.submit_minimization_run,
            cls.inspect_minimization,

            cls.make_gromacs_equilibration_input,
            cls.submit_equilibration_init,
            cls.submit_equilibration_run,
            cls.inspect_equilibration,

            cls.extract_equilibrated_box_length,
        )

        # OUTPUTS ############################################################################
        spec.output(
            'equilibrated_box_length_nm',
            valid_type=orm.Float,
            help='The edge length of the equilibrated cubic simulation box in nanometers.'
        )
        spec.output(
            'system_gro', valid_type=orm.SinglefileData,
            help='The .gro file of the full simulation box after inserting all molecules.'
        )
        spec.output(
            'equilibrated_gro', valid_type=orm.SinglefileData,
            help='The equilibrated .gro file after minimization and equilibration.'
        )

        # ERRORS ############################################################################
        spec.exit_code(
            340, 'ERROR_SUB_PROCESS_FAILED_GMX_INSERTMOL',
            message='A GROMACS insert-molecules subprocess calculation failed.'
        )
        spec.exit_code(
            341, 'ERROR_SUB_PROCESS_FAILED_GMX_MINIMIZATION',
            message='A GROMACS minimization subprocess calculation failed.'
        )
        spec.exit_code(
            342, 'ERROR_SUB_PROCESS_FAILED_GMX_EQUILIBRATION',
            message='A GROMACS equilibration subprocess calculation failed.'
        )

    def setup(self):
        """Setup context variables."""
        self._create_metadata()
        self._gmx_setup()

        self.ctx.nmols = self.inputs.nmols.value
        self.ctx.box_size = self.inputs.box_size

    def get_box_size(self):
        """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
        box_size = fnc.get_box_size(
            nmols=self.inputs.nmols,
            smiles_string=self.inputs.smiles_string
        )

        self.report(f"Calculated box edge length: {box_size.value:.2f} nm")
        self.ctx.box_size = box_size

    def submit_insertmol(self):
        self.report(f'Running GROMACS insert-molecules to create a box of {self.inputs.nmols.value} molecules... ')
        filename = f'{BASENAME}.gro'
        metadata = copy.deepcopy(self.ctx.gmx_serial_metadata)
        metadata['call_link_label'] = 'insert_molecules'
        _, node = launch_shell_job(
            self.ctx.gmx_code_local,
            arguments=(
                f'insert-molecules -ci {{grofile}} -o {filename} -nmol {{nmols}} ' +
                '-try 1000 -box {box_vector} {box_vector} {box_vector}'
            ),
            nodes={
                'grofile': self.inputs.gro_file,
                'nmols': self.inputs.nmols,
                'box_vector': self.ctx.box_size
            },
            metadata=metadata,
            outputs=[filename],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(insertmol_calc=node)

    def inspect_insertmol(self):
        """Inspect the output of the insert-molecules calculation."""
        calc = self.ctx.insertmol_calc
        if not calc.is_finished_ok:
            self.report('GROMACS insert-molecules calculation failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_INSERTMOL

        try:
            system_gro = calc.outputs[f'{BASENAME}_gro']
        except exc.NotExistentKeyError:
            self.report('GROMACS insert-molecules did not produce the expected output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.ctx.system_gro = system_gro
        self.out('system_gro', system_gro)

    def make_gromacs_minimization_input(self):
        """Generate a basic GROMACS minimization input file."""
        self.ctx.minimize_mdp = fnc.generate_gromacs_minimization_input(
            minimization_steps=self.inputs.gromacs_minimization_steps
        )

    def submit_minimization_init(self):
        """Initialize GROMACS minimization run to generate .tpr file."""
        node = self._submit_grompp_calc(
            mdp_file=self.ctx.minimize_mdp,
            gro_file=self.ctx.system_gro,
            top_file=self.inputs.top_file,
            itp_file=self.inputs.itp_file,
            filename='minimize',
            docname='minimization'
        )

        return ToContext(gromp_minimize_calc=node)

    def submit_minimization_run(self):
        """Run GROMACS minimization mdrun."""
        tpr_file = self._check_gromp_calc(self.ctx.gromp_minimize_calc, filename='minimize')

        self.report('Running GROMACS minimization mdrun...')
        metadata = copy.deepcopy(self.ctx.gmx_parall_metadata)
        metadata['call_link_label'] = 'minimization_mdrun'
        # gmx_mpi mdrun -v -deffnm minimize
        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm minimize',
            nodes={
                'tprfile': tpr_file
            },
            metadata=metadata,
            outputs=['minimize.gro'],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(minimize_calc=node)

    def inspect_minimization(self):
        """Inspect the output of the minimization calculation."""
        calc = self.ctx.minimize_calc
        if not calc.is_finished_ok:
            self.report('GROMACS minimization calculation failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MINIMIZATION

        try:
            minimized_gro = calc.outputs['minimize_gro']
        except exc.NotExistentKeyError:
            self.report('GROMACS minimization did not produce the expected output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.ctx.minimized_gro = minimized_gro

    def make_gromacs_equilibration_input(self):
        """Generate a basic GROMACS equilibration input file."""
        self.ctx.equilibrate_mdp = fnc.generate_gromacs_equilibration_input(
            num_steps=self.inputs.gromacs_equilibration_steps,
            time_step=self.inputs.time_step,
            reference_temperature=self.inputs.reference_temperature
        )

    def submit_equilibration_init(self):
        """Initialize GROMACS equilibration run to generate .tpr file."""
        node = self._submit_grompp_calc(
            mdp_file=self.ctx.equilibrate_mdp,
            gro_file=self.ctx.minimized_gro,
            top_file=self.inputs.top_file,
            itp_file=self.inputs.itp_file,
            filename='equilibrate',
            docname='equilibration'
        )

        return ToContext(gromp_equilibrate_calc=node)

    def submit_equilibration_run(self):
        """Run GROMACS equilibration mdrun."""
        tpr_file = self._check_gromp_calc(self.ctx.gromp_equilibrate_calc, filename='equilibrate')

        self.report('Running GROMACS equilibration run MDRUN...')
        metadata = copy.deepcopy(self.ctx.gmx_parall_metadata)
        metadata['call_link_label'] = 'equilibration_mdrun'
        out_filename = 'equilibrate.gro'
        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm equilibrate',
            nodes={
                'tprfile': tpr_file
            },
            metadata=metadata,
            outputs=[out_filename],
            submit=True
        )
        self.report(f'Submitted job: {node}')

        return ToContext(equilibrate_calc=node)

    def inspect_equilibration(self):
        """Inspect the output of the equilibration calculation."""
        calc = self.ctx.equilibrate_calc
        if not calc.is_finished_ok:
            self.report('GROMACS equilibration calculation failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_EQUILIBRATION
        try:
            equilibrated_gro = calc.outputs['equilibrate_gro']
        except exc.NotExistentKeyError:
            self.report('GROMACS equilibration did not produce the expected output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.ctx.equilibrated_gro = equilibrated_gro
        self.out('equilibrated_gro', equilibrated_gro)

    def extract_equilibrated_box_length(self):
        """Extract box length from the equilibrated .gro file."""
        box_length_nm = fnc.extract_box_length(self.ctx.equilibrated_gro)
        self.report(f"Box length extracted: {box_length_nm.value} nm")
        self.out('equilibrated_box_length_nm', box_length_nm)
