"""Common base workchain for GROMACS-based simulations."""
import copy

from aiida import orm
from aiida.common import exceptions as exc
from aiida.engine import WorkChain
from aiida.plugins import CalculationFactory
from aiida_shell import launch_shell_job

from .utils import clean_workchain_calcs, create_metadata

BASENAME = 'aiida'
ShellJob = CalculationFactory('core.shell')

class GromacsBaseWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input(
            'reference_temperature', valid_type=orm.Float,
            required=False,
            help='The reference temperature in Kelvin for the simulation.'
        )
        spec.input(
            'time_step', valid_type=orm.Float,
            default=lambda: orm.Float(0.001),
            help='The MD time step in picoseconds.'
        )

        spec.input(
            'itp_file', valid_type=orm.SinglefileData,
            help='The ITP file containing the molecule topology.'
        )
        spec.input(
            'top_file', valid_type=orm.SinglefileData,
            help='The TOP file containing the system topology.'
        )
        spec.input(
            'gro_file', valid_type=orm.SinglefileData,
            help='The GRO file containing the atoms position/velocities.'
        )

        spec.input('gmx_code', valid_type=orm.AbstractCode, help='Code for running `gmx` or `gmx_mpi`.')
        spec.input(
            'gmx_code_local', valid_type=orm.AbstractCode,
            required=False,
            help='Code for running `gmx` or `gmx_mpi` locally for initialization/serial runs.'
        )

        spec.expose_inputs(
            ShellJob,
            namespace='shelljob',
            include=('metadata', ),
            namespace_options={
                'required': True, 'populate_defaults': False
            }
        )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )

        # ERRORS ############################################################################
        spec.exit_code(
            330, 'ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT',
            message='A GROMACS minimization subprocess calculation failed.'
        )
        spec.exit_code(
            335, 'ERROR_SUB_PROCESS_FAILED_GMX_GROMP',
            message='A GROMACS grompp subprocess calculation failed.'
        )

    def setup(self):
        """Setup context variables."""
        gmx_computer: orm.Computer = self.inputs.gmx_code.computer
        metadata_tpl = dict(self.inputs.shelljob.metadata)

        serial_mdata, parall_mdata = create_metadata(gmx_computer, metadata_tpl, report_func=self.report)

        gmx_remote = self.inputs.gmx_code
        gmx_local = self.inputs.gmx_code_local if 'gmx_code_local' in self.inputs else gmx_remote
        self.report(f'Using GROMACS <{gmx_remote.pk}> for remote execution.')
        self.ctx.gmx_code_local = gmx_local
        self.report(f'Using GROMACS <{gmx_local.pk}> for local execution.')

        self.ctx.gmx_code_local = gmx_local
        self.ctx.gmx_computer = self.inputs.gmx_code.computer
        self.ctx.gmx_serial_metadata = serial_mdata
        self.ctx.gmx_parall_metadata = parall_mdata

    def _submit_grompp_calc(
            self,
            mdp_file, gro_file, top_file, itp_file,
            filename: str = BASENAME,
            docname: str = '',
            with_mdout: bool = True
        ):
        """Submit a GROMACS grompp calculation."""
        self.report(f'Submitting GROMACS grompp {docname} calculation...')
        metadata = copy.deepcopy(self.ctx.gmx_serial_metadata)
        metadata['call_link_label'] = f'{docname}_grompp' if docname else 'grompp'

        output_tpr_fname=f'{filename}.tpr'

        outputs = [output_tpr_fname]
        if with_mdout:
            outputs.insert(0, 'mdout.mdp')

        _, node = launch_shell_job(
            self.ctx.gmx_code_local,
            arguments=(
                'grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o ' + output_tpr_fname
            ),
            nodes={
                'mdpfile': mdp_file,
                'grofile': gro_file,
                'topfile': top_file,
                'itpfile': itp_file,
            },
            metadata=metadata,
            outputs=outputs,
        )

        self.report(f'Submitted GROMACS grompp calculation: {node}')
        return node

    def _check_gromp_calc(self, gromp_calc, filename: str = BASENAME):
        """Check the GROMACS grompp calculation for errors."""
        if not gromp_calc.is_finished_ok:
            self.report('GROMACS grompp for minimization failed.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_GROMP

        try:
            tpr_file = gromp_calc.outputs[f'{filename}_tpr']
        except exc.NotExistentKeyError:
            self.report('GROMP for minimization failed')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        return tpr_file

    def _on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()
        self._on_terminated()
