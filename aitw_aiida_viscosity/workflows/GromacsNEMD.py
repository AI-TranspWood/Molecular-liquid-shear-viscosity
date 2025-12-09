"""Implementation of the WorkChain for AITW viscosity calculation."""
import copy

from aiida import orm
from aiida.common import exceptions as exc
from aiida.engine import ToContext
from aiida.plugins import CalculationFactory
from aiida_shell import launch_shell_job

from .GromacsBase import GromacsBaseWorkChain

BASENAME = 'aiida'
ShellJob = CalculationFactory('core.shell')

class GromacsNEMDWorkChain(GromacsBaseWorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input('num_steps', valid_type=orm.Int, help='The number of MD steps to run in the NEMD simulation.')

        spec.input(
            'mdp_file', valid_type=orm.SinglefileData,
            help='The MDB file containing the calculation parameters.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            cls.submit_nemd_init,
            cls.submit_nemd_run,
            cls.inspect_nemd,
        )

        # OUTPUTS ############################################################################
        spec.output(
            'edr_file', valid_type=orm.SinglefileData,
            help='The .edr file containing the energy data from the equilibration run.'
        )

        # ERRORS ############################################################################
        spec.exit_code(
            343, 'ERROR_SUB_PROCESS_FAILED_GMX_NEMD',
            message='A GROMACS NEMD subprocess calculation failed.'
        )

    def submit_nemd_init(self):
        """Submit GROMACS grompp for each deformation velocity to generate .tpr files."""
        node = self._submit_grompp_calc(
            mdp_file=self.inputs.mdp_file,
            gro_file=self.inputs.gro_file,
            top_file=self.inputs.top_file,
            itp_file=self.inputs.itp_file,
            docname='nemd',
            with_mdout=False
        )
        return ToContext(grompp_nemd_init=node)

    def submit_nemd_run(self):
        """Submit all NEMD runs as parallel/concurrent jobs."""
        tpr_file = self._check_gromp_calc(self.ctx.grompp_nemd_init)

        self.report('Submitting GROMACS NEMD runs as parallel jobs...')
        metadata = copy.deepcopy(self.ctx.gmx_parall_metadata)
        metadata['call_link_label'] = 'nemd_mdrun'

        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tpr_file} -deffnm ' + BASENAME,
            nodes={
                'tpr_file': tpr_file,
            },
            metadata=metadata,
            outputs=[f'{BASENAME}.edr'],
            submit=True
        )

        self.report(f'Submitted job for NEMD run: {node}')
        return ToContext(nemd_calc=node)
        # self.to_context(**{f'nemd_{str_defvel}': node})

    def inspect_nemd(self):
        """Collect .edr files from the NEMD parallel run."""
        calc = self.ctx.nemd_calc
        if not calc.is_finished_ok:
            self.report('NEMD run did not finish successfully.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_NEMD

        try:
            edr_file = calc.outputs['aiida_edr']
        except exc.NotExistentKeyError:
            self.report('NEMD run is missing the .edr output file.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_MISSING_OUTPUT

        self.out('edr_file', edr_file)
