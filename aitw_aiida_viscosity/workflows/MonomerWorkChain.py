"""Implementation of the WorkChain for AITW viscosity calculation."""
from aiida import orm
from aiida.engine import ToContext, WorkChain, if_, while_
from aiida.plugins import CalculationFactory, SchedulerFactory
from aiida_shell import launch_shell_job

from . import functions as fnc
from .GromacsBase import GromacsBaseWorkChain
from .GromacsEquilibration import GromacsEquilibrationWorkChain
from .GromacsNEMD import GromacsNEMDWorkChain
from .RespCharges import RespChargesWorkChain
from .utils import clean_workchain_calcs, create_metadata

BASENAME = 'aiida'
DIRECT_SCHEDULER = SchedulerFactory('core.direct')
ShellJob = CalculationFactory('core.shell')

def validate_deform_velocities(node: orm.List, _):
    """Validate that all deformation velocities are positive."""
    for value in node.get_list():
        if value <= 0.0:
            return 'All deformation velocities must be positive.'

class MonomerWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.expose_inputs(
            RespChargesWorkChain,
            exclude=('clean_workdir', 'metadata'),
        )
        spec.expose_inputs(
            GromacsBaseWorkChain,
            exclude=('clean_workdir', 'itp_file', 'top_file', 'gro_file'),
        )
        spec.expose_inputs(
            GromacsEquilibrationWorkChain,
            include=('nmols', 'gromacs_minimization_steps', 'gromacs_equilibration_steps'),
        )

        spec.input('num_steps', valid_type=orm.Int, help='The number of MD steps to run in the NEMD simulation.')
        spec.input(
            'averaging_start_time', valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help='The time in picoseconds to skip before starting the averaging of the pressure tensor.'
        )
        spec.input(
            'deform_velocities', valid_type=orm.List,
            default=lambda: orm.List(list=[0.005, 0.002, 0.05, 0.02, 0.01, 0.1, 0.2]),
            validator=validate_deform_velocities,
            help=(
                'List of deformation velocities to use in the NEMD simulations. '
                'See https://manual.gromacs.org/current/user-guide/mdp-options.html#mdp-deform for details.'
            )
        )

        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.validate_inputs,
            cls.setup,

            cls.submit_resp_charges,
            cls.inspect_resp_charges,

            cls.run_resp_injection,
            cls.update_top_file,
            cls.get_box_size,

            cls.submit_gromacs_equilibration,
            cls.inspect_gromacs_equilibration,

            cls.make_nemd_inputs,
            # cls.submit_nemd_init,
            if_(cls.should_do_alltogheter)(
                cls.submit_nemd_run_parallel,
            ).else_(
                while_(cls.do_nemd_serial)(
                    cls.submit_nemd_run_serial,
                )
            ),
            cls.inspect_nemd,

            cls.submit_energy_parallel,
            cls.inspect_energy,

            cls.collect_pressure_averages,
            cls.compute_viscosity_data,
            cls.fit_viscosity,
        )

        # OUTPUTS ############################################################################
        spec.expose_outputs(RespChargesWorkChain, exclude=('pdb', ),)
        spec.expose_outputs(GromacsEquilibrationWorkChain)

        spec.output(
            'itp_with_resp', valid_type=orm.SinglefileData,
            help='The .itp file updated with RESP charges from the PDB file.'
        )
        spec.output(
            'top_updated', valid_type=orm.SinglefileData,
            help='The updated .top file referencing the new .itp file and correct molecule count.'
        )

        spec.output(
            'viscosity_data',
            valid_type=orm.ArrayData,
            help=(
                'ArrayData containing `deformation_velocities`, `pressure_averages`, `shear_rates`, '
                'and `viscosities` arrays.'
            )
        )

        spec.output_namespace(
            'nemd',
            valid_type=orm.SinglefileData,
            dynamic=True,
            help='NEMD .edr output files for each deformation velocity.'
        )
        spec.output(
            'eta_N', valid_type=orm.Float, required=False,
            help='Newtonian viscosity in millipascal-seconds'
        )
        spec.output(
            'sigma_E', valid_type=orm.Float, required=False,
            help='Eyring stress in millipascals'
        )

        # ERRORS ############################################################################
        spec.exit_code(
            200, 'ERROR_INVALID_AVERAGING_START_TIME',
            message='The averaging start time is greater than or equal to the total simulation time.'
        )
        spec.exit_code(
            344, 'ERROR_SUB_PROCESS_FAILED_GMX_ENERGY',
            message='A GROMACS energy subprocess calculation failed.'
        )
        spec.exit_code(
            400, 'ERROR_RESP_CHARGES_FAILED',
            message='The RESP charges WorkChain did not finish successfully.'
        )
        spec.exit_code(
            410, 'ERROR_GROMACS_EQUILIBRATION_FAILED',
            message='The GROMACS equilibration WorkChain did not finish successfully.'
        )
        spec.exit_code(
            420, 'ERROR_GROMACS_NEMD_FAILED',
            message='The GROMACS NEMD WorkChain did not finish successfully.'
        )

    def validate_inputs(self):
        """Perform validation on the inputs that require knowledge of multiple inputs."""
        total_sim_time = self.inputs.num_steps.value * self.inputs.time_step.value
        if self.inputs.averaging_start_time.value >= total_sim_time:
            self.report(
                f'Averaging start time {self.inputs.averaging_start_time.value} ps is '
                f'greater than or equal to total simulation time {total_sim_time} ps.'
            )
            return self.exit_codes.ERROR_INVALID_AVERAGING_START_TIME

    def _create_metadata(self):
        """Setup the metadata templates for the calculations."""
        gmx_computer: orm.Computer = self.inputs.gmx_code.computer
        metadata_tpl = dict(self.inputs.shelljob.metadata)

        serial_mdata, parall_mdata = create_metadata(gmx_computer, metadata_tpl, report_func=self.report)

        self.ctx.gmx_serial_metadata = serial_mdata
        self.ctx.gmx_parall_metadata = parall_mdata

    def setup(self):
        """Setup context variables."""
        # Use remote code if local code not provided
        self._create_metadata()

        gmx_remote = self.inputs.gmx_code
        gmx_local = self.inputs.gmx_code_local if 'gmx_code_local' in self.inputs else gmx_remote
        self.report(f'Using GROMACS <{gmx_remote.pk}> for remote execution.')
        self.ctx.gmx_code_local = gmx_local
        self.report(f'Using GROMACS <{gmx_local.pk}> for local execution.')

        self.ctx.gmx_code_local = gmx_local

        gmx_computer: orm.Computer = self.inputs.gmx_code.computer
        gmx_sched = gmx_computer.get_scheduler()
        self.ctx.gmx_scheduler = gmx_sched

    def submit_resp_charges(self):
        """Submit the RESP charges WorkChain to compute RESP charges for the molecule."""
        self.report('Submitting RESP charges WorkChain...')

        inputs = self.exposed_inputs(RespChargesWorkChain)
        inputs['clean_workdir'] = self.inputs.clean_workdir

        running = self.submit(RespChargesWorkChain, **inputs)

        self.report(f'Submitted RESP charges WorkChain: {running.pk}')

        return ToContext(resp_charges_wc=running)

    def inspect_resp_charges(self):
        """Inspect the output of the RESP charges WorkChain."""
        workchain = self.ctx.resp_charges_wc
        if not workchain.is_finished_ok:
            self.report('RESP charges WorkChain failed.')
            return self.exit_codes.ERROR_RESP_CHARGES_FAILED

        self.ctx.pdb = workchain.outputs.pdb
        for key in ('itp', 'top', 'gro'):
            self.ctx[key] = workchain.outputs.acpype[key]

        self.out_many(
            self.exposed_outputs(workchain, RespChargesWorkChain)
        )

    def run_resp_injection(self):
        """
        Inject RESP charges from a PDB file into an ITP file.
        Assumes self.ctx.pdb and self.ctx.itp are SinglefileData nodes.
        """
        itp_with_resp = fnc.run_resp_injection(
            pdb_file=self.ctx.pdb,
            itp_file=self.ctx.itp
        )
        self.ctx.itp_with_resp = itp_with_resp
        self.report(f'Updated ITP file with RESP charges stored as node {itp_with_resp}')
        self.out('itp_with_resp', itp_with_resp)

    def update_top_file(self):
        """Update the .top file to reference the new .itp file and correct molecule count."""
        top_updated = fnc.update_top_file(
            nmols=self.inputs.nmols,
            top_file=self.ctx.top,
            itp_file=self.ctx.itp_with_resp,
        )

        self.report(f"Updated .top file stored: {top_updated.filename} {top_updated}")
        self.ctx.top_updated = top_updated
        self.out('top_updated', top_updated)

    def get_box_size(self):
        """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
        box_size = fnc.get_box_size(
            nmols=self.inputs.nmols,
            smiles_string=self.inputs.smiles_string
        )

        self.report(f"Calculated box edge length: {box_size.value:.2f} nm")
        self.ctx.box_size = box_size

    def submit_gromacs_equilibration(self):
        """Submit the GROMACS equilibration WorkChain to equilibrate the system."""
        self.report('Submitting GROMACS equilibration WorkChain...')

        inputs = self.exposed_inputs(GromacsBaseWorkChain)
        inputs['clean_workdir'] = self.inputs.clean_workdir
        inputs['nmols'] = self.inputs.nmols
        inputs['box_size'] = self.ctx.box_size
        inputs['gro_file'] = self.ctx.gro
        inputs['top_file'] = self.ctx.top_updated
        inputs['itp_file'] = self.ctx.itp_with_resp
        inputs['gromacs_minimization_steps'] = self.inputs.gromacs_minimization_steps
        inputs['gromacs_equilibration_steps'] = self.inputs.gromacs_equilibration_steps

        running = self.submit(GromacsEquilibrationWorkChain, **inputs)

        self.report(f'Submitted GROMACS equilibration WorkChain: {running.pk}')

        return ToContext(gromacs_equil_wc=running)

    def inspect_gromacs_equilibration(self):
        """Inspect the output of the GROMACS equilibration WorkChain."""
        workchain = self.ctx.gromacs_equil_wc
        if not workchain.is_finished_ok:
            self.report('GROMACS equilibration WorkChain failed.')
            return self.exit_codes.ERROR_GROMACS_EQUILIBRATION_FAILED

        for key in workchain.outputs:
            self.ctx[key] = workchain.outputs[key]

        self.out_many(
            self.exposed_outputs(workchain, GromacsEquilibrationWorkChain)
        )

    def make_nemd_inputs(self):
        """Prepare input files for NEMD simulations with different deformation velocities."""
        self.report('Preparing NEMD input files for each deformation velocity...')
        self.report(str(self.inputs.deform_velocities))

        self.ctx.str_defvel = {defvel: fnc.string_safe_float(defvel) for defvel in self.inputs.deform_velocities}

        self.ctx.mdp_files = fnc.generate_gromacs_deform_vel_inputs(
            nsteps=self.inputs.num_steps,
            time_step=self.inputs.time_step,
            ref_t=self.inputs.reference_temperature,
            deform_velocities=self.inputs.deform_velocities
        )

    def should_do_alltogheter(self) -> bool:
        """Check if all deformation velocity can be in parallel runs."""
        sched = self.ctx.gmx_scheduler
        if isinstance(sched, DIRECT_SCHEDULER):
            self.report('Direct scheduler does not support running multiple jobs.')
            self.ctx.nemd_serial_cnt = 0
            return False
        return True

    def submit_nemd_run_parallel(self):
        """Submit all NEMD runs as parallel/concurrent jobs."""
        self.report('Submitting GROMACS NEMD runs as parallel jobs...')

        inputs = self.exposed_inputs(GromacsBaseWorkChain)
        inputs['clean_workdir'] = self.inputs.clean_workdir
        inputs['num_steps'] = self.inputs.num_steps
        inputs['gro_file'] = self.ctx.equilibrated_gro
        inputs['top_file'] = self.ctx.top_updated
        inputs['itp_file'] = self.ctx.itp_with_resp
        for defvel, str_defvel in self.ctx.str_defvel.items():
            inputs['mdp_file'] = self.ctx.mdp_files[f'mdp_{str_defvel}']
            node = self.submit(GromacsNEMDWorkChain, **inputs)

            self.report(f'Submitted job for NEMD run for deformation velocity {defvel}: {node}')

            self.to_context(**{f'nemd_{str_defvel}': node})

    def do_nemd_serial(self) -> bool:
        """Check if there are remaining deformation velocity to run in serial."""
        return self.ctx.nemd_serial_cnt < len(self.inputs.deform_velocities)

    def submit_nemd_run_serial(self):
        """Submit the next NEMD run in serial."""
        defvel = self.inputs.deform_velocities[self.ctx.nemd_serial_cnt]
        str_defvel = self.ctx.str_defvel[defvel]
        self.ctx.nemd_serial_cnt += 1

        inputs = self.exposed_inputs(GromacsBaseWorkChain)
        inputs['clean_workdir'] = self.inputs.clean_workdir
        inputs['num_steps'] = self.inputs.num_steps
        inputs['mdp_file'] = self.ctx.mdp_files[f'mdp_{str_defvel}']
        inputs['gro_file'] = self.ctx.equilibrated_gro
        inputs['top_file'] = self.ctx.top_updated
        inputs['itp_file'] = self.ctx.itp_with_resp

        node = self.submit(GromacsNEMDWorkChain, **inputs)

        self.report(f'Submitted job for NEMD run for deformation velocity {defvel}: {node}')

        return ToContext(**{f'nemd_{str_defvel}': node})

    def inspect_nemd(self):
        """Collect .edr files from the NEMD parallel run."""
        self.report('Collecting .edr files from NEMD runs...')

        calc_map = {defvel: self.ctx[f'nemd_{str_defvel}'] for defvel, str_defvel in self.ctx.str_defvel.items()}
        failed = [defvel for defvel, calc in calc_map.items() if not calc.is_finished_ok]
        if failed:
            self.report(f'NEMD runs for deformation velocities {failed} did not finish successfully.')
            return self.exit_codes.ERROR_GROMACS_NEMD_FAILED

        edr_files = {}
        edr_outputs = {}
        self.ctx.edr_outputs = {}
        for defvel, calc in calc_map.items():
            str_defvel = self.ctx.str_defvel[defvel]
            edr_file = calc.outputs['edr_file']
            edr_files[defvel] = edr_file
            edr_outputs[f'edr_{str_defvel}'] = edr_file
            self.report(f'Collected .edr file for deformation velocity {defvel}')

        self.ctx.edr_files = edr_files
        self.out('nemd', edr_outputs)

    def submit_energy_parallel(self):
        """Run `gmx energy` to extract pressure data from each EDR file."""
        self.report('Submitting GROMACS energy extraction runs for each deformation velocity...')
        self.ctx.pressure_xvg = {}
        for defvel, edr_file in self.ctx.edr_files.items():
            str_defvel = self.ctx.str_defvel[defvel]
            _, node = launch_shell_job(
                self.ctx.gmx_code_local,
                arguments=f'energy -f {{edr}} -o {BASENAME}.xvg -b {{start_time}}',
                nodes={
                    'edr': edr_file,
                    'start_time': self.inputs.averaging_start_time,
                    # Select term Pres-XY as the quantity to be analyzed, confirm with 0
                    'stdin': orm.SinglefileData.from_string('Pres-XY\n0\n', filename='stdin'),
                },
                outputs=[f'{BASENAME}.xvg'],
                metadata={
                    'call_link_label': f'gromacs_energy',
                    'options': {
                        'resources': {'num_machines': 1},
                        'withmpi': False,
                        'redirect_stderr': True,
                        'filename_stdin': 'stdin'
                    }
                },
                submit=True
            )

            self.report(f'Submitted job: {node}')
            self.to_context(**{f'energy_{str_defvel}': node})

    def inspect_energy(self):
        """Collect .xvg files from the energy extraction runs."""
        calc_map = {defvel: self.ctx[f'energy_{str_defvel}'] for defvel, str_defvel in self.ctx.str_defvel.items()}

        failed = [defvel for defvel, calc in calc_map.items() if not calc.is_finished_ok]
        if failed:
            self.report(f'Energy extraction runs for deformation velocities {failed} did not finish successfully.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_ENERGY

        self.ctx.pressure_xvg = {}
        for defvel, calc in calc_map.items():
            xvg_file = calc.outputs['aiida_xvg']
            self.ctx.pressure_xvg[defvel] = xvg_file

    def collect_pressure_averages(self):
        """Collect average pressures from the postprocessing workchain."""
        pressures = {}
        has_negatives = False
        for defvel in self.inputs.deform_velocities:
            str_defvel = self.ctx.str_defvel[defvel]
            xvg_file = self.ctx.pressure_xvg[defvel]
            avg_pressure = fnc.extract_pressure_from_xvg(xvg_file)
            pressures[f'pressure_{str_defvel}'] = avg_pressure
            if avg_pressure.value < 0:
                has_negatives = True
            self.report(f"Average pressure for deformation velocity {defvel}: {avg_pressure.value} bar")

        if has_negatives:
            self.report('WARNING: Negative pressures detected, results may be unphysical!')

        self.ctx.pressures = fnc.join_pressure_results(self.inputs.deform_velocities, **pressures)

    def compute_viscosity_data(self):
        """Compute average pressures, shear rates, and viscosities."""
        res = fnc.compute_viscosities(
            deformation_velocities=self.inputs.deform_velocities,
            pressures=self.ctx.pressures,
            box_length=self.ctx.equilibrated_box_length_nm
        )

        self.ctx.viscosity_data = res
        self.out('viscosity_data', res)

    def fit_viscosity(self):
        """Fit viscosity data to the Eyring model."""
        try:
            dct = fnc.fit_viscosity(self.ctx.viscosity_data)
        except ValueError:
            self.report('Fitting viscosity data to Eyring model failed.')
        else:
            eta_N = dct['eta_N']
            sigma_E = dct['sigma_E']
            self.out('eta_N', eta_N)
            self.out('sigma_E', sigma_E)
            self.report(
                f'Fitted viscosity data to Eyring model: eta_N={eta_N.value:.3e}, sigma_E={sigma_E.value:.3e}'
            )

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = clean_workchain_calcs(self.node)

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
