"""Implementation of the WorkChain for AITW viscosity calculation."""
from glob import glob
import os

from aiida import orm
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida.orm import load_node
from aiida.plugins import SchedulerFactory
from aiida_shell import launch_shell_job

from . import functions as fnc
from .NemdParallelWorkChain import NemdParallelWorkChain
from .PostprocessPressureWorkChain import PostprocessPressureWorkChain

DIRECT_SCHEDULER = SchedulerFactory('core.direct')

def wrap_file(folder, filename):
    with folder.open(filename, 'rb') as f:
        return orm.SinglefileData(f, filename=filename).store()
        # return orm.SinglefileData(file=io.BytesIO(f.read()), filename=filename).store()

# TODO: make all aiida_shell launch_shell_job use submit=True

class MonomerWorkChain(WorkChain):
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
            'nmols', valid_type=orm.Int,
            default=lambda: orm.Int(1000),
            help='The number of molecules to insert into the simulation box.'
        )
        spec.input(
            'time_step', valid_type=orm.Float,
            default=lambda: orm.Float(0.001),
            help='The MD time step in picoseconds.'
        )
        spec.input(
            'shear_rates', valid_type=orm.List,
            default=lambda: orm.List(list=[0.005, 0.002, 0.05, 0.02, 0.01, 0.1, 0.2]),
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
        spec.input(
            'gromacs_equilibration_steps', valid_type=orm.Int,
            default=lambda: orm.Int(5000),
            # default=lambda: orm.Int(500000),  # TODO: use this value after testing
        )
        # spec.input('computer_label', valid_type=orm.Str, help='The computer to run the workflow on.')
        spec.input('acpype_code', valid_type=orm.AbstractCode, help='Code for running the `acpype` program.')
        spec.input('obabel_code', valid_type=orm.AbstractCode, help='Code for running the `obabel` program.')
        spec.input('veloxchem_code', valid_type=orm.AbstractCode, help='Code for python with `veloxchem` installed.')
        spec.input('gmx_code', valid_type=orm.AbstractCode, help='Code for running `gmx` or `gmx_mpi`.')

        spec.input(
            'with_mpi', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, run calculations with MPI when possible.'
        )
        spec.input(
            'max_num_machines', valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help='The maximum number of machines (nodes) to use for the calculations.'
        )
        spec.input(
            'max_wallclock_seconds', valid_type=orm.Int,
            default=lambda: orm.Int(3600),
            help='The maximum wallclock time in seconds for the calculations.'
        )

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
            cls.make_gromacs_equilibration_input,
            cls.submit_equilibration_init,
            cls.submit_equilibration_run,
            # cls.make_gromacs_nemd_inputs,
            cls.submit_nemd_init,
            if_(cls.should_do_alltogheter)(
                cls.submit_nemd_run_parallel,
            ).else_(
                while_(cls.do_nemd_serial)(
                    cls.submit_nemd_run_serial,
                )
            ),
            # cls.should_do_alltogheter,
            # cls.set_nemd_inputs,
            # cls.submit_parallel_nemd,
            cls.collect_outputs,
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
        spec.exit_code(
            370, 'ERROR_SUB_PROCESS_FAILED_GMX_NEMD',
            message='A GROMACS NEMD subprocess calculation failed.'
        )

    def setup(self):
        """Setup context variables."""
        self.ctx.smiles_string = self.inputs.smiles_string.value
        self.ctx.ff = self.inputs.force_field.value
        self.ctx.nmols = self.inputs.nmols.value

        gmx_computer: orm.Computer = self.inputs.gmx_code.computer
        gmx_sched = gmx_computer.get_scheduler()
        self.ctx.gromacs_serial_metadata = {
            'options': {
                'withmpi': False,
                'resources': {
                    'num_machines': 1,
                    'num_mpiprocs_per_machine': 1,
                },
                'max_wallclock_seconds': self.inputs.max_wallclock_seconds.value,
                'redirect_stderr': True,
            }
        }
        self.ctx.gmx_run_metadata = ptr = {
            'options': {
                'withmpi': self.inputs.with_mpi.value,
                'resources': {
                    'num_machines': self.inputs.max_num_machines.value,
                    'num_mpiprocs_per_machine': gmx_computer.get_default_mpiprocs_per_machine(),
                },
                'max_wallclock_seconds': self.inputs.max_wallclock_seconds.value,
                'redirect_stderr': True,
            }
        }

        self.report(f'{self.ctx.gmx_run_metadata}')

        max_mem = gmx_computer.get_default_memory_per_machine()
        if max_mem is not None:
            ptr['options']['max_memory_kb'] = max_mem

        self.ctx.gmx_computer = gmx_computer
        self.ctx.gmx_scheduler = gmx_sched

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
            outputs=[f'{basename}.acpype'],
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
            metadata=self.ctx.gromacs_serial_metadata,
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
            metadata=self.ctx.gromacs_serial_metadata,
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
            metadata=self.ctx.gmx_run_metadata,
            outputs=['minimize.gro']
        )

        self.report(f'Submitted job: {node_mdrun}')
        self.report(f'Outputs: {results_mdrun}')

        self.ctx.minimized_gro = results_mdrun['minimize_gro']

    def make_gromacs_equilibration_input(self):
        """Generate a basic GROMACS equilibration input file."""
        self.ctx.equilibrate_mdp = fnc.generate_gromacs_equilibration_input(
            num_steps=self.inputs.gromacs_equilibration_steps,
            time_step=self.inputs.time_step,
            reference_temperature=self.inputs.reference_temperature
        )

    def submit_equilibration_init(self):
        """Initialize GROMACS equilibration run to generate .tpr file."""
        self.report('Running GROMACS equilibration run INIT...')
        out_filename = 'equilibrate.tpr'
        results_grompp, node_grompp = launch_shell_job(
            self.inputs.gmx_code,
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o ' + out_filename,
            nodes={
                'mdpfile': self.ctx.equilibrate_mdp,
                'grofile': self.ctx.minimized_gro,
                'topfile': self.ctx.top_updated,
                'itpfile': self.ctx.itp_with_resp
            },
            metadata=self.ctx.gromacs_serial_metadata,
            outputs=['mdout.mdp', out_filename]
        )

        self.report(f'Submitted job: {node_grompp}')
        self.report(f'Outputs: {results_grompp}')

        self.ctx.tpr = results_grompp[out_filename.replace('.', '_')]

    def submit_equilibration_run(self):
        """Run GROMACS equilibration mdrun."""
        self.report('Running GROMACS equilibration run MDRUN...')
        out_filename = 'equilibrate.gro'
        results_mdrun, node_mdrun = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tprfile} -deffnm equilibrate',
            nodes={
                'tprfile': self.ctx.tpr
            },
            metadata=self.ctx.gmx_run_metadata,
            outputs=[out_filename]
        )

        self.report(f'Submitted job: {node_mdrun}')
        self.report(f'Outputs: {results_mdrun}')

        self.ctx.equilibrated_gro = results_mdrun[out_filename.replace('.', '_')]

    # def make_gromacs_nemd_inputs(self):
    #     """Generate GROMACS input files for each shear rate."""
    #     self.report('Preparing GROMACS NEMD input files for each shear rate...')

    #     self.ctx.srate_inputs = {}
    #     for shear_rate in self.inputs.shear_rates:
    #         mdp = fnc.generate_gromacs_shear_rate_input(
    #             nsteps=self.inputs.num_steps,
    #             time_step=self.inputs.time_step,
    #             ref_t=self.inputs.reference_temperature,
    #             shear_rate=orm.Float(shear_rate)
    #         )
    #         self.ctx.srate_inputs[shear_rate] = mdp

    def submit_nemd_init(self):
        self.report('Running GROMACS NEMD initialization for each shear rate...')
        fname = 'aiida.tpr'
        for shear_rate in self.inputs.shear_rates:
            mdp_file = fnc.generate_gromacs_shear_rate_input(
                nsteps=self.inputs.num_steps,
                time_step=self.inputs.time_step,
                ref_t=self.inputs.reference_temperature,
                shear_rate=orm.Float(shear_rate)
            )
            _, node = launch_shell_job(
                'gmx_mpi',
                arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o ' + fname,
                nodes={
                    'mdpfile': mdp_file,
                    'grofile': self.ctx.equilibrated_gro,
                    'topfile': self.ctx.top_updated,
                    'itpfile': self.ctx.itp_with_resp,
                },
                metadata=self.ctx.gromacs_serial_metadata,
                outputs=[fname],
                submit=True
            )
            self.report(f'Submitted job for shear rate {shear_rate}: {node}')
            self.to_context(**{f'grompp_{shear_rate}': node})

    def should_do_alltogheter(self):
        """Check if all shear rates can be in parallel runs."""
        sched = self.ctx.gmx_scheduler
        if isinstance(sched, DIRECT_SCHEDULER):
            self.report('Direct scheduler does not support running multiple jobs.')
            self.ctx.nemd_serial_cnt = 0
            return False
        return True

    def submit_nemd_run_parallel(self):
        """Submit the parallel NEMD WorkChain."""
        self.report('Submitting GROMACS NEMD runs as parallel jobs...')
        basename = 'aiida'
        for srate in self.inputs.shear_rates:
            tpr_calc = self.ctx[f'grompp_{srate}']
            tpr_file = tpr_calc.outputs['aiida_tpr']

            _, node = launch_shell_job(
                self.inputs.gmx_code,
                arguments='mdrun -v -s {tpr_file} -deffnm ' + basename,
                nodes={
                    'tpr_file': tpr_file,
                },
                metadata=self.ctx.gmx_run_metadata,
                outputs=[f'{basename}.edr'],
                submit=True
            )

            self.report(f'Submitted job: {node}')
            self.to_context(**{f'nemd_{srate}': node})

    def do_nemd_serial(self):
        """Check if there are remaining shear rates to run in serial."""
        return self.ctx.nemd_serial_cnt < len(self.inputs.shear_rates)

    def submit_nemd_run_serial(self):
        """Submit the serial NEMD WorkChain."""
        srate = self.inputs.shear_rates[self.ctx.nemd_serial_cnt]
        self.ctx.nemd_serial_cnt += 1

        basename = 'aiida'
        tpr_calc = self.ctx[f'grompp_{srate}']
        tpr_file = tpr_calc.outputs['aiida_tpr']

        self.report(f'Submitting GROMACS NEMD run for shear rate {srate} as a serial job...')
        _, node = launch_shell_job(
            self.inputs.gmx_code,
            arguments='mdrun -v -s {tpr_file} -deffnm ' + basename,
            nodes={
                'tpr_file': tpr_file,
            },
            metadata=self.ctx.gmx_run_metadata,
            outputs=[f'{basename}.edr'],
            submit=True
        )

        self.report(f'Submitted job: {node}')

        return ToContext(**{f'nemd_{srate}': node})

    # def set_nemd_inputs(self):
    #     """Load prepared input files for NEMD step."""
    #     mdp_files = sorted(glob('eta_*.mdp'))
    #     self.ctx.mdp_files = orm.List(list=[os.path.abspath(f) for f in mdp_files])

    # def submit_parallel_nemd(self):
    #     """Submit the parallel NEMD WorkChain."""
    #     inputs = {
    #         'grofile': self.ctx.equilibrated_gro,
    #         'topfile': self.ctx.top_updated,
    #         'itpfile': self.ctx.itp_with_resp,
    #         'mdp_files': self.ctx.mdp_files
    #     }
    #     future = self.submit(NemdParallelWorkChain, **inputs)
    #     return ToContext(nemd=future)

    def collect_outputs(self):
        """Collect .edr files from the NEMD parallel run."""
        self.report('Collecting .edr files from NEMD runs...')

        calc_map = {srate: self.ctx[f'nemd_{srate}' ] for srate in self.inputs.shear_rates}

        failed = [srate for srate, calc in calc_map.items() if not calc.is_finished_ok]
        if failed:
            self.report(f'NEMD runs for shear rates {failed} did not finish successfully.')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GMX_NEMD

        self.ctx.edr_outputs = []
        for srate, calc in calc_map.items():
            edr_file = calc.outputs['aiida_edr']
            self.ctx.edr_outputs.append(edr_file)
            self.report(f'Collected .edr file for shear rate {srate}: {edr_file.filename}')
            self.out(f'edr_output_{srate}', edr_file)

    # def submit_postprocessing(self):
    #     edr_inputs = {f'edr_{i}': node for i, node in enumerate(self.ctx.edr_outputs)}
    #     future = self.submit(
    #         PostprocessPressureWorkChain,
    #         edr_files=edr_inputs,
    #         grofile=self.ctx.equilibrated_gro,
    #         mdp_files=self.ctx.mdp_files
    #     )
    #     return ToContext(postprocess=future)

    # def finalize(self):
    #     """Expose outputs from postprocessing."""
    #     for key in self.ctx.postprocess.outputs:
    #         value = self.ctx.postprocess.outputs[key]
    #         self.out(key, value)
