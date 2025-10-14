# -*- coding: utf-8 -*-
"""Command line scripts to launch a `MonomerWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
import click

from . import cmd_launch
from ..utils import defaults, launch, options, validate


@cmd_launch.command('viscosity')
# Required options
@options.NUM_STEPS(required=True)
@options.SMILES_STRING(required=True)
@options.REFERENCE_TEMPERATURE(required=True)
# Codes
@options.ACPYPE_CODE(required=True)
@options.OBABEL_CODE(required=True)
@options.VELOXCHEM_CODE(required=True)
@options.GMX_CODE(required=True)
@options.GMX_CODE_LOCAL(required=False)
# Optional parameters,
@options.FORCE_FIELD(required=False)
@options.NMOLS(required=False)
@options.TIME_STEP(required=False)
@options.CLEAN_WORKDIR()
# Resources
@options.MAX_NUM_MACHINES()
@options.MAX_WALLCLOCK_SECONDS()
@options.WITH_MPI()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(
    # Required parameters
    num_steps, smiles_string, reference_temperature,
    # Codes
    acpype_code, obabel_code, veloxchem_code, gmx_code,
    gmx_code_local,
    # Optional parameters,
    clean_workdir,
    nmols, force_field, time_step,
    # Resources
    max_num_machines, max_wallclock_seconds, with_mpi, daemon
):
    """Run a `MonomerWorkChain` to compute the viscosity of a liquid."""
    from aiida.plugins import WorkflowFactory

    builder = WorkflowFactory('aitw.gromacs.viscosity').get_builder()

    builder.num_steps = num_steps
    builder.smiles_string = smiles_string
    builder.reference_temperature = reference_temperature
    if nmols is not None:
        builder.nmols = nmols
    if force_field is not None:
        builder.force_field = force_field
    if time_step is not None:
        builder.time_step = time_step

    builder.acpype_code = acpype_code
    builder.obabel_code = obabel_code
    builder.veloxchem_code = veloxchem_code
    builder.gmx_code = gmx_code
    if gmx_code_local is not None:
        builder.gmx_code_local = gmx_code_local

    builder.clean_workdir = clean_workdir

    builder.max_num_machines = max_num_machines
    builder.max_wallclock_seconds = max_wallclock_seconds
    builder.with_mpi = with_mpi

    launch.launch_process(builder, daemon)
