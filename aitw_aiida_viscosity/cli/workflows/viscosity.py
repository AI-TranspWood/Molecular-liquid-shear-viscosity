# -*- coding: utf-8 -*-
"""Command line scripts to launch a `MonomerWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
import click

from . import cmd_launch
from ..utils import defaults, launch, options, validate


@cmd_launch.command('viscosity')
@options.ACPYPE_CODE()
@options.OBABEL_CODE()
@options.VELOXCHEM_CODE()
@options.GMX_CODE()
@options.SMILES_STRING()
@options.CLEAN_WORKDIR()
@options.MAX_NUM_MACHINES()
@options.MAX_WALLCLOCK_SECONDS()
@options.WITH_MPI()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(
    # code,
    smiles_string,
    acpype_code, obabel_code, veloxchem_code, gmx_code,
    # computer,
    clean_workdir, max_num_machines, max_wallclock_seconds, with_mpi, daemon
):
    """Run a `MonomerWorkChain` to compute the viscosity of a liquid."""
    from aiida.plugins import WorkflowFactory

    builder = WorkflowFactory('aitw.gromacs.viscosity').get_builder()

    builder.smiles_string = smiles_string
    builder.clean_workdir = clean_workdir

    builder.acpype_code = acpype_code
    builder.obabel_code = obabel_code
    builder.veloxchem_code = veloxchem_code
    builder.gmx_code = gmx_code

    launch.launch_process(builder, daemon)
