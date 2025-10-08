# -*- coding: utf-8 -*-
"""Command line scripts to launch a `MonomerWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
import click

from . import cmd_launch
from ..utils import defaults, launch, options, validate


@cmd_launch.command('viscosity')
@options_core.CODE(required=True, type=types.CodeParamType(entry_point='gromacs.mdrun'))
@options.CLEAN_WORKDIR()
@options.MAX_NUM_MACHINES()
@options.MAX_WALLCLOCK_SECONDS()
@options.WITH_MPI()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(
    code, clean_workdir, max_num_machines, max_wallclock_seconds, with_mpi, daemon
):
    """Run a `MonomerWorkChain` to compute the viscosity of a liquid."""
    from aitw_aiida_viscosity.workflows.torefactor import MonomerWorkChain

    inputs = {
        'metadata': {
            'description': 'MonomerWorkChain to compute the viscosity of a liquid',
            'call_link_label': 'monomer_workchain',
            'options': {
                'max_num_machines': max_num_machines,
                'max_wallclock_seconds': max_wallclock_seconds,
                'with_mpi': with_mpi,
                'clean_workdir': clean_workdir,
            },
        },
        'pw_code': code,
    }


    launch.launch_process(MonomerWorkChain, daemon, **inputs)
