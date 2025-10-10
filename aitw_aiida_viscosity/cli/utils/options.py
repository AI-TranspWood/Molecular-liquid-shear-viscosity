# -*- coding: utf-8 -*-
"""Pre-defined overridable options for commonly used command line interface parameters."""
from aiida.cmdline.params import types
from aiida.cmdline.params.options import OverridableOption
import click

ACPYPE_CODE = OverridableOption(
    '--acpype', 'acpype_code', type=types.CodeParamType(entry_point='core.shell'),
    help='A single code for acpype (e.g. acpype@localhost).'
)

OBABEL_CODE = OverridableOption(
    '--obabel', 'obabel_code', type=types.CodeParamType(entry_point='core.shell'),
    help='A single code for Open Babel (e.g. obabel@localhost).'
)

VELOXCHEM_CODE = OverridableOption(
    '--veloxchem', 'veloxchem_code', type=types.CodeParamType(),
    help='A single code for VeloxChem (e.g. veloxchem@localhost).'
)

GMX_CODE = OverridableOption(
    '--gromacs', 'gmx_code', type=types.CodeParamType(entry_point='core.shell'),
    help='A single code for GROMACS mdrun (e.g. gromacs@localhost).'
)

SMILES_STRING = OverridableOption(
    '-s',
    '--smiles-string',
    type=click.STRING,
    required=True,
    help='The SMILE code representation of the molecule to simulate.'
)

FORCE_FIELD = OverridableOption(
    '-f',
    '--force-field',
    'force_field',
    type=click.Choice(['gaff', 'amber', 'gaff2', 'amber2']),
    help='The force field to use for the simulation.'
)

REFERENCE_TEMPERATURE = OverridableOption(
    '-T',
    '--temperature',
    'reference_temperature',
    type=click.FLOAT,
    help='The reference temperature in Kelvin for the simulation.'
)

NMOLS = OverridableOption(
    '-N',
    '--num-molecules',
    'nmols',
    type=click.INT,
    help='The number of molecules to include in the simulation box.'
)

NUM_STEPS = OverridableOption(
    '-n',
    '--num-steps',
    'num_steps',
    type=click.INT,
    help='The number of MD steps to perform in the production run.'
)

TIME_STEP = OverridableOption(
    '-t',
    '--time-step',
    'time_step',
    type=click.FLOAT,
    help='The time step (in ps) to use for the MD simulations.'
)

MAX_NUM_MACHINES = OverridableOption(
    '-m',
    '--max-num-machines',
    type=click.INT,
    default=1,
    show_default=True,
    help='The maximum number of machines (nodes) to use for the calculations.'
)

MAX_WALLCLOCK_SECONDS = OverridableOption(
    '-w',
    '--max-wallclock-seconds',
    type=click.INT,
    default=1800,
    show_default=True,
    help='the maximum wallclock time in seconds to set for the calculations.'
)

WITH_MPI = OverridableOption(
    '-i', '--with-mpi', is_flag=True, default=False, show_default=True, help='Run the calculations with MPI enabled.'
)

PARENT_FOLDER = OverridableOption(
    '-P',
    '--parent-folder',
    'parent_folder',
    type=types.DataParamType(sub_classes=('aiida.data:core.remote',)),
    show_default=True,
    required=False,
    help='The PK of a parent remote folder (for restarts).'
)

DAEMON = OverridableOption(
    '-d',
    '--daemon',
    is_flag=True,
    default=False,
    show_default=True,
    help='Submit the process to the daemon instead of running it locally.'
)

CLEAN_WORKDIR = OverridableOption(
    '-x',
    '--clean-workdir',
    is_flag=True,
    default=False,
    show_default=True,
    help='Clean the remote folder of all the launched calculations after completion of the workchain.'
)
