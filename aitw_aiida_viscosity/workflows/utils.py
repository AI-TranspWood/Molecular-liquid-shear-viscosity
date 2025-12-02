"""Implementation of the WorkChain for AITW viscosity calculation."""
import copy

from aiida import orm


def clean_calcjob_remote(node):
    """Clean the remote directory of a ``CalcJobNode``."""
    cleaned = False
    try:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        cleaned = True
    except (IOError, OSError, KeyError):
        pass
    return cleaned

def clean_workchain_calcs(workchain):
    """Clean all remote directories of a workchain's descendant calculations."""
    cleaned_calcs = []

    for called_descendant in workchain.called_descendants:
        if isinstance(called_descendant, orm.CalcJobNode):
            if clean_calcjob_remote(called_descendant):
                cleaned_calcs.append(called_descendant.pk)

def create_metadata(gmx_computer: orm.Computer, metadata_tpl: dict, report_func = lambda msg: None):
    """Setup the metadata templates for the calculations."""
    # gmx_computer: orm.Computer = self.inputs.gmx_code.computer

    # metadata_tpl = dict(self.inputs.shelljob.metadata)
    options = metadata_tpl['options'] = dict(metadata_tpl.get('options', {}))
    resources = options['resources'] = dict(options.get('resources', {}))

    if 'max_wallclock_seconds' not in options:
        report_func('WARNING: max_wallclock_seconds not set in metadata; using default of 3600 seconds.')
        options['max_wallclock_seconds'] = 3600

    if 'num_machines' not in resources:
        report_func('WARNING: num_machines not set in metadata; using default of 1 machine.')
        resources['num_machines'] = 1

    computer_num_mpiprocs = gmx_computer.get_default_mpiprocs_per_machine()
    if 'num_mpiprocs_per_machine' in resources:
        nmpm = resources['num_mpiprocs_per_machine']
        report_func(
            f'Using `num_mpiprocs_per_machine` from metadata: {nmpm} instead of value from'
            f'computer {computer_num_mpiprocs}.'
        )
    else:
        resources['num_mpiprocs_per_machine'] = computer_num_mpiprocs

    max_mem = gmx_computer.get_default_memory_per_machine()
    if max_mem is not None:
        options['max_memory_kb'] = max_mem

    options['redirect_stderr'] = True

    serial_mdata = copy.deepcopy(metadata_tpl)
    parall_mdata = copy.deepcopy(metadata_tpl)

    serial_mdata['options']['withmpi'] = False
    serial_mdata['options']['resources']['num_machines'] = 1
    serial_mdata['options']['resources']['num_mpiprocs_per_machine'] = 1

    return serial_mdata, parall_mdata
