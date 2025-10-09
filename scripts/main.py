"""Implementation of the WorkChain for virtual lab."""
from aiida import load_profile
from aiida.engine import run

from aitw_aiida_viscosity.workflows.MonomerWorkChain import MonomerWorkChain

if __name__ == '__main__':
    load_profile()
    result = run(MonomerWorkChain)
    print(result)
