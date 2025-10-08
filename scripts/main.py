"""Implementation of the WorkChain for virtual lab."""
from aiida.engine import run

from aitw_aiida_viscosity.workflows.torefactor import MonomerWorkChain

if __name__ == '__main__':
    result = run(MonomerWorkChain)
    print(result)
