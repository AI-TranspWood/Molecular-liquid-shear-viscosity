"""Implementation of the WorkChain for virtual lab."""
import io
import os

from MonomerWorkChain import MonomerWorkChain
from aiida.engine import WorkChain, run

if __name__ == '__main__':
    result = run(MonomerWorkChain)
    print(result)
