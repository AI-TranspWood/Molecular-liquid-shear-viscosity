"""Implementation of the WorkChain for virtual lab."""
import io, os
from aiida.engine import WorkChain, run
from MonomerWorkChain import MonomerWorkChain

if __name__ == '__main__':    
    result = run(MonomerWorkChain)
    print(result)
