from aiida.engine import WorkChain
from aiida.orm import SinglefileData, Str, load_computer
from aiida_shell import launch_shell_job

class NemdGromppWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('mdpfile', valid_type=SinglefileData)
        spec.input('grofile', valid_type=SinglefileData)
        spec.input('topfile', valid_type=SinglefileData)
        spec.input('itpfile', valid_type=SinglefileData)
        spec.input('defnm', valid_type=Str)
        spec.outline(cls.run_grompp)
        spec.output('tpr_file', valid_type=SinglefileData)

    def run_grompp(self):
        res, _ = launch_shell_job(
            'gmx_mpi',
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o {defnm}.tpr',
            nodes={
                'mdpfile': self.inputs.mdpfile,
                'grofile': self.inputs.grofile,
                'topfile': self.inputs.topfile,
                'itpfile': self.inputs.itpfile,
                'defnm': self.inputs.defnm
            },
            metadata={
                'computer': load_computer('Tohtori'),
                'options': {
                    'withmpi': False,
                    'resources': {'num_machines': 1}
                }
            },
            outputs=[f"{self.inputs.defnm.value}.tpr"]
        )
        self.out('tpr_file', list(res.values())[0])
