from aiida.engine import WorkChain
from aiida.orm import SinglefileData, Str, load_computer
from aiida_shell import launch_shell_job

class NemdMdrunWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('tpr_file', valid_type=SinglefileData)
        spec.input('defnm', valid_type=Str)
        spec.outline(cls.run_mdrun)
        spec.output('edr_file', valid_type=SinglefileData)

    def run_mdrun(self):
        custom_scheduler_commands = "\n".join([
            "#SBATCH -p gen02_ivybridge",
            "#SBATCH --exclude=c[1003-1006,2001]"
        ])
        prepend_text = "\n".join([
            "export OMP_NUM_THREADS=1",
            "module load gcc/12.2.0"
        ])
        res, _ = launch_shell_job(
            'gmx_mpi',
            arguments='mdrun -v -s {tpr_file} -deffnm {defnm}',
            nodes={
                'tpr_file': self.inputs.tpr_file,
                'defnm': self.inputs.defnm
            },
            metadata={
                'computer': load_computer('Tohtori'),
                'options': {
                    'withmpi': True,
                    'resources': {'num_machines': 2, 'num_mpiprocs_per_machine': 20},
                    'custom_scheduler_commands': custom_scheduler_commands,
                    'prepend_text': prepend_text
                }
            },
            outputs=[f"{self.inputs.defnm.value}.edr"]
        )
        self.out('edr_file', list(res.values())[0])
