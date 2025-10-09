import os

from aiida.engine import ToContext, WorkChain
from aiida.orm import List, SinglefileData, Str, load_computer
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

class NemdMdrunWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('tpr_file', valid_type=SinglefileData)
        spec.input('defnm', valid_type=Str)
        spec.outline(cls.run_mdrun)
        spec.output('edr_file', valid_type=SinglefileData)

    def run_mdrun(self):
        custom_scheduler_commands = '\n'.join([
            '#SBATCH -p gen02_ivybridge',
            '#SBATCH --exclude=c[1003-1006,2001]'
        ])
        prepend_text = '\n'.join([
            'export OMP_NUM_THREADS=1',
            'module load gcc/12.2.0'
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

class NemdParallelWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('grofile', valid_type=SinglefileData)
        spec.input('topfile', valid_type=SinglefileData)
        spec.input('itpfile', valid_type=SinglefileData)
        spec.input('mdp_files', valid_type=List)
        spec.outline(
            cls.submit_grompps,
            cls.submit_mdruns,
            cls.finalize
        )
        spec.output_namespace('edr_outputs', valid_type=SinglefileData, dynamic=True)

    def submit_grompps(self):
        futures = {}
        for mdp_path in self.inputs.mdp_files.get_list():
            defnm = os.path.basename(mdp_path).replace('.mdp', '')
            inputs = {
                'mdpfile': SinglefileData(file=mdp_path),
                'grofile': self.inputs.grofile,
                'topfile': self.inputs.topfile,
                'itpfile': self.inputs.itpfile,
                'defnm': Str(defnm)
            }
            futures[f'grompp_{defnm}'] = self.submit(NemdGromppWorkChain, **inputs)
        return ToContext(**futures)

    def submit_mdruns(self):
        futures = {}
        for key, grompp_wc in self.ctx.items():
            if not grompp_wc.is_finished_ok:
                self.report(f"[{key}] Skipping GromppWorkChain that failed.")
                continue
            defnm = key.replace('grompp_', '')
            inputs = {
                'tpr_file': grompp_wc.outputs.tpr_file,
                'defnm': Str(defnm)
            }
            futures[f'mdrun_{defnm}'] = self.submit(NemdMdrunWorkChain, **inputs)
        return ToContext(**futures)

    def finalize(self):
        for key, mdrun_wc in self.ctx.items():
            if not key.startswith('mdrun_'):
                continue
            if not mdrun_wc.is_finished_ok:
                self.report(f"[{key}] Skipping failed mdrun WorkChain.")
                continue
            self.out(f'edr_outputs.{key.replace('mdrun_', '')}', mdrun_wc.outputs.edr_file)
            self.report(f"[{key}] Output collected: {mdrun_wc.outputs.edr_file.filename}")
