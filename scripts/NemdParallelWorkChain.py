from aiida.engine import WorkChain, ToContext
from aiida.orm import List, SinglefileData, Str
from NemdGromppWorkChain import NemdGromppWorkChain
from NemdMdrunWorkChain import NemdMdrunWorkChain
import os

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
            self.out(f'edr_outputs.{key.replace("mdrun_", "")}', mdrun_wc.outputs.edr_file)
            self.report(f"[{key}] Output collected: {mdrun_wc.outputs.edr_file.filename}")
