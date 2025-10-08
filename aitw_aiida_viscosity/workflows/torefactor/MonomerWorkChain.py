"""Implementation of the WorkChain for AITW viscosity calculation."""
from glob import glob
import os

from aiida.engine import ToContext, WorkChain
from aiida.orm import List

from .NemdParallelWorkChain import NemdParallelWorkChain
from .PostprocessPressureWorkChain import PostprocessPressureWorkChain
from .compute_box_dimensions import get_box_size
from .compute_partial_charges_Tohtori import submit_veloxchem
# Local modules (commented out if you don't need in this minimal example)
from .initialize_system_Tohtori import submit_acpype
from .inject_resp_charges import run_resp_injection
from .insert_molecules_Tohtori import build_gro
from .modify_topology import update_top_file
from .run_equilibration_Tohtori import submit_equilibration
from .run_minimization_Tohtori import submit_minimization


class MonomerWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.outline(
            cls.set_initial_values,
            submit_acpype,
            submit_veloxchem,
            run_resp_injection,
            update_top_file,
            get_box_size,
            build_gro,
            submit_minimization,
            submit_equilibration,
            cls.set_nemd_inputs,
            cls.submit_parallel_nemd,
            cls.collect_outputs,
            cls.submit_postprocessing,
            cls.finalize
        )
        spec.outputs.dynamic = True

    def set_initial_values(self):
        """Optional if you want to include ACPYPE and VeloxChem."""
        self.ctx.mol = 'ibma'
        self.ctx.smiles_string = 'CC(=C)C(=O)OC1C[C@H]2CC[C@]1(C)C2(C)C'
        self.ctx.ff = 'gaff2'
        self.ctx.basename = self.ctx.mol + '_' + self.ctx.ff + '_am1bcc'
        self.ctx.nmols = 1000

    def set_nemd_inputs(self):
        """Load prepared input files for NEMD step."""
        mdp_files = sorted(glob('eta_*.mdp'))
        self.ctx.mdp_files = List(list=[os.path.abspath(f) for f in mdp_files])

    def submit_parallel_nemd(self):
        """Submit the parallel NEMD WorkChain."""
        inputs = {
            'grofile': self.ctx.equilibrated_gro,
            'topfile': self.ctx.top_updated,
            'itpfile': self.ctx.itp_with_resp,
            'mdp_files': self.ctx.mdp_files
        }
        future = self.submit(NemdParallelWorkChain, **inputs)
        return ToContext(nemd=future)

    def collect_outputs(self):
        """Collect .edr files from the NEMD parallel run."""
        nemd_wc = self.ctx.nemd

        if not nemd_wc.is_finished_ok:
            self.report('NemdParallelWorkChain did not finish successfully.')
            return

        self.ctx.edr_outputs = list(nemd_wc.outputs.edr_outputs.values())

        self.report('All done! Collected outputs:')
        for i, node in enumerate(self.ctx.edr_outputs):
            self.report(f"edr_output {i}: {node.filename}")
            self.out(f'edr_output_{i}', node)

    def submit_postprocessing(self):
        edr_inputs = {f'edr_{i}': node for i, node in enumerate(self.ctx.edr_outputs)}
        future = self.submit(
            PostprocessPressureWorkChain,
            edr_files=edr_inputs,
            grofile=self.ctx.equilibrated_gro,
            mdp_files=self.ctx.mdp_files
        )
        return ToContext(postprocess=future)

    def finalize(self):
        """Expose outputs from postprocessing."""
        for key in self.ctx.postprocess.outputs:
            value = self.ctx.postprocess.outputs[key]
            self.out(key, value)
