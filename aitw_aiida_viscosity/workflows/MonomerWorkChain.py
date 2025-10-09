"""Implementation of the WorkChain for AITW viscosity calculation."""
from glob import glob
import io
import os
import re
from tempfile import NamedTemporaryFile

from aiida import orm
from aiida.engine import ToContext, WorkChain
from aiida.orm import load_code, load_computer, load_node
from aiida_shell import launch_shell_job
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .NemdParallelWorkChain import NemdParallelWorkChain
from .PostprocessPressureWorkChain import PostprocessPressureWorkChain


def wrap_file(folder, filename):
    with folder.open(filename, 'rb') as f:
        return orm.SinglefileData(file=io.BytesIO(f.read()), filename=filename).store()

class MonomerWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################

        # OUTLINE ############################################################################
        spec.outline(
            cls.set_initial_values,
            cls.submit_acpype,
            cls.submit_veloxchem,
            cls.run_resp_injection,
            cls.update_top_file,
            cls.get_box_size,
            cls.build_gro,
            cls.submit_minimization,
            cls.submit_equilibration,
            cls.set_nemd_inputs,
            cls.submit_parallel_nemd,
            cls.collect_outputs,
            cls.submit_postprocessing,
            cls.finalize
        )

        # OUTPUTS ############################################################################
        spec.outputs.dynamic = True

    def set_initial_values(self):
        """Optional if you want to include ACPYPE and VeloxChem."""
        self.ctx.mol = 'ibma'
        self.ctx.smiles_string = 'CC(=C)C(=O)OC1C[C@H]2CC[C@]1(C)C2(C)C'
        self.ctx.ff = 'gaff2'
        self.ctx.basename = self.ctx.mol + '_' + self.ctx.ff + '_am1bcc'
        self.ctx.nmols = 1000

    def submit_acpype(self):
        self.report('Submitting the aiida-shell subprocess ')
        self.report('Submitting the aiida-shell subprocess ')
        mol = self.ctx.mol
        smiles_string = self.ctx.smiles_string
        ff = self.ctx.ff
        basename = self.ctx.basename
        code=load_code('acpype@Tohtori')
        results_acpype, node_acpype = launch_shell_job(
            'acpype',
            arguments = '-i '+ smiles_string +' -n 0 -c bcc -q sqm -b '+ basename +' -a '+ ff +' -s 108000',
            metadata={
                'options': {
                    'computer': load_computer('Tohtori'),
                    'withmpi': False,
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 1,
                    }
                }
            },
            outputs=[f"{basename}.acpype"]
        )

        self.report('...in acpype...')
        self.report(f'Calculation terminated: {node_acpype.process_state}')
        self.report('Outputs:')
        self.report(results_acpype.keys())

        # Expected filename patterns
        target_suffixes = {
            'gro': 'GMX.gro',
            'itp': 'GMX.itp',
            'top': 'GMX.top',
            'pdb': 'NEW.pdb',
        }

        # Initialize ctx fields
        self.ctx.gro = None
        self.ctx.itp = None
        self.ctx.top = None
        self.ctx.pdb = None

        for key, node in results_acpype.items():
    #        self.report(f'{key}: {node.__class__.__name__}<{node.pk}>')
            if isinstance(node, orm.FolderData):
                for filename in node.list_object_names():
                    # Match based on suffix
                    if filename.endswith(target_suffixes['gro']):
                        self.ctx.gro = wrap_file(node, filename)
                    elif filename.endswith(target_suffixes['itp']):
                        self.ctx.itp = wrap_file(node, filename)
                    elif filename.endswith(target_suffixes['top']):
                        self.ctx.top = wrap_file(node, filename)
                    elif filename.endswith(target_suffixes['pdb']):
                        self.ctx.pdb = wrap_file(node, filename)

        # Optionally verify what was found
        self.report(f"GRO: {self.ctx.gro.filename if self.ctx.gro else 'Not found'}")
        self.report(f"ITP: {self.ctx.itp.filename if self.ctx.itp else 'Not found'}")
        self.report(f"TOP: {self.ctx.top.filename if self.ctx.top else 'Not found'}")
        self.report(f"PDB: {self.ctx.pdb.filename if self.ctx.pdb else 'Not found'}")

        # Now convert .pdb to .xyz using OpenBabel

        code=load_code('obabel@Tohtori')
        results_obabel, node_obabel = launch_shell_job(
                                'obabel',
                                arguments = '{pdbfile} -O mol.xyz',
                                nodes={
                                        'pdbfile': self.ctx.pdb
                                },
                                metadata={
                                    'options': {
                                        'computer': load_computer('Tohtori'),
                                        'withmpi': False,
                                        'resources': {
                                            'num_machines': 1,
                                            'num_mpiprocs_per_machine': 1,
                                        }
                                    }
                                },
                                outputs=['mol.xyz']
        )

        self.report('...in obabel...')
        self.report(f'Calculation terminated: {node_acpype.process_state}')
        self.report('Outputs:')
        self.report(results_obabel.keys())

        nodelist=[]
        self.ctx.xyz = None
        for key, node in results_obabel.items():
            self.report(f'{key}: {node.__class__.__name__}<{node.pk}>')
            nodelist.append(int({node.pk}.pop()))
        self.ctx.xyz = nodelist[0]

    def submit_veloxchem(self):
        self.report('Submitting the aiida-shell subprocess ')
        xyzfile = load_node(self.ctx.xyz)
        # xyzfile = orm.SinglefileData(file=os.path.join(os.getcwd(), 'mol.xyz')) # This for test purposes

        script_content = '\n'.join([
            'import sys',
            'import veloxchem as vlx',
            'infile = sys.argv[1]',
            'molecule = vlx.Molecule.read_xyz(infile)',
            'mol_xyz = molecule.get_xyz_string()',
            'basis = vlx.MolecularBasis.read(molecule, "6-31G*")',
            'resp_drv = vlx.RespChargesDriver()',
            'resp_charges = resp_drv.compute(molecule, basis, "resp")',
        ])
        script_file = orm.SinglefileData(file=io.StringIO(script_content), filename='resp.py').store()
        code = load_code('veloxchem@Tohtori')

        # Inputs to launch_shell_job
        results_veloxchem, node_veloxchem = launch_shell_job(
            'python',
            arguments = '{script_file} {xyzfile}',
            nodes={
                'script_file': script_file,
                'xyzfile': xyzfile
            },
            metadata = {
                'options': {
                    'computer': load_computer('Tohtori'),
                    'withmpi': False,
                    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                    'max_wallclock_seconds': 1200,
                    'custom_scheduler_commands': ('#SBATCH --cpus-per-task=20\n'),
                    'prepend_text': (
                        'export OMP_NUM_THREADS=20\n'
                        'eval "$(conda shell.bash hook)"\n'
                        'conda activate echem')
                }
            },
            outputs = ['*.pdb']  # <- Tell AiiDA to fetch this file
        )

        self.report('...in veloxchem...')
        self.report('Outputs:')
        self.report(results_veloxchem.keys())

        nodelist=[]
        self.ctx.pdb = None  # This deletes the pdb file from veloxchem but it is no longer needed
        for key, node in results_veloxchem.items():
            print(f'{key}: {node.__class__.__name__}<{node.pk}>')
            nodelist.append(int({node.pk}.pop()))
        self.ctx.pdb = load_node(nodelist[0])

    def run_resp_injection(self):
        """
        Inject RESP charges from a PDB file into an ITP file and store the result in self.ctx.
        Assumes self.ctx.pdb and self.ctx.itp are SinglefileData nodes.
        """
        # Retrieve file contents

        with self.ctx.pdb.open() as f_pdb:
            pdb_lines = f_pdb.readlines()

        with self.ctx.itp.open() as f_itp:
            itp_lines = f_itp.readlines()

        # Extract RESP charges from the PDB file (columns 71-76)
        charges = []
        for line in pdb_lines:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    charge = float(line[70:76].strip())
                    charges.append(charge)
                except ValueError:
                    continue  # Skip any malformed lines

        # Inject charges into ITP file
        updated_lines = []
        in_atoms_section = False
        charge_index = 0
        for line in itp_lines:
            if line.strip().startswith('[ atoms ]'):
                in_atoms_section = True
                updated_lines.append(line)
                continue
            if in_atoms_section:
                if line.strip().startswith('['):  # End of atoms section
                    in_atoms_section = False
                elif line.strip() and not line.strip().startswith(';'):
                    fields = re.split(r'\s+', line.strip())
                    if len(fields) >= 7 and charge_index < len(charges):
                        fields[6] = f"{charges[charge_index]:.6f}"
                        charge_index += 1
                        updated_lines.append('    '.join(fields) + '\n')
                        continue
            updated_lines.append(line)

        # Write updated ITP file to disk
        updated_filename = 'updated.itp'
        with open(updated_filename, 'w') as f_out:
            f_out.writelines(updated_lines)

        # Store as new AiiDA node
        updated_itp_node = orm.SinglefileData(file=os.path.abspath('updated.itp')).store()
        self.ctx.itp_with_resp = updated_itp_node
        self.report(f'Updated ITP file with RESP charges stored as node <{updated_itp_node.pk}>')

        # Optional: clean up local temp file
        os.remove(updated_filename)

    def update_top_file(self):
        # Load original top file from ctx

        nmols = self.ctx.nmols

        with self.ctx.top.open() as f:
            lines = f.readlines()

        # Determine the updated .itp filename
        updated_itp_filename = self.ctx.itp_with_resp.filename

        # Process lines: update .itp reference and [ molecules ] count
        new_lines = []
        in_molecules_section = False
        for line in lines:
            stripped = line.strip()

            # Replace .itp file reference
            if stripped.startswith('#include') and stripped.endswith('.itp"'):
                # Replace with updated itp filename
                newline = f'#include "{updated_itp_filename}"\n'
                new_lines.append(newline)
                continue

            # Update molecule count
            if '[ molecules ]' in stripped:
                in_molecules_section = True
                new_lines.append(line)
                continue
            elif in_molecules_section:
                if stripped and not stripped.startswith(';'):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        parts[1] = str(nmols)  # Update molecule count
                        newline = f'{parts[0]:<20s}{parts[1]}\n'
                        new_lines.append(newline)
                        in_molecules_section = False  # Done with this section
                        continue

            new_lines.append(line)

        # Write to a new file
        with NamedTemporaryFile('w+', delete=False, suffix='.top') as f_new:
            f_new.writelines(new_lines)
            updated_top_path = f_new.name

        # Store as SinglefileData node
        self.ctx.top_updated = orm.SinglefileData(file=os.path.abspath(updated_top_path)).store()

        self.report(f"Updated .top file stored: {self.ctx.top_updated.filename} (pk={self.ctx.top_updated.pk})")

    def get_box_size(self):
        """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
        smiles = self.ctx.smiles_string
        mol = Chem.MolFromSmiles(smiles)
        nmols = self.ctx.nmols

        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        mw = rdMolDescriptors.CalcExactMolWt(mol)

        Na = 6.022e23
        rho = 0.5 # g/cm3 - small enough so that gmx insert-molecule works with ease
        box_volume_cm3 = nmols * mw / (rho * Na)
        box_volume_nm3 = box_volume_cm3 * 1e21
        edge_length_nm = box_volume_nm3 ** (1/3)

        # Store in self.ctx and provenance
        self.ctx.box_size = orm.Float(edge_length_nm).store()
        self.report(f"Calculated box edge length: {edge_length_nm:.2f} nm")

    def build_gro(self):
        self.report('Building system... ')
        code = load_code('gromacs2024@Tohtori')
        grofile = self.ctx.gro
        nmols = orm.Int(self.ctx.nmols)
        box_vector = self.ctx.box_size
        results_insert, node_insert = launch_shell_job(
            'gmx_mpi',
            arguments=(
                'insert-molecules -ci {grofile} -o system.gro -nmol {nmols} '
                '-try 1000 -box {box_vector} {box_vector} {box_vector}'
            ),
            nodes={
                'grofile': grofile,
                'nmols': nmols,
                'box_vector': box_vector
            },
            metadata={
            'options': {
                'computer': load_computer('Tohtori'),
                'withmpi': False,
                'resources': {
                    'num_machines': 1,
                    'num_mpiprocs_per_machine': 1,
                }
            }
        },
        outputs=['system.gro']
        )

        nodelist=[]
        for key, node in results_insert.items():
            nodelist.append(int({node.pk}.pop()))
        self.ctx.system_gro = load_node(nodelist[0])

    def submit_minimization(self):
        print('Running grompp... ')
        code = load_code('gromacs2024@Tohtori')
        grofile = self.ctx.system_gro
        topfile = self.ctx.top_updated
        itpfile = self.ctx.itp_with_resp
        mdpfile = orm.SinglefileData(file=os.path.join(os.getcwd(), 'minimize.mdp'))
        results_grompp, node_grompp = launch_shell_job(
            'gmx_mpi',
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o minimize.tpr',
            nodes={
                'mdpfile': mdpfile,
                'grofile': grofile,
                'topfile': topfile,
                'itpfile': itpfile
            },
            metadata={
                'options': {
                    'computer': load_computer('Tohtori'),
                    'withmpi': False,
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 1,
                    }
                }
            },
            outputs=['mdout.mdp','minimize.tpr']
        )

        self.report('...in grompp...')
        self.report(f'Calculation terminated: {node_grompp.process_state}')

        nodelist=[]
        for key, node in results_grompp.items():
            nodelist.append(node.pk)
        self.ctx.tpr = load_node(nodelist[1])

        # gromacs command
        # gmx_mpi mdrun -v -deffnm minimize
        self.report('Running mdrun... ')
        tprfile = self.ctx.tpr

        custom_scheduler_commands = '\n'.join([
            '#SBATCH -p gen02_ivybridge',
            '#SBATCH --exclude=c[1003-1006,2001]'
        ])

        prepend_text = '\n'.join([
            'export OMP_NUM_THREADS=1',
            'module load gcc/12.2.0'
        ])

        results_mdrun, node_mdrun = launch_shell_job(
            'gmx_mpi',
            arguments='mdrun -v -s {tprfile} -deffnm minimize',
            nodes={
                'tprfile': tprfile
            },
            metadata={
            'options': {
                'computer': load_computer('Tohtori'),
                'withmpi': True,
                'resources': {
                    'num_machines': 1,
                    'num_mpiprocs_per_machine': 20},
                'custom_scheduler_commands': custom_scheduler_commands,
                'prepend_text': prepend_text
            }
            },
            outputs=['minimize.gro']
        )

        self.report('...in mdrun...')
        self.report(f'Calculation terminated: {node_mdrun.process_state}')

        nodelist=[]
        for key, node in results_mdrun.items():
            nodelist.append(node.pk)
        self.ctx.minimized_gro = load_node(nodelist[0])

    def submit_equilibration(self):
        self.report('Running grompp... ')
        code = load_code('gromacs2024@Tohtori')
        grofile = self.ctx.minimized_gro
        topfile = self.ctx.top_updated
        itpfile = self.ctx.itp_with_resp
        mdpfile = orm.SinglefileData(file=os.path.join(os.getcwd(), 'equilibrate.mdp'))
        results_grompp, node_grompp = launch_shell_job(
            'gmx_mpi',
            arguments='grompp -f {mdpfile} -c {grofile} -r {grofile} -p {topfile} -o equilibrate.tpr',
            nodes={
                'mdpfile': mdpfile,
                'grofile': grofile,
                'topfile': topfile,
                'itpfile': itpfile
            },
            metadata={
                'options': {
                    'computer': load_computer('Tohtori'),
                    'withmpi': False,
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 1,
                    }
                }
            },
            outputs=['mdout.mdp','equilibrate.tpr']
        )

        self.report('...in grompp...')
        self.report(f'Calculation terminated: {node_grompp.process_state}')

        nodelist=[]
        for key, node in results_grompp.items():
            nodelist.append(node.pk)
        self.ctx.tpr = load_node(nodelist[1])

        # gromacs command
        # gmx_mpi mdrun -v -deffnm minimize

        self.report('Running mdrun... ')
        tprfile = self.ctx.tpr

        custom_scheduler_commands = '\n'.join([
            '#SBATCH -p gen02_ivybridge',
            '#SBATCH --exclude=c[1003-1006,2001]'
        ])

        prepend_text = '\n'.join([
            'export OMP_NUM_THREADS=1',
            'module load gcc/12.2.0'
        ])

        results_mdrun, node_mdrun = launch_shell_job(
            'gmx_mpi',
            arguments='mdrun -v -s {tprfile} -deffnm equilibrate',
            nodes={
                'tprfile': tprfile
            },
            metadata={
                'options': {
                    'computer': load_computer('Tohtori'),
                    'withmpi': True,
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 20},
                    'custom_scheduler_commands': custom_scheduler_commands,
                    'prepend_text': prepend_text
                }
            },
            outputs=['equilibrate.gro']
        )

        self.report('...in mdrun...')
        self.report(f'Calculation terminated: {node_mdrun.process_state}')

        nodelist=[]
        for key, node in results_mdrun.items():
            nodelist.append(node.pk)
        self.ctx.equilibrated_gro = load_node(nodelist[0])

    def set_nemd_inputs(self):
        """Load prepared input files for NEMD step."""
        mdp_files = sorted(glob('eta_*.mdp'))
        self.ctx.mdp_files = orm.List(list=[os.path.abspath(f) for f in mdp_files])

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
