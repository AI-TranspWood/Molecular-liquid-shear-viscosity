"""Collection of calcfunctions used by the workchains."""
import re

from aiida import orm
from aiida.engine import calcfunction
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@calcfunction
def run_resp_injection(
        pdb_file: orm.SinglefileData, itp_file: orm.SinglefileData
    ) -> orm.SinglefileData:
    """
    Inject RESP charges from a PDB file into an ITP file.

    Args:
        pdb_file (SinglefileData): The PDB file containing RESP charges.
        itp_file (SinglefileData): The ITP file to be updated with RESP charges.

    Returns:
        SinglefileData: The updated ITP file with RESP charges injected.
    """
    # Retrieve file contents

    with pdb_file.open() as f_pdb:
        pdb_lines = f_pdb.readlines()

    with itp_file.open() as f_itp:
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

    updated_itp_node = orm.SinglefileData.from_string('\n'.join(updated_lines), filename='updated.itp')

    return updated_itp_node

@calcfunction
def update_top_file(
        nmols: orm.Int,
        top_file: orm.SinglefileData,
        itp_file: orm.SinglefileData,
    ) -> orm.SinglefileData:
    """
    Update the .top file to reference the new .itp file and adjust the molecule count.

    Args:
        nmols (Int): Number of molecules.
        top_file (SinglefileData): The original .top file to be updated.
        itp_file (SinglefileData): The updated .itp file with RESP charges.
    Returns:
        SinglefileData: The updated .top file.
    """
    nmols = nmols.value
    with top_file.open() as f:
        lines = f.readlines()

    # Determine the updated .itp filename
    updated_itp_filename = itp_file.filename

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

    top_updated = orm.SinglefileData.from_string('\n'.join(new_lines), filename='updated.top')

    return top_updated

@calcfunction
def get_box_size(
        nmols: orm.Int,
        smiles_string: orm.Str,
    ):
    """Approximate molar mass from SMILES string using RDKit and store as AiiDA Float node."""
    smiles = smiles_string.value
    mol = Chem.MolFromSmiles(smiles)
    nmols = nmols.value

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    mw = rdMolDescriptors.CalcExactMolWt(mol)

    Na = 6.022e23
    rho = 0.5 # g/cm3 - small enough so that gmx insert-molecule works with ease
    box_volume_cm3 = nmols * mw / (rho * Na)
    box_volume_nm3 = box_volume_cm3 * 1e21
    edge_length_nm = box_volume_nm3 ** (1/3)

    return  orm.Float(edge_length_nm)

@calcfunction
def generate_gromacs_minimization_input(minimization_steps: orm.Int) -> orm.SinglefileData:
    """Generate a basic GROMACS minimization input file."""
    template = '\n'.join([
        'integrator          = steep',
        f'nsteps              = {minimization_steps.value}',
        'nstcgsteep          = 100',
        'emtol               = 0',
        'emstep              = 0.01',
        'nstlog              = 100',
        'nstenergy           = 100',
        'nstlist             = 10',
        'ns_type             = grid',
        'pbc                 = xyz',
        'cutoff-scheme       = Verlet',
        'vdwtype             = cutoff',
        'vdw-modifier        = None',
        'rlist               = 1.0',
        'rvdw                = 1.0',
        'rvdw-switch         = 1.0',
        'coulombtype         = PME',
        'rcoulomb            = 1.0',
        'DispCorr            = EnerPres'
    ])

    return orm.SinglefileData.from_string(template, filename='minim.mdp')

@calcfunction
def generate_veloxchem_input(basis_set: orm.Str) -> orm.SinglefileData:
    """Generate a basic VeloxChem input file."""
    template = '\n'.join([
        'import sys',
        'import veloxchem as vlx',
        'infile = sys.argv[1]',
        'molecule = vlx.Molecule.read_xyz(infile)',
        'mol_xyz = molecule.get_xyz_string()',
        # 'basis = vlx.MolecularBasis.read(molecule, "6-31G*")',
        f'basis = vlx.MolecularBasis.read(molecule, "{basis_set.value}")',
        'resp_drv = vlx.RespChargesDriver()',
        'resp_charges = resp_drv.compute(molecule, basis, "resp")',
    ])

    return orm.SinglefileData.from_string(template, filename='aiida_vlx.py')
