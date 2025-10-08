import os
import re

from aiida.orm import SinglefileData, load_node


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
    updated_itp_node = SinglefileData(file=os.path.abspath('updated.itp')).store()
    self.ctx.itp_with_resp = updated_itp_node
    self.report(f'Updated ITP file with RESP charges stored as node <{updated_itp_node.pk}>')

    # Optional: clean up local temp file
    os.remove(updated_filename)
