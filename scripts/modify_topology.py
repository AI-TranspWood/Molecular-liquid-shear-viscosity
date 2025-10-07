"""
Modifies the topology file from ACPYPE by inserting the actual number of molecules
Also updates the name of the .itp file as this was changed when RESP charges were inserted
"""

import os
from aiida.orm import SinglefileData
from tempfile import NamedTemporaryFile

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
    self.ctx.top_updated = SinglefileData(file=os.path.abspath(updated_top_path)).store()

    print(f"Updated .top file stored: {self.ctx.top_updated.filename} (pk={self.ctx.top_updated.pk})")
