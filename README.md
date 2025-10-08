# AiiDA Workchain for Viscosity

AiiDA workflow for automated calculation of shear viscosity of a molecular liquid

## Features

- Generation of Gromacs molecular topology using ACPYPE and GAFF force field from SMILES string
- RESP partial charge calculation using Veloxchem
- Parallel shear viscosity calculation for several shear rates using Gromacs
- Output: Newtonian shear viscosity

![Workflow Diagram](images/workflow.png)
