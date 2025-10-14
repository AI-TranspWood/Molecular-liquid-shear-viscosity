# AiiDA Workchain for Viscosity

AiiDA workflow for automated calculation of shear viscosity of a molecular liquid

## Features

- Generation of Gromacs molecular topology using ACPYPE and GAFF force field from SMILES string
- RESP partial charge calculation using Veloxchem
- Parallel shear viscosity calculation for several shear rates using Gromacs
- Output: Newtonian shear viscosity

![Workflow Diagram](images/workflow.png)

## Installation

```bash
cd <PATH to folder with pyproject.toml>
pip install .[plotting]
```

**NOTE**: The plotting extra deps are needed to perform the plotting of the results.

## Usage

The package provides the following AiiDA entry point that can be used to load the workchain:

- `aitw.gromacs.viscosity`: Compute the viscosity of a molecule starting from the SMILES string.


### CLI

The tool makes available 2 main cli commands grouped following the normal aiida CLI style.

- `aitw-viscosity workflow launch viscosity`: Launch the viscosity workchain from the command line.
- `aitw-viscosity data plot-viscosity`: Plot the results of a completed viscosity workchain.

For both commands you can use the `-h` / `--help` flag to get more information about the available options.

example

```bash

aitw-viscosity workflow launch viscosity \
    --acpype <ACPYPE CODE IDENTIFIER> \
    --gromacs <GROMACS CODE IDENTIFIER> \
    --veloxchem <VELOXCHEM CODE IDENTIFIER> \
    --obabel <OPENBABEL CODE IDENTIFIER> \
    --smiles-string 'CC(=C)C(=O)OC1C[C@H]2CC[C@]1(C)C2(C)C' \
    --num-molecules 100 \
    --num-steps 5000000 \
    --temperature 343
```

Replace the `<CODE IDENTIFIER>` with the actual code identifiers (PK or label) of the installed codes in your AiiDA database. \
Adjust the other parameters as needed.

OUTPUT:

``` bash
# ...
# REPORT
# ...
# Output link               Node pk and type
# ------------------------------------------------------------
# acpype__gro               SinglefileData<6905>
# acpype__itp               SinglefileData<6908>
# acpype__pdb               SinglefileData<6914>
# acpype__top               SinglefileData<6911>
# equilibrated_box_length_nm Float<6978>
# equilibrated_gro          SinglefileData<6975>
# nemd__edr_0_002           SinglefileData<7061>
# nemd__edr_0_005           SinglefileData<7054>
# nemd__edr_0_01            SinglefileData<7082>
# nemd__edr_0_02            SinglefileData<7075>
# nemd__edr_0_05            SinglefileData<7068>
# nemd__edr_0_1             SinglefileData<7089>
# nemd__edr_0_2             SinglefileData<7096>
# system_gro                SinglefileData<6941>
# viscosity_data            ArrayData<7171>
# xyz                       SinglefileData<6920>
```

Plotting the results

```bash
aitw-viscosity data plot-viscosity 7171 --show-plot
```

Replace the PK of the viscosity `ArrayData` with the actual PK from your run.

**NOTE**: Note all possible inputs to the workchain are exposed through the CLI. For more advanced usage, consider running the workchain programmatically.

### Programmatically

See the [CLI file](aitw_aiida_viscosity/cli/workflows/viscosity.py) for an example of how to run the workchain programmatically.

### Tab autocompletion

Enabling tab autocompletion https://click.palletsprojects.com/en/stable/shell-completion/

E.G for `bash` run the command

```bash
eval "$(_AITW_VISCOSITY_COMPLETE=bash_source aitw-viscosity)"
```

You can also add it to either `~/.bashrc` or, if you are using a virtual environment, to `bin/activate` of the virtual environment to avoid running the command for every new shell.
