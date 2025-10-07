"""
Estimates box edge length for gmx insert-molecules
Uses a density below actual density in order for insert-molecules to work efficiently
Actual densities of the liquid molecular systems are roughly 1 g/mol
"""

from aiida.orm import Float
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
    self.ctx.box_size = Float(edge_length_nm).store()
    self.report(f"Calculated box edge length: {edge_length_nm:.2f} nm")
