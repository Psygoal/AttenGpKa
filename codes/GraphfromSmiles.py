# -*- coding: utf-8 -*-
"""
@author: LXY
"""

import numpy as np
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem import AllChem
from Featurizer import AtomFeaturizer, BondFeaturizer
from tqdm import tqdm

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"Br","C","Cl","F","H","N","O","S","P","Si"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        'n_degree':{0,1,2,3,4,5,6},
        "chiraltag":{'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER','CHI_TETRAHEDRAL',\
                     'CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL','CHI_OCTAHEDRAL'},
        'isinring':{True},
        'isaromatic':{True},
        'isin3ring':{True},
        'isin4ring':{True},
        'isin5ring':{True},
        'isin6ring':{True},
        'isin8ring':{True},
        'isin16ring':{True},
        'is_linked_to_conjugated_system':{True}
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "stereo":{'STEREOANY','STEREOCIS','STEREOE','STEREOTRANS','STEREOZ','STEREONONE'},
        'isinring':{True}
    }
)

def remove_nitro_charge(molecule):
    for pattern in ['[N+](=O)[O-]']:
        nitro_pattern = Chem.MolFromSmarts(pattern)
        matches = molecule.GetSubstructMatches(nitro_pattern)

        for match in matches:
            for atom_index in match:
                atom = molecule.GetAtomWithIdx(atom_index)
                atom.SetFormalCharge(0)

def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without the sanitization step that caused the error
    try:
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    except:
        print(smiles)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []
    # 计算电子分布
#     AllChem.ComputeGasteigerCharges(molecule)
    
    for atom in molecule.GetAtoms():
        atom_feature = atom_featurizer.encode(atom)
        charge = atom.GetFormalCharge()
        atom_feature = np.hstack((atom_feature,charge))
        atom_features.append(atom_feature)

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def graphs_from_smiles(smiles_array,selected_atom_indices,concentrations):
    # Initialize graphs
    mol_atom_features_list = []
    mol_bond_features_list = []
    mol_pair_indices_list = []
    
    solv1_atom_features_list = []
    solv1_bond_features_list = []
    solv1_pair_indices_list = []
    
    solv2_atom_features_list = []
    solv2_bond_features_list = []
    solv2_pair_indices_list = []
    
    selected_atom_indices = selected_atom_indices
    
    for name,smiles_list in zip(['mol','solv1','solv2'],smiles_array.T):
        
        for smiles in tqdm(smiles_list):
            
            molecule = molecule_from_smiles(smiles)
            remove_nitro_charge(molecule)
            atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

            exec('%s_atom_features_list.append(atom_features)'%(name))
            exec('%s_bond_features_list.append(bond_features)'%(name))
            exec('%s_pair_indices_list.append(pair_indices)'%(name))
      
    # Convert lists to ragged tensors for tf.data.Dataset later on
    
    return (
            tf.ragged.constant(mol_atom_features_list, dtype=tf.float32),
            tf.ragged.constant(mol_bond_features_list, dtype=tf.float32),
            tf.ragged.constant(mol_pair_indices_list, dtype=tf.int64),
            tf.ragged.constant(solv1_atom_features_list, dtype=tf.float32),
            tf.ragged.constant(solv1_bond_features_list, dtype=tf.float32),
            tf.ragged.constant(solv1_pair_indices_list, dtype=tf.int64),
            tf.ragged.constant(solv2_atom_features_list, dtype=tf.float32),
            tf.ragged.constant(solv2_bond_features_list, dtype=tf.float32),
            tf.ragged.constant(solv2_pair_indices_list, dtype=tf.int64),
            tf.constant(selected_atom_indices, dtype=tf.int64),
            tf.constant(concentrations, dtype=tf.float32)
    )


def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    mol_atom_features, mol_bond_features,mol_pair_indices,\
    solv1_atom_features, solv1_bond_features,solv1_pair_indices,\
    solv2_atom_features, solv2_bond_features, solv2_pair_indices,\
    selected_atom_indices, concentrations = x_batch
    
    # 1 mol
    # Obtain number of atoms and bonds for each graph (molecule)
    mol_num_atoms = mol_atom_features.row_lengths()
    mol_num_bonds = mol_bond_features.row_lengths()
    
    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    mol_molecule_indices = tf.range(len(mol_num_atoms))
    mol_molecule_indicator = tf.repeat(mol_molecule_indices, mol_num_atoms)
    
    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(mol_molecule_indices[:-1], mol_num_bonds[1:])
    increment = tf.cumsum(mol_num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(mol_num_bonds[0], 0)])
    mol_pair_indices = mol_pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    mol_pair_indices = mol_pair_indices + increment[:, tf.newaxis]
    mol_atom_features = mol_atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    mol_bond_features = mol_bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    
    # 2 solv1
    # Obtain number of atoms and bonds for each graph (molecule)
    solv1_num_atoms = solv1_atom_features.row_lengths()
    solv1_num_bonds = solv1_bond_features.row_lengths()
    
    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    solv1_molecule_indices = tf.range(len(solv1_num_atoms))
    solv1_molecule_indicator = tf.repeat(solv1_molecule_indices, solv1_num_atoms)
    
    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(solv1_molecule_indices[:-1], solv1_num_bonds[1:])
    increment = tf.cumsum(solv1_num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(solv1_num_bonds[0], 0)])
    solv1_pair_indices = solv1_pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    solv1_pair_indices = solv1_pair_indices + increment[:, tf.newaxis]
    solv1_atom_features = solv1_atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    solv1_bond_features = solv1_bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    
    # 2 solv2
    # Obtain number of atoms and bonds for each graph (molecule)
    solv2_num_atoms = solv2_atom_features.row_lengths()
    solv2_num_bonds = solv2_bond_features.row_lengths()
    
    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    solv2_molecule_indices = tf.range(len(solv2_num_atoms))
    solv2_molecule_indicator = tf.repeat(solv2_molecule_indices, solv2_num_atoms)
    
    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(solv2_molecule_indices[:-1], solv2_num_bonds[1:])
    increment = tf.cumsum(solv2_num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(solv2_num_bonds[0], 0)])
    solv2_pair_indices = solv2_pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    solv2_pair_indices = solv2_pair_indices + increment[:, tf.newaxis]
    solv2_atom_features = solv2_atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    solv2_bond_features = solv2_bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (mol_atom_features, mol_bond_features, mol_pair_indices, selected_atom_indices, mol_molecule_indicator,\
             solv1_atom_features, solv1_bond_features, solv1_pair_indices, solv1_molecule_indicator,\
             solv2_atom_features, solv2_bond_features, solv2_pair_indices, solv2_molecule_indicator,concentrations), y_batch


def MPNNDataset(X, y, batch_size=128, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(25000)
        dataset = dataset.batch(batch_size).map(prepare_batch, -1)
    else:
        dataset = dataset.batch(batch_size).map(prepare_batch, -1)
    
    return dataset