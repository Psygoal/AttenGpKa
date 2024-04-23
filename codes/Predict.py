import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage,DrawingOptions, rdMolDraw2D
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib.colors import ColorConverter
from IPython.display import Image
import matplotlib as mpl
from tqdm import tqdm
from .Model import PredictionModel

IPythonConsole.ipython_useSVG = False
IPythonConsole.molSize = (500, 500)
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# solvent information
data = np.load('./MPNN/solvent_info.npz',allow_pickle=True)
unique_smile_array = data['unique_smile_array']
unique_concentration_array = data['unique_concentration_array']
data.close()

# Feature encoder
class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items(): # items返回键值对列表
            s = sorted(list(s)) #sorted 升序排列，list 的 sort 方法返回的是对已经存在的列表进行操作，sorted 返回的是一个新的 list。
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))# dict创建字典。zip打包为元组的列表
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs) # getattr() 函数用于返回一个对象属性值
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs(includeNeighbors=True)

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    
    def n_degree(self,atom):
        return atom.GetTotalDegree() 
    
    def chiraltag(self, atom):
        return atom.GetChiralTag().name
    
    def isinring(self, atom):
        return atom.IsInRing()
    
    def isaromatic(self, atom):
        return atom.GetIsAromatic()
    
    
    def isin3ring(self, atom):
        return atom.IsInRingSize(3)
    
    def isin4ring(self, atom):
        return atom.IsInRingSize(4)
    
    def isin5ring(self, atom):
        return atom.IsInRingSize(5)
    
    def isin6ring(self, atom):
        return atom.IsInRingSize(6)
    
    def isin8ring(self, atom):
        return atom.IsInRingSize(8)
    
    def isin16ring(self,atom):
        return atom.IsInRingSize(16)
    
    def formalcharge(self, atom):
        return atom.GetFormalCharge()
    
    def is_linked_to_conjugated_system(self, atom):
        for neighbors in atom.GetNeighbors():
                for neighbor_bond in neighbors.GetBonds():
                    if neighbor_bond.GetIsConjugated():
                        return True
        return False

        


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()
    
    def stereo(self, bond):
        return bond.GetStereo().name
    
    def isinring(self, bond):
        return bond.IsInRing()


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
#         radical_electron = atom.GetNumRadicalElectrons()
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



def find_tobe_removed(smiles_list):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []
    
    toberemoved_list = []
    # 计算电子分布
    for smile in smiles_list:
        molecule = molecule_from_smiles(smile)
        AllChem.ComputeGasteigerCharges(molecule)
        
        statu = 1
        
        for atom in molecule.GetAtoms():
            # cal _GasteigerCharge
            gast_charge = round(atom.GetDoubleProp('_GasteigerCharge'), 4)
            if (np.isnan(gast_charge) or np.isinf(gast_charge)):
                statu *= 0
                
        toberemoved_list.append(statu)
    return toberemoved_list



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
        dataset = dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
    else:
        dataset = dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)
    
    return dataset


def make_prediction(smile,picked_atom_id):
    
    smile_array = unique_smile_array.copy()
    smile_array[:,0] = smile
    
    picked_atoms = np.int64(np.ones((len(unique_smile_array)))*picked_atom_id)

    
    x_test = graphs_from_smiles(smile_array,picked_atoms,unique_concentration_array)
    
    # no use
    y_test = np.zeros(len(unique_smile_array),dtype=np.float64)[...,None]
    
    test_dataset = MPNNDataset(x_test, y_test)
    
    predictionmodel = PredictionModel(52,14,128,128,3,3,l2_param=0.1)
    predictionmodel.load_weights("weightstobeloaded.h5")
    test_prediction = predictionmodel.predict(test_dataset,verbose=0).flatten()
    
    return predictionmodel
    
    
    
    
    