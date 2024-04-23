# -*- coding: utf-8 -*-
"""
@author: LXY
"""
import numpy as np

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