# -*- coding: utf-8 -*-
"""
@author: LXY
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from PIL import Image

# custom functions
from GraphfromSmiles import graphs_from_smiles,molecule_from_smiles
from GraphfromSmiles import MPNNDataset
from GraphfromSmiles import prepare_batch
from Model import PredictionModel
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Temporaty suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='training or predictig')
    
    parser.add_argument('-m', "--mode", choices=['train','predict'])
    
    # specify when mode is train
    parser.add_argument('-l', '--loadweights', type=str, help="path of the model to be loaded")
    parser.add_argument('-d', '--data', type=str, help="path of the data to be predicted")
    
    # specify when mode is predict
    parser.add_argument('-s','--saveweights',type=str, help='path of the model to be saved')
    parser.add_argument('-f','--fold',type=int, help='fold of cross-validation')
    
    args = parser.parse_args()
    
    
    mode=args.mode
    
    predictionmodel = PredictionModel(52,14,128,128,3,3,l2_param=0.1)
    
    if mode == 'train':
        model_path = args.saveweights
        fold = args.fold
        
        print('This data is collected from the iBond database. If you use this data in your work, please cite:\nJ.-D. Yang, X.-S. Xue, P. Ji, X. Li, J.-P. Cheng, Internet Bond-energy Databank (pKa and BDE): iBonD Home Page. http://ibond.nankai.edu.cn or  http://ibond.chem.tsinghua.edu.cn.')
        
        df = pd.read_csv('./data.csv',skiprows=[0,1,2]) 
        mol_smiles = df.smiles.values
        
        # solvent smiles & cocentrations
        solvent_smiles_source = df.solvent_smiles.values
        solvent_smiles = [solv_smile.split(' + ') for solv_smile in solvent_smiles_source]
        for i in solvent_smiles:
            if len(i)==1:
                i.append('O')
        solvent_smiles = np.array(solvent_smiles,dtype=object)
        solvent_concentrations = df.solvent_concentrations.values
        
        smiles_array = np.hstack((mol_smiles[...,None],solvent_smiles))
        picked_atoms = df.picked_atoms.values
        concentrations = df['solvent_concentrations'].values[...,None]
        concentrations = np.hstack((np.ones_like(concentrations),concentrations,1-concentrations))
        pka_values = df['pka_values'].values
        
        # data preparation
        index = np.arange(df.values.shape[0])
        init = fold
        test_index = index[init::10]
        valid_index = index[2::10]
        train_index = np.array([i for i in index if ((i not in test_index)&(i not in valid_index))])
        train_index = np.random.permutation(train_index)

        # Train set: 80 % of data
        x_train = graphs_from_smiles(smiles_array[train_index],\
                                     picked_atoms[train_index],concentrations[train_index])
        y_train = pka_values[train_index]
        
        # Valid set: 10 % of data
        x_valid = graphs_from_smiles(smiles_array[valid_index],\
                                     picked_atoms[valid_index],concentrations[valid_index])
        y_valid = pka_values[valid_index]

        # Test set: 10 % of data
        x_test = graphs_from_smiles(smiles_array[test_index],\
                                    picked_atoms[test_index],concentrations[test_index])
        y_test = pka_values[test_index]
        
        train_dataset = MPNNDataset(x_train, y_train[...,None], shuffle=True)
        train_dataset_for_check = MPNNDataset(x_train, y_train[...,None])
        valid_dataset = MPNNDataset(x_valid, y_valid[...,None])
        test_dataset = MPNNDataset(x_test, y_test[...,None])
        
        # model configs
        
        
        optimizer = keras.optimizers.Adam(1e-4)
        savebestmodel = keras.callbacks.ModelCheckpoint('%s'%(model_path),monitor='val_mse',save_best_only=True,verbose=1)
        earlystop = keras.callbacks.EarlyStopping(monitor='val_mse',patience=500)

        predictionmodel.compile(loss=keras.losses.MeanSquaredError(),
                                loss_weights=1.,
                                optimizer=optimizer,
                                metrics = 'mse')
        
        # model training
        history = predictionmodel.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=3000,
            verbose=1,
            callbacks=[savebestmodel,earlystop]
        )
        
        
        predictionmodel.load_weights('%s'%(model_path))
        
        # print test result
        test_prediction = predictionmodel.predict(test_dataset,verbose=0)
        
        rmse = keras.losses.MeanSquaredError()(test_prediction[:], y_test[:,None]).numpy()**(0.5)
        mae = keras.losses.MeanAbsoluteError()(test_prediction[:], y_test[:,None]).numpy()
        r = np.corrcoef(test_prediction[:,0], y_test)[0,1]
        r2 = r2_score(test_prediction[:,0], y_test)
        
        print('test rmse is %.4f\nmae is %.4f\nR is %.4f\nR2 is %.4f'%(rmse,mae,r,r2))
        plt.figure(figsize=[9,9])
        plt.scatter(y_test[:],test_prediction[:,0],s=20)
        plt.scatter(y_test[idx],test_prediction[idx,0],s=200,marker='*',c='r')
        plt.plot(y_test[:],y_test[:],lw=5,c='k',alpha=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('reference',fontsize=30)
        plt.ylabel('prediction',fontsize=30)
        ax=plt.gca()
        plt.text(0.1,0.7,'RMSE = %.2f\nMAE = %.2f\nR = %.2f\nR$^2$ = %.2f'%(rmse,mae,r,r2),transform=ax.transAxes,fontsize=30)
        plt.savefig('test_predict.jpg',dpi=500)
        
    elif mode == 'predict':
        data = args.data
        model_path = args.loadweights
        
        predictionmodel.load_weights('%s'%(model_path))
        
        df = pd.read_csv(data) 
        mol_smiles = df.smiles.values
        
        # solvent smiles & cocentrations
        solvent_smiles_source = df.solvent_smiles.values
        solvent_smiles = [solv_smile.split(' + ') for solv_smile in solvent_smiles_source]
        for i in solvent_smiles:
            if len(i)==1:
                i.append('O')
        solvent_smiles = np.array(solvent_smiles,dtype=object)
        solvent_concentrations = df.solvent_concentrations.values
        
        smiles_array = np.hstack((mol_smiles[...,None],solvent_smiles))
        picked_atoms = df.picked_atoms.values
        concentrations = df['solvent_concentrations'].values[...,None]
        concentrations = np.hstack((np.ones_like(concentrations),concentrations,1-concentrations))
        
        x_test = graphs_from_smiles(smiles_array,picked_atoms,concentrations)
        y_test = np.zeros((smiles_array.shape[0]))
        test_dataset = MPNNDataset(x_test, y_test[...,None])

        test_prediction = predictionmodel.predict(test_dataset,verbose=0)
        with open('prediction.csv','w') as f:
            f.write('prediction_pka,\n')
            for i in test_prediction[:,0]:
                f.write('%.4f\n'%(i))
    
