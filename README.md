# AttenGpKa
AttenGpKa is a graph neural network for predicting pKa of different molecules in various solvents. If you use resources of this [project](https://zenodo.org/records/11436576), please cite:                         

1. H. An, X. Liu, W. Cai, X. Shao. AttenGpKa: A Universal Predictor of Solvation Acidity Using Graph Neural Network and Molecular Topology. Journal of Chemical Information and Modeling, 2024. DOI: 10.1021/acs.jcim.4c00449

The training data, which is collected from the iBond database, can be found in the Supporting Information of our published paper (https://doi.org/10.1021/acs.jcim.4c00449).

If you use this data in your work, in addition to this work, please also cite:

2. J.-D. Yang, X.-S. Xue, P. Ji, X. Li, J.-P. Cheng, Internet Bond-energy Databank (pKa and BDE): iBonD Home Page. http://ibond.nankai.edu.cn.

### 1 environment requirements  
The required packages and their versions are included in the [requirements.txt](./requirements.txt) file. Run the following commands to build your environment:   
```bash
conda create -y -n AttenGpKa python==3.8.11
conda activate AttenGpKa
pip install -r requirements.txt
conda install -y ipython
```
     
### 2 usage
We open-sourced all the [data](https://doi.org/10.1021/acs.jcim.4c00449), [codes, and models](https://zenodo.org/records/11436576) used in this work. Additionally, we have developed a user-friendly software for Windows OS, which allows anyone to easily use our model. 

### 3 model training and predicting
If users want to train their own model with our provided training data:  
```bash
python AttenGpka.py -m train -s ./test.h5 -f 0
```
```-m train```  or ```--mode train``` represents the training mode
```-s ./test.h5```  or ```--saveweights ./test.h5``` is the path of the trained model to be saved       
```-f 0```  or ```--fold 0``` is the way to split training and test dataset       
         
If users want to predict pKa with the available model:
```bash
python AttenGpka.py -m predict -l ./model.h5 -d ./test_data.csv
```
```-l /trained_model/model.h5```  or ```--loadweights /trained_model/model.h5``` is the path of weights to be loaded   
```-d ./test_data.csv```  or ```--data ./test_data.csv``` is the path of data to be predicted       
        

### 4 Notification of commercial use
Commercialization of this product is prohibited without our permission.


### 5 Citing this work
Our published data is collected from the iBond database. If you use this data in your work, please cite:    
```bash
1. H. An, X. Liu, W. Cai, X. Shao. AttenGpKa: A Universal Predictor of Solvation Acidity Using Graph Neural Network and Molecular Topology. Journal of Chemical Information and Modeling, 2024. DOI: 10.1021/acs.jcim.4c00449
2. J.-D. Yang, X.-S. Xue, P. Ji, X. Li, J.-P. Cheng, Internet Bond-energy Databank (pKa and BDE): iBonD Home Page. http://ibond.nankai.edu.cn or  http://ibond.chem.tsinghua.edu.cn.
```
