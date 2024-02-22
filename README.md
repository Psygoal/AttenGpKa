# AttenGpKa
AttenGpKa is a graph neural network for predicting pKa of different molecules in various solvents.

### 1 environment requirements

* python 3.8.11
* tensorflow 2.11.0
* rdkit 2023.3.1
* pandas 1.5.3
* numpy 1.22.4
* matplotlib 3.7.1
* scikit-learn 1.2.2
     
### 2 usage
We open-source all data, codes and models used in this work. A user-friendly software for Windows OS is also developed for anyone who wants to use our model.

### 3 model training and predicting
We have open-sourced all the data, code, and models used in this work. Additionally, we have developed a user-friendly software for Windows OS, which allows anyone to easily use our model.   

If users want to train their own model with our provided training data:  
```bash
python AttenGpka.py -m train -s ./test.h5 -f 0
```
```-m train```  or ```--mode train``` is that the running mode of code is training   
```-s ./test.h5```  or ```--saveweights ./test.h5``` is the path of the trained model to be saved       
```-f 0```  or ```--fold 0``` is the way to split traing and test dataset       
         
If users want to predict pKa with the avilable model:
```bash
python AttenGpka.py -m predict -l ./trained_model/0_crossvalidation.h5 -d ./test_data.csv
```
```-l /trained_model/0_crossvalidation.h5```  or ```--loadweights /trained_model/0_crossvalidation.h5``` is the path of weights to be loaded   
```-d ./test_data.csv```  or ```--data ./test_data.csv``` is the path of data to be predicted       
        

### 4 Notification of commercial use
Commercialization of this product is prohibited without our permission.


### 5 Citing this work
Our published data is collected from the iBond database. If you use this data in your work, please cite:    
```bash
J.-D. Yang, X.-S. Xue, P. Ji, X. Li, J.-P. Cheng, Internet Bond-energy Databank (pKa and BDE): iBonD Home Page. http://ibond.nankai.edu.cn or  http://ibond.chem.tsinghua.edu.cn.
```
