# AttenGpKa
AttenGpKa is a graph neural network for predicting pKa of different molecules in various solvents.

### 1 environment requirements  
The required packages and their versions are included in the [requirements.txt](./requirements.txt) file. Run the following commands to build your environment:   
```bash
conda create -y -n AttenGpKa python==3.8.11
conda activate AttenGpKa
pip install -r requirements.txt
conda install -y ipython
```
     
### 2 usage
We open-sourced all the data, codes, and models used in this work. Additionally, we have developed a user-friendly software for Windows OS, which allows anyone to easily use our model. 

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
J.-D. Yang, X.-S. Xue, P. Ji, X. Li, J.-P. Cheng, Internet Bond-energy Databank (pKa and BDE): iBonD Home Page. http://ibond.nankai.edu.cn or  http://ibond.chem.tsinghua.edu.cn.
```
