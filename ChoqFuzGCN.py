## -*- coding: -*-
## ChoqFuzGCN ,  GCN applied to GE and CNV then extracted features from both the GCN are concatenated with CLN features. The final classification is done using Choquet Integral


# 
#steller graph has been used for graph convolution
import sys
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.0.0rc1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.0.0rc1, but a different version {sg.__version__} is installed.  "
    ) from None


import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from stellargraph import StellarGraph
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from numpy import savetxt
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed
seed(1)

from imblearn.over_sampling import SMOTE

def smote_upsample(stacked_feature_train, y_train_rf): # apply smote for over_sampling
  oversample = SMOTE()
  #print("before upsampling shape: \n ")
  #print(y_train_rf)

  X, y = oversample.fit_resample(stacked_feature_train, y_train_rf)
  #print(" after upsampling shape: \n ")
  #print(y)
  return(X,y)



# choqet fuzzy==========

import numpy as np
from sympy import solve, symbols
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve,auc
from sklearn import svm
#from sklearn.naive_bayes import GaussianNB

#X=X1=ensemble_model(stacked_feature_train,y_train,stacked_feature_test, y_te)

def ensemble_model(X_train,y_train,X_test, y_test):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)# apply train test split

    cla1 = RandomForestClassifier(n_estimators = 70, criterion = 'entropy', random_state = 22)
    cla1.fit(X_train,y_train)  
    y_pred1=cla1.predict(X_test)


    cla2 = SVC(kernel = 'rbf', random_state = 0)   
    cla2.probability=True
    cla2.fit(X_train, y_train)
    y_pred2=cla2.predict(X_test)

    cla3 = LogisticRegression(random_state=0)
    cla3.fit(X_train, y_train)
    y_pred3=cla3.predict(X_test)

    # prediction on validation set
    y_pred1V=cla1.predict(X_valid)
    y_pred2V=cla2.predict(X_valid)
    y_pred3V=cla3.predict(X_valid)
    
    # calculate prediction probability
    y_pred1_prob=cla1.predict_proba(X_test)[:, 1] 
    y_pred2_prob=cla2.predict_proba(X_test)[:, 1]   
    y_pred6_prob=cla6.decision_function(X_test)
    

    # accuracy evaluation for each classifier based on test set      
    p1=metrics.accuracy_score(y_test, y_pred1)
    p2=metrics.accuracy_score(y_test, y_pred2) 
    p3=metrics.accuracy_score(y_te, y_pred3)
   

    # test sample prediction
    y_pred1=list(y_pred1) #rf
    y_pred2=list(y_pred2)# rbf
    y_pred6=list(y_pred3) # lr
  
    model_predictions=[y_pred1_prob,y_pred2_prob,y_pred3_prob]

    # accuracy evaluation for each classifier based on validation set      
    v1=metrics.accuracy_score(y_valid, y_pred1V)
    v2=metrics.accuracy_score(y_valid, y_pred2V)
    v3=metrics.accuracy_score(y_valid, y_pred3V)
    sum=v1+v2+v3
    m1=v1/sum
    m2=v2/sum
    m3=v3/sum
    measures=[m1,m2,m3] #fuzzy measures of validation set
    X=[p1,p2,p6] # prediction value of each of the classifier     
    # choqet fuzzy ensemble
    final_prediction=ensemble(model_predictions, measures, mode='choquet')
    
    return np.array(final_prediction)
    
    
    



def get_lambda2(measures):
    g1, g2, g3 = measures
    x = symbols('x')
    lmbd = solve((g1 * g2 * g3) * x ** 3 + 
                 (g1 * g2 + g2 * g3 + g1 * g3) * x **2  + 
                 (g1 + g2 + g3-1)*x , x)
    
    return lmbd[1]
    



def get_lambda(measures):
    g1, g2 = measures
    x = symbols('x')
    lmbd = solve((g1 * g2 ) * x **2  + 
                 (g1 + g2-1)*x , x)
    
    return lmbd[1]


def choquet_fuzzy_integral(X, lmbd):	
	sorted_data = np.sort(X, order="prediction_score")[::-1]
	f_prev = sorted_data[0][1]	
	pred = sorted_data[0][0] * sorted_data[0][1]	
	for i in range(1, len(sorted_data)):
		f_cur = f_prev + sorted_data[i][1] + lmbd * sorted_data[i][1] * f_prev
		pred = pred + sorted_data[i][0] * (f_cur - f_prev)
		f_prev = f_cur
	return pred


def ensemble(model_predictions, measures, mode='choquet'):
    models_count = len(model_predictions)    
    assert models_count == len(measures)
    
    lmbd = get_lambda2(measures)
    dtype = [('prediction_score', float), ('fuzzy_measure', float)]
    final_predictions = list()   
    for i in range(len(model_predictions[0])):
        if mode == 'choquet':    
            score_values = [(model_predictions[j][i], measures[j]) for j in range(models_count)]         
            data_belong = np.array(score_values, dtype=dtype)
            #print("data belong",data_belong)
            x_belong_agg = choquet_fuzzy_integral(data_belong, lmbd)
            #print(x_belong_agg)
        else:
            break
    # print("final prediction is",final_predictions)    
    return final_predictions 




#=============== Graph Based Feature extraction===============================

def StellerGraphConvolution(train_LABELMain, train_LABEL,test_LABEL,val_LABEL,G,node_label,str1,Modality,fold):
      print(train_LABELMain.value_counts().to_frame())     
      print(train_LABEL.value_counts().to_frame())
      print(test_LABEL.value_counts().to_frame())
      print(val_LABEL.value_counts().to_frame())
      target_encoding = preprocessing.LabelBinarizer()

      print(train_LABEL)

      train_targets = target_encoding.fit_transform(train_LABEL)
      train_targets = to_categorical(train_targets, num_classes=2)
      val_targets = target_encoding.transform(val_LABEL)
      val_targets = to_categorical(val_targets, num_classes=2)
      test_targets = target_encoding.transform(test_LABEL)
      test_targets = to_categorical(test_targets, num_classes=2)
      

      #=================================================
      train_targets_main = target_encoding.fit_transform(train_LABELMain)
      train_targets_main = to_categorical(train_targets_main, num_classes=2)
      #=================================================

      generator = FullBatchNodeGenerator(G, method="gcn")
      ######
      train_gen_main = generator.flow(train_LABELMain.index, train_targets_main)
      ######
      train_gen = generator.flow(train_LABEL.index, train_targets)
      if Modality==1 :
        gcn = GCN(    
            layer_sizes=[200, 150 , 100], activations=["relu", "relu","relu"], generator=generator, dropout=0.5
        )
      elif Modality==2: 
         gcn = GCN(          
            layer_sizes=[300, 200,150], activations=["relu", "relu","relu"], generator=generator, dropout=0.5
        )   
      elif Modality==3: 
         gcn = GCN(            
            layer_sizes=[16, 16,16], activations=["relu", "relu","relu"], generator=generator, dropout=0.5
        ) 
      x_inp, x_out = gcn.in_out_tensors()
      print(x_out)

      predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

      model = Model(inputs=x_inp, outputs=predictions)
      #print(x_inp)
      model.compile(
          optimizer=optimizers.Adam(lr=0.01),
          loss=losses.categorical_crossentropy,
          metrics=["acc"],
      )

      val_gen = generator.flow(val_LABEL.index, val_targets)


      patience_=10
      es_callback = EarlyStopping(monitor="val_acc", patience=patience_, restore_best_weights=True)

      history = model.fit(
          train_gen,
          epochs=200,
          validation_data=val_gen,
          verbose=2,
          shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
          callbacks=[es_callback],
      )

      sg.utils.plot_history(history)

      test_gen = generator.flow(test_LABEL.index, test_targets)
      test_metrics = model.evaluate(test_gen)
      # ============================================
      # all_nodes = node_label.index
      # train_gen_main = generator.flow(train_LABELMain.index, train_targets_main)
      # all_gen = generator.flow(all_nodes)
      test_metrics = model.evaluate(train_gen_main)
      # ================================================
      print("\nModality============", Modality)
      text1=str1+", Modality: "+str(Modality)+", fold: "+str(fold)+'\n'
      file3.write(text1)
      for name, val in zip(model.metrics_names, test_metrics):
          print("\t{}: {:0.4f}".format(name, val))
          file3.write(str(name))
          file3.write("\t")
          file3.write(str(val))
          file3.write("\n")

      all_nodes = node_label.index
      all_gen = generator.flow(all_nodes)
      all_predictions = model.predict(all_gen)
      node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

      node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
      df = pd.DataFrame({"Predicted": node_predictions, "True": node_label})
      df.head(20)

     

      #/”predict” the node embedding vectors. 
      embedding_model = Model(inputs=x_inp, outputs=x_out)
      train_emb = embedding_model.predict(train_gen_main)
      test_emb = embedding_model.predict(test_gen)
      all_emb= embedding_model.predict(all_gen)
      #print(emb)
      train_result = train_emb[0,:, :]
      test_result= test_emb[0,:, :]
      all_result=all_emb[0,:, :]
      #print(result.shape)
      file_Name=str1+"_6mod_Metabric_Embedding.csv"
      savetxt(file_Name, all_result, delimiter=',')
      return train_result, test_result





#=========== MAIN ==========================

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score



EDGE_Connection = pd.read_csv(
 
    "/DATA/susmita_2121cs34/susmita/GRAPH based work/6modBreastCancer/edges/cnv_edges.cites",# collected from TCGA 6 mod dataset
  
    sep="\t",  # tab-separated
    # sep=" ",
    header=None,  # no heading row
    names=["target", "source"],  # set our own names for the columns
)
EDGE_Connection.shape

DATASET_content = pd.read_csv(    
    "/DATA/susmita_2121cs34/susmita/GRAPH based work/6modBreastCancer/data/file_cnv.csv",# collected from TCGA 6 mod dataset
    sep=",",  # tab-separated
    # sep="\t", # space-separated
    header=None,  # no heading row
    #names=["id", *TCGA_feature_names, "labels"],  # set our own names for the columns
)

DATASET_content.rename(columns={ DATASET_content.columns[0]: "id" }, inplace = True)
DATASET_content.rename(columns={ DATASET_content.columns[-1]: "labels" }, inplace = True)
#DATASET_content
DATASET_str_labels = DATASET_content.set_index("id")
DATASET_no_labels = DATASET_str_labels.drop(columns="labels")

#print("shape is",DATASET_no_labels)
TCGA_no_labels = StellarGraph({"paper": DATASET_no_labels}, {"cites": EDGE_Connection})
G=TCGA_no_labels
node_label = DATASET_str_labels["labels"]

#=============================================================================================================
#==== 2nd Modality ==================================================================================

EDGE_Connection1 = pd.read_csv(
  
    "/DATA/susmita/GRAPH based work/6modBreastCancer/edges/mir_edges.cites",
    # collected from Metabtic 3mod dataset    
    sep="\t",  # tab-separated
    # sep=" ",
    header=None,  # no heading row
    names=["target", "source"],  # edge exist between source node and target node
)


DATASET_content1 = pd.read_csv(   
  
    "/DATA/susmita_2121cs34/susmita/GRAPH based work/6modBreastCancer/data/file_mir.csv",# collected from TCGA gene expression  
    sep=",",  # tab-separated   
    header=None,  # no heading row    
)

# print("Gene exp shape",DATASET_content1.shape)

DATASET_content1.rename(columns={ DATASET_content1.columns[0]: "id" }, inplace = True)
DATASET_content1.rename(columns={ DATASET_content1.columns[-1]: "labels" }, inplace = True)
#DATASET_content
DATASET_str_labels1 = DATASET_content1.set_index("id")
DATASET_no_labels1 = DATASET_str_labels1.drop(columns="labels")

#print("shape is",DATASET_no_labels)

TCGA_no_labels1 = StellarGraph({"paper": DATASET_no_labels1}, {"cites": EDGE_Connection1})
G1=TCGA_no_labels1
node_label1 = DATASET_str_labels1["labels"]

#==============================================================================================================
#============== 3rd modality==================================================================================
"
df3 = pd.read_csv('/DATA/susmita/GRAPH based work/6modBreastCancer/data/file_cln.csv',header = None) # read Clinical data
array = df3.values

X3 = array[:,1:-1]
X3 = preprocessing.scale(X3)
array1 = df3.values
y3 = array1[:,-1]


#=========================================================================================================



file2 = open("File2.txt","w+")
file3 = open("StorefoldRes.txt","w+")

X_1=DATASET_content.values
row1,col1=X_1.shape
col1=col1-1
X_2=DATASET_content1.values
X_1=X_1[:,1:col1]
row2,col2=X_2.shape
col2=col2-1
X_2=X_2[:,1:col2]
# X3=df3.values
ACC_=0
MCC_=0
PRE_=0
SEN_=0
SPE_=0
BALN=0
F1_=0

print("node label",node_label)

for itr in range (0,1):
      AVG_SENSITIVITY=0
      AVG_SPECIFICITY=0 
      AVG_PRECISION=0
      avg_f1=0
      avg_acc=0
      avgMcc=0
      avgBalAcc=0
      no_of_fold=10
      i=1
      kf=StratifiedKFold(n_splits=no_of_fold, random_state=22, shuffle=True)
      for train_index,test_index in kf.split(DATASET_no_labels,node_label):
            # print(train_index)
            # print(test_index)
            # break
            print("fold number ################################################",i)
            X3_train, X3_test = X3[train_index], X3[test_index]
            y3_train, y3_test = y3[train_index], y3[test_index]

            # print(type(train_index))
            train_index=train_index+1
            test_index=test_index+1

            # print(node_label)
            
            #  # get the training sample index
            train_LABELMain, test_LABEL = node_label[train_index], node_label[test_index] 
            #  # get the training label index
            train_LABEL, val_LABEL = model_selection.train_test_split(train_LABELMain, test_size=0.10, random_state=20,stratify=train_LABELMain) 

          #   for cnv=====
            str1="CNV"
            train_embd1, test_embd1=StellerGraphConvolution(train_LABELMain, train_LABEL,test_LABEL,val_LABEL,G,node_label,str1,1,i)
          #   for GSE=====
            
            str1="GSE" 
            train_embd2, test_embd2=StellerGraphConvolution(train_LABELMain, train_LABEL,test_LABEL,val_LABEL,G1,node_label,str1,2,i)
         
            # call Choquet Ensemble==============================
            
            X_train=np.concatenate((train_embd1,train_embd2,X3_train), axis=1)
            X_test=np.concatenate((test_embd1,test_embd2,X3_test), axis=1)
        
            # X_train=np.concatenate((train_embd1,train_embd2,train_embd3), axis=1)
            # X_test=np.concatenate((test_embd1,test_embd2,test_embd3), axis=1)
            X_train,y3_train=smote_upsample( X_train,y3_train)
            model_predictions=ensemble_model(X_train,y3_train,X_test,y3_test)
            # print("model_predictions:========\n",model_predictions)
             
            l=[]
            for element in model_predictions:
                if element >0.5: #0.75
                  l.append(1)
                else:
                  l.append(0)
                    
            y_pred1=l
              
              
            cm1 = confusion_matrix(y3_test,y_pred1)
            #print('Confusion Matrix : \n', cm1)

            #report = classification_report(y_test, y_pred1)
            #print('rf Classification Report: \n {}'.format(report))
            # cm1 = confusion_matrix(y3_test,y_pred1)
            #print('Confusion Matrix : \n', cm1)

            TP=cm1[1][1]
            TN=cm1[0][0]
            FP= cm1[0][1]
            FN=cm1[1][0]

            #print("TP,TN,FP,FN",TP,TN,FP,FN)
            
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP/(TP+FN)            
            # Specificity or true negative rate
            TNR = TN/(TN+FP)             
            # Precision or positive predictive value
            PPV = TP/(TP+FP)         
            # F- measure          
            f1_value=(2*PPV*TPR)/(PPV+TPR)            
            AVG_SENSITIVITY=AVG_SENSITIVITY+  TPR
            AVG_SPECIFICITY=AVG_SPECIFICITY+TNR
            AVG_PRECISION=AVG_PRECISION+ PPV
            # balanced accuracy
            b_ac=(TPR+TNR)/2
            avgBalAcc=avgBalAcc+b_ac            
            avg_f1=avg_f1+f1_value  
            # accuracy
            acc= (TP+TN)/(TP+FP+TN+FN)
            avg_acc=avg_acc+acc
            mcc=matthews_corrcoef(y3_test,y_pred1)
            avgMcc=avgMcc+mcc        
           
            i=i+1
      
      # calculate score of all metices for ove 
      avg_acc=avg_acc/10
      avg_acc = round(avg_acc, 3)
      avgMcc=avgMcc/10
      avgMcc = round(avgMcc, 3)
      AVG_PRECISION=AVG_PRECISION/10
      AVG_PRECISION = round(AVG_PRECISION, 3)
      AVG_SENSITIVITY=AVG_SENSITIVITY/10
      AVG_SENSITIVITY = round(AVG_SENSITIVITY, 3)
      AVG_SPECIFICITY=AVG_SPECIFICITY/10
      AVG_SPECIFICITY = round(AVG_SPECIFICITY, 3)
      avgBalAcc=avgBalAcc/10
      avgBalAcc = round(avgBalAcc, 3)
      #AVG_PRAUC=AVG_PRAUC/10
      avg_f1=avg_f1/10
      avg_f1 = round(avg_f1, 3)

      print("Average acc, avgMcc , avg precision , AVG_SENSITIVITY, AVG_SPECIFICITY, avgBalAcc, avg f1 score ")  
      print(avg_acc,",",avgMcc,",",AVG_PRECISION,",",AVG_SENSITIVITY,",",AVG_SPECIFICITY,",",avgBalAcc,",", avg_f1)  
     

file2.close()
file3.close()
