# create graph edges base on Pearson correlation 
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from numpy import nan

file1 ="/ data/ breast cancer/withIndex/file_cnv.csv"
#file1="/ data/ breast cancer/withIndex/file_mir.csv"


#==========================for CNV data 

df = pd.read_csv(file1,header=None)
df.replace('', nan)
# print("sum of null",df.isnull().sum())
df.fillna(df.mean(), inplace=True)
     
sample,total_feature1=df.shape
print("sample ",sample, "feature",total_feature1)
X1 = df.values  




with open('cnv_edges.cites', 'w') as f:
  #print(X1[0].shape)
  sample1=sample
  #sample1=20
  count=0
  for i in range (0,sample1):
    for j in range (i+1,sample1):
      corr, _ = pearsonr(X1[i],X1[j])
      print(" ",i," ",j,": ",corr)
      #if(corr>=0.4 or corr<=-0.4):
      if(corr>=0.998):
        count=count+1
        f.write(str(i+1))
        f.write('\t')
        f.write(str(j+1))  
        #f.write('\t')
        #f.write(str(corr))       
        f.write('\n')
f.close()
print("total connection=",count)



