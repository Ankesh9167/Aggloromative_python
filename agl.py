# %matplotlib inline
import matplotlib.pyplot as plt  
import pandas as pd  
import csv
import numpy as np
import math  
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
data = pd.read_csv('plot.csv')   
data = data.iloc[:].values 
print(" Input data is",data)
def CalculateDistance(p1,q1,p2,q2):
    sq1=(p2-p1)*(p2-p1)
    sq2=(q2-q1)*(q2-q1)
    return round((math.sqrt(sq1+sq2)),3)


def GenerteMatrix(n,x,y):
    global matrix
    a = np.tile(np.nan, (n, n))
    for i in range(n):
        for j in range(n):
            if(i==j):
                a[i][j]=0.00
            elif(j<i):
                a[i][j]=CalculateDistance(x[i],y[i],x[j],y[j])
            else:
     
               a[i][j]=0.00
    return a
x=[]	    
y=[]
a=[]
with open('plot.csv', 'r') as benFile:
    benReader = csv.reader(benFile)
    for row2 in benReader:
        a.append(row2)
benFile.close()
n=len(a)

n=n-1
print(n)
k=0
z=0
for i in range(1,n+1):
    for j in range(0,2):
        if(j==0):
            x.append(a[i][j])
        else:
            y.append(a[i][j])
x = [float(i) for i in x]
y = [float(i) for i in y]
print("----------------------OUTPUT---------------------------")
for i in range(n-1):
    a=GenerteMatrix(n,x,y) 
    n=n-1
    print("--------------------Iteration",i,"----------------------")
    print(a)

plt.figure(figsize=(10, 7))  
plt.title("Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward'))  
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
op=cluster.fit_predict(data)
c1=[]
c2=[]
for i in range(len(op)):
    if(op[i]==0):
        c1.append("p"+str(i))
    else:
        c2.append("p"+str(i))
for i in range(len(op)):
    if(len(c1)>1):
        v1=c1.pop()
        v2=c1.pop()
        print("cluster",i ," Merging between ---->",v1, "and", v2)
        c1.append(v1+v2)
    elif(len(c2)>1):
        v1=c2.pop()
        v2=c2.pop()
        print("cluster",i ," Merging between ---->",v1, "and", v2)
        c2.append(v1+v2)
v1=c1.pop()
v2=c2.pop()
print("cluster",i ," Merging between ---->",v1, "and", v2)

plt.show()

