from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
D=load_wine()
Wine=pd.DataFrame(D.data,columns=D.feature_names)

km=KMeans(n_clusters=3)
prediction=km.fit_predict(Wine[['alcohol','ash']])
Wine['cluster']=prediction.copy()
centre=km.cluster_centers_
w1=Wine[Wine.cluster==0]
w2=Wine[Wine.cluster==1]
w3=Wine[Wine.cluster==2]
plt.scatter(w1.alcohol,w1['ash'],color='yellow')
plt.scatter(w2.alcohol,w2['ash'],color='orange')
plt.scatter(w3.alcohol,w3['ash'],color='black')
plt.scatter(centre[:,0],centre[:,1],color='purple',marker="*")
