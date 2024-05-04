import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load data
def load_data():
    return pd.read_csv('online_shoppers_intention.csv')

data = load_data()

# Sidebar - Select features
st.sidebar.title('Select Features')
feature1 = st.sidebar.selectbox('Select first feature:', data.columns[5:])
feature2 = st.sidebar.selectbox('Select second feature:', data.columns[6:])

x = data[[feature1, feature2]].values

# Clustering
st.subheader('Clustering Analysis')

# Elbow Method
st.write('### Elbow Method')
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10,
               random_state=0, algorithm='lloyd', tol=0.001)
    km.fit(x)
    wcss.append(km.inertia_)

st.line_chart(wcss)

# Scatter Plot
st.write('### Scatter Plot')
num_clusters = st.slider('Number of Clusters:', min_value=2, max_value=10, value=2)
km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

plt.figure(figsize=(8, 6))
for cluster in range(num_clusters):
    plt.scatter(x[y_means == cluster, 0], x[y_means == cluster, 1], s=50, label=f'Cluster {cluster}')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'{feature1} vs {feature2}')
plt.legend()
st.pyplot(plt)

# Evaluation
st.subheader('Evaluation')

# Encode labels
le = LabelEncoder()
labels_true = le.fit_transform(data['Revenue'])
labels_pred = y_means

# Adjusted Rand Index
st.write('### Adjusted Rand Index')
adjusted_rand_index = metrics.adjusted_rand_score(labels_true, labels_pred)
st.write(f'Adjusted Rand Index: {adjusted_rand_index}')

# Confusion Matrix
cm = confusion_matrix(labels_true, labels_pred)

st.write('### Confusion Matrix')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Pass the current plt.figure() object to st.pyplot()
st.pyplot(plt.gcf())

# Normalize the confusion matrix
cm_sum = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(cm.astype('float'), cm_sum, where=cm_sum != 0)

st.write('### Normalized Confusion Matrix')
plt.figure(figsize=(8, 6))

# Plot normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Normalized Confusion Matrix')

# Pass the current plt.figure() object to st.pyplot()
st.pyplot(plt.gcf())
