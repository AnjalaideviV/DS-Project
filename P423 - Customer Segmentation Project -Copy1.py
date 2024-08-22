#!/usr/bin/env python
# coding: utf-8

#                                       Customer Personality Analysis                                

# #  Objectives :
-->   Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviour’s and concerns of different types of customers.

-->   Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.


       
Target: Need to perform clustering to summarize customer segments.


# # Import libraries  :
# 
# 
# 

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch


import warnings
warnings.filterwarnings('ignore')


# # Exploratory Data Analysis :

# In[71]:


# Dataset

data=pd.read_csv('marketing_campaign.csv')
data.head(10)


# In[72]:


# data shape :

data.shape


# In[73]:


# data Information :

data.info()


# In[74]:


# find the null values:

missing_values = data.isnull().sum()
missing_values


# In[75]:


missing_percentages = (missing_values / len(data)) * 100
print(missing_percentages)


# In[76]:


income_median=data['Income'].median()
income_median


# In[77]:


# Replace the null values :

data.fillna(income_median, inplace=True)
data.isnull().sum()


# In[78]:


# find duplicates in dataset

data.duplicated().sum()


# In[79]:


# value_counts of categorical column :

marital_Status = data['Marital_Status'].value_counts()
print(marital_Status) 


# In[80]:


data['Marital_Status'].unique()


# In[81]:


# value_counts of categorical column :

education = data['Education'].value_counts()
print(education)


# In[82]:


# Description of the dataset :

data.describe().T


# # Visualizations 
we can categorize customer into groups based on common themes or logical associations. 

1.Demographic Information
 --> ID
 --> Year_Birth
 --> Education
 --> Marital_Status
 --> Income
 --> Age
 
2.Household Information
 --> Kidhome
 --> Teenhome
 
3.Customer Relationship
 --> Dt_Customer (Date of customer registration)
 --> Recency (Number of days since the last purchase)
 
4. Purchase Behavior
 --> MntWines (Amount spent on wines)
 --> MntFruits (Amount spent on fruits)
 --> MntMeatProducts (Amount spent on meat products)
 --> MntFishProducts (Amount spent on fish products)
 --> MntSweetProducts (Amount spent on sweet products)
 --> MntGoldProds (Amount spent on gold products)
 --> NumDealsPurchases (Number of purchases made with a discount)
 --> NumWebPurchases (Number of purchases made through the web)
 --> NumCatalogPurchases (Number of purchases made using a catalog)
 --> NumStorePurchases (Number of purchases made directly in stores)
 --> NumWebVisitsMonth (Number of web visits in the last month)
 
5.Campaign Information
 --> AcceptedCmp1 (1 if customer accepted the offer in the 1st campaign, 0 otherwise)
 --> AcceptedCmp2 (1 if customer accepted the offer in the 2nd campaign, 0 otherwise)
 --> AcceptedCmp3 (1 if customer accepted the offer in the 3rd campaign, 0 otherwise)
 --> AcceptedCmp4 (1 if customer accepted the offer in the 4th campaign, 0 otherwise)
 --> AcceptedCmp5 (1 if customer accepted the offer in the 5th campaign, 0 otherwise)
 --> Response (1 if customer accepted the offer in the last campaign, 0 otherwise)
 --> Customer Feedback
 --> Complain (1 if customer complained in the last 2 years, 0 otherwise)
 
6.Cost and Revenue
 --> Z_CostContact (Fixed cost of contact)
 --> Z_Revenue (Revenue of the company)
# ##                                   1. Demographic Information

# In[83]:


# (A) Year of Birth

current_year = 2024
data['Age'] = current_year - data['Year_Birth']

# Plotting the histogram for Age
plt.figure(figsize=(18, 4))
sn.histplot(data['Age'], bins=30, kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()


# In[84]:


# (B) Education

plt.figure(figsize=(15, 4)) 
education.plot(kind='bar')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.title('Histogram of Education Levels')
plt.xticks(rotation=0)
plt.show()


# In[85]:


# (C) Marital Status

plt.figure(figsize=(18, 4)) 
marital_Status.plot(kind='bar')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.title('Histogram of Marital Status')
plt.xticks(rotation=0)
plt.show()


# In[86]:


# (D) Income
plt.figure(figsize=(18, 4))
sn.histplot(data['Income'], bins=30, kde=True)
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution')
plt.show()


# ## 2.Household Information

# In[87]:


kidhome_counts = data['Kidhome'].value_counts()
teenhome_counts = data['Teenhome'].value_counts()


# In[88]:


# Kidhome  and Teenhome Distribution

data_melted = data.melt(value_vars=['Kidhome', 'Teenhome'], var_name='HomeType', value_name='Count')
palette = {'Kidhome': 'skyblue', 'Teenhome': 'green'}

plt.figure(figsize=(18, 5))
sn.countplot(x='Count', hue='HomeType', data=data_melted, palette=palette)
plt.title('Distribution of Kids and Teens at Home')
plt.xlabel('Number at Home')
plt.ylabel('Count')
plt.legend(title='Home Type', loc='upper right')
plt.show()


# ## 3.Customer Relationship

# In[89]:


# Analyze

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
registration_year_month = data['Dt_Customer'].dt.to_period('Y').value_counts().sort_index()


# In[90]:


# Plotting customer registration dates

plt.figure(figsize=(16, 8))
registration_year_month.plot(kind='bar', color='skyblue')
plt.title('Customer Registration Over Time')
plt.xlabel('Registration Year-Month')
plt.ylabel('Number of Registrations')
plt.xticks(rotation=False)
plt.tight_layout()
plt.show()


# ## 4. Purchase Behavior

# In[91]:


#  the total amount spent for each category

totals = data[['MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds']].sum()

plt.figure(figsize=(16, 5))
sn.barplot(x=totals.index, y=totals.values, palette='viridis')
plt.title('Total Amount Spent on Various Products')
plt.xlabel('Product Category')
plt.ylabel('Total Amount Spent')
plt.xticks(rotation=False)
plt.show()


# In[92]:


# Number of Purchases and Visits

totals = data[['NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases']].sum()

plt.figure(figsize=(16, 5))
sn.barplot(x=totals.index, y=totals.values, palette='viridis')
plt.title('Total Number of Purchases and Visits')
plt.xlabel('Purchase/Visit Type')
plt.ylabel('Total Count')
plt.xticks(rotation=False)
plt.show()



# ## 5.Campaign Information

# In[93]:


# Bar plot for campaign acceptance and feedback

campaign_totals = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum()
feedback_totals = data['Complain'].sum()
plt.figure(figsize=(14, 7))
sn.barplot(x=campaign_totals.index.tolist() + ['Complain'], y=list(campaign_totals.values) + 
           [feedback_totals], palette='pastel')
plt.title('Campaign Acceptance and Customer Feedback')
plt.xlabel('Campaign / Feedback')
plt.ylabel('Total Count')
plt.xticks(rotation=45)
plt.show()


# ## 6.Cost and Revenue

# In[94]:


# Pie chart for cost and revenue

totals = data[['Z_CostContact', 'Z_Revenue']].sum()
plt.figure(figsize=(5, 5))
plt.pie(totals, labels=totals.index, autopct='%1.1f%%', colors=sn.color_palette('coolwarm'))
plt.title('Total Cost of Contact and Revenue')
plt.show()


# # Feature Engineering  

# In[95]:


data.head()


# In[96]:


le=LabelEncoder()

data['Education']=le.fit_transform(data['Education'])
data['Marital_Status']=le.fit_transform(data['Marital_Status'])
data["Age"] = 2021 - data["Year_Birth"]
data["Children"] = data["Kidhome"] + data["Teenhome"]

data.head()


# In[97]:


data1 = data[['Age','Education', 'Marital_Status','Income','Children','Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
              'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']]
data1.head()


# In[98]:


data1.shape


# In[99]:


pca=PCA(n_components=5)
pca_data=pca.fit_transform(data1)

E_varaice_ratio=pca.explained_variance_ratio_.sum()
print(E_varaice_ratio)


# #  Model Buildings 
Clustering is a technique used in machine learning to group similar data points together.


clustering techniques:

1. centriod based clustering (k-means)
2. connectivity based clustering  ( hierarchical clustering )
3. density based clustering  (DBSCAN)
4. Distribution based clustering (Gaussian distribution)
# ## 1.  K-Means clustering

# In[100]:


# Using Elbow method to find n_clusters

wcss=[]
for i in range(1,11):
    kmean = KMeans(n_clusters=i, init='k-means++', random_state= 0)
    kmean.fit(data1)
    wcss.append(kmean.inertia_)
    
plt.figure(figsize=(8,3))
sn.lineplot( wcss, marker='o',color='blue')
plt.xlabel('No of Clusters',size=15,color='blue')
plt.ylabel('Inertia',size=15,color='blue')
plt.title('Elbow Method')
plt.show()


# In[101]:


kmean_model = KMeans(n_clusters=3, random_state=42)
data1['cluster']= kmean_model.fit_predict(data1)

label = kmean_model.labels_
print(label)


# In[102]:


# Silhouette_Score

kmeans_score = silhouette_score(pca_data, label)
print('K-Mean Silhouette_Score : ', kmeans_score)


# In[103]:


# Plotting without specifying a palette
sn.countplot(x='cluster', data=data1)
plt.title('Customer Distribution Within Clusters')
plt.show()


# In[104]:


def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 4))
    sn.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis', s=100)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()
    
# Plotting the clusters for K-Means
kmean_model = KMeans(n_clusters=2, init='k-means++')
kmeans_labels = kmean_model.fit_predict(pca_data)
plot_clusters(pca_data, kmeans_labels, "K-Means Clusters")


# In[105]:


# create list of categories
count_cols = ['Education', 'Marital_Status', 'Children']

# Create a 2x2 grid of subplots
fig, ax1 = plt.subplots(2, 2, figsize=(16,8))

# Plotting the count plots in the first three subplots
for i, col in enumerate(count_cols):
    sn.countplot(x='cluster', data=data1, ax=ax1[i // 2, i % 2], hue=col)

# Remove the last subplot (bottom-right)
ax1[1, 1].set_visible(False)

# Display the plots
plt.show()


# ## 2. Hierarchical  / Agglomerative Clustering

# In[106]:


Agg_model = AgglomerativeClustering(n_clusters=2, affinity= 'euclidean', linkage= 'average')
cl = Agg_model.fit_predict(pca_data)
cl


# In[107]:


agg_score = silhouette_score(pca_data, cl)
print('Hierarchical Silhouette_Score : ', agg_score)


# In[124]:


# Apply Hierarchical Clustering and plot dendrogram

plt.figure(figsize=(8,4))
dendrogram = sch.dendrogram(sch.linkage(pca_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[109]:


# Plotting the clusters for Agglomerative Clustering

agg_model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
agg_labels = agg_model.fit_predict(pca_data)
plot_clusters(pca_data, agg_labels, "Agglomerative Clustering Clusters")


# ### 3. Density Based Clustering  (DBSCAN)

# In[110]:


neighbors=NearestNeighbors(n_neighbors=2)
neighbors_fit=neighbors.fit(pca_data)


# In[111]:


distance, indices= neighbors_fit.kneighbors(pca_data)
distance=np.sort(distance, axis=0)
distance=distance[:, 1]
plt.plot(distance)


# In[112]:


dbscan_model = DBSCAN(eps=500000, min_samples=1000)
label = dbscan_model.fit_predict(pca_data)
print(label)


# In[113]:


dbscan_score = silhouette_score(pca_data, label)
print('DBSCAN Silhouette_Score : ', dbscan_score)


# In[114]:


# Plotting the clusters for DBSCAN

dbscan_model = DBSCAN(eps=500000, min_samples=1000)
dbscan_labels = dbscan_model.fit_predict(pca_data)
plot_clusters(pca_data, dbscan_labels, "DBSCAN Clusters")


# # Clusters Summary 

# In[115]:


from tabulate import tabulate

# Print silhouette scores
silhouette_scores = {
    'Clustering Method': ['K-Means', 'DBSCAN', 'Agglomerative Clustering'],
    'Silhouette Score': [kmeans_score, dbscan_score, agg_score]
}

silhouette_df = pd.DataFrame(silhouette_scores)
print(tabulate(silhouette_df, headers='keys', tablefmt='fancy_grid'))


# In[116]:


# Plot the Cluster Summary

def plot_clusters(data, labels, title, ax):
    sn.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis', s=100, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(title='Cluster')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# K-Means Clustering
kmean_model = KMeans(n_clusters=2, init='k-means++')
kmeans_labels = kmean_model.fit_predict(pca_data)
plot_clusters(pca_data, kmeans_labels, "K-Means Clusters", axes[0])

# Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
agg_labels = agg_model.fit_predict(pca_data)
plot_clusters(pca_data, agg_labels, "Agglomerative Clusters", axes[1])

# DBSCAN Clustering
dbscan_model = DBSCAN(eps=500000, min_samples=1000)
dbscan_labels = dbscan_model.fit_predict(pca_data)
plot_clusters(pca_data, dbscan_labels, "DBSCAN Clusters", axes[2])

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# # Customer segments.

# In[117]:


products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data1['Age'] = 2024 - data['Year_Birth']
age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
data1['Age_Group'] = pd.cut(data1['Age'], bins=age_bins)

spending_by_age = data1.groupby('Age_Group')[products].mean()
spending_by_age.plot(kind='bar', figsize=(14,5))
plt.title('Average Spending on Products by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Spending')
plt.show()


# In[118]:


childhome_distribution = data1.groupby('Children')[products].mean()

childhome_distribution.plot(kind='bar', figsize=(14, 5))
plt.title('Average Consumption by Number of Children in Home')
plt.xlabel('Number of Children in Home')
plt.ylabel('Average Consumption')
plt.xticks(rotation=False)
plt.legend(title='Product Categories')
plt.tight_layout()
plt.show()


# In[119]:


bins = [0, 25000, 50000, 75000, 100000, data1['Income'].max()]
labels = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']


data1['IncomeCategory'] = pd.cut(data1['Income'], bins=bins, labels=labels)
income_distribution = data1.groupby('IncomeCategory')[products].mean()

# Plotting the bar chart
income_distribution.plot(kind='bar', figsize=(16, 6))
plt.title('Average Consumption by Income Category')
plt.xlabel('Income Category')
plt.ylabel('Average Consumption')
plt.xticks(rotation=False)
plt.legend(title='Product Categories')
plt.tight_layout()
plt.show()


# In[120]:


response_by_age_group = data1.groupby('Age_Group')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                                    'AcceptedCmp4', 'AcceptedCmp5']].mean()
response_by_age_group.plot(kind='bar', figsize=(16,5))
plt.title('Acceptance of Promotional Offers by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Acceptance Rate')
plt.xticks(rotation=False)
plt.show()


# In[121]:


response_by_age_group = data1.groupby('IncomeCategory')[['NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases']].mean()
response_by_age_group.plot(kind='bar', figsize=(16,5))
plt.title('Acceptance of Promotional Offers by Income Category')
plt.xlabel('Income Category')
plt.ylabel('Acceptance Rate')
plt.xticks(rotation=False)
plt.show()


# In[122]:


response_by_age_group = data1.groupby('IncomeCategory')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                                    'AcceptedCmp4', 'AcceptedCmp5']].mean()
response_by_age_group.plot(kind='bar', figsize=(16,5))
plt.title('Acceptance of Promotional Offers by Income Category')
plt.xlabel('Income Category')
plt.ylabel('Acceptance Rate')
plt.xticks(rotation=False)
plt.show()


# # Conclusion

In our analysis above, we have concluded following:

1.Product analysis 

--> Senior (80-90) customers tend to buy more wines as compared to other age group.
--> Customers have no children buy more products as compared to those who have more children.
--> Customers with upper-higher incomes tend to buy more products as compared to middle and low incomes. 



2. Promotion analysis 

--> Senior(80-90) and adult (20-30) customers accept more campaigns in 3 & 5 as compared with other age group.
--> Customers with upper-higher & high incomes accept more campaigns as compared to middle and low incomes. 
--> Customers with low incomes to upper-middle income tend to visit the store more frequently and customer with high-income tend  make purchases by catalog.



These segments provide valuable insights for targeted marketing strategies, allowing businesses to tailor their campaigns and promotions to specific customer groups.
# In[ ]:





# In[ ]:





# In[ ]:




