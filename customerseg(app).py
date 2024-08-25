import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer

def segment_customers(input_data, preprocessor, kmeans_model):
    input_df = pd.DataFrame(input_data, columns=['Income', 'Kidhome', 'Teenhome', 'Age', 'Partner', 'Education_Level'])
    
    # Ensure correct data types
    input_df = input_df.astype({
        'Income': 'float64',
        'Kidhome': 'int64',
        'Teenhome': 'int64',
        'Age': 'int64',
        'Partner': 'object',
        'Education_Level': 'object'
    })

    try:
        # Apply preprocessing
        input_df_transformed = preprocessor.transform(input_df)
        
        # Predict cluster labels using K-Means
        cluster_label = kmeans_model.predict(input_df_transformed)
        
        # Map the cluster label to a readable string
        cluster_labels = {
            0: 'Cluster 0',
            1: 'Cluster 1'
        }
        
        return cluster_labels.get(cluster_label[0], f'Cluster {cluster_label[0]}')
    except Exception as e:
        return f"Error: {e}"

def plot_clusters(X_transformed, kmeans_model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=kmeans_model.labels_, cmap='viridis', s=50)
    plt.colorbar(label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.savefig("C:\\Users\\Lenovo-PC\\Downloads\\clusters.png")
    plt.close()

def show_cluster_plot():
    img = plt.imread("C:\\Users\\Lenovo-PC\\Downloads\\clusters.png")
    st.image(img, caption='Cluster Visualization')

def suggest_offers(cluster_label):
    # Define offers for each cluster
    offers = {
        'Cluster 0': 'Offer 1: 10% discount on all items!',
        'Cluster 1': 'Offer 2: Buy one get one free on selected items!'
    }
    
    return offers.get(cluster_label, 'No offer available.')

def main():
    st.title('Customer Segmentation Web App')

    # Load the K-Means model and preprocessing pipeline
    try:
        with open("C:\\Users\\Lenovo-PC\\Downloads\\kmeans_model.pkl", 'rb') as file:
            kmeans_model = pickle.load(file)
        with open("C:\\Users\\Lenovo-PC\\Downloads\\preprocessor.pkl", 'rb') as file:
            preprocessor = pickle.load(file)
        st.success("Model and preprocessor loaded successfully!")
        
        # Prepare data for visualization
        train_data = pd.DataFrame({
            'Income': [1000, 2000, 1500],
            'Kidhome': [1, 0, 2],
            'Teenhome': [0, 1, 0],
            'Age': [30, 40, 35],
            'Partner': ['Yes', 'No', 'Yes'],
            'Education_Level': ['Basic', 'Master', 'PhD']
        })

        X_train = train_data[['Income', 'Kidhome', 'Teenhome', 'Age', 'Partner', 'Education_Level']]
        X_train_transformed = preprocessor.transform(X_train)
        plot_clusters(X_train_transformed, kmeans_model)
        
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    # Input form
    income = st.number_input("Income", min_value=0.0, format="%.2f")
    kidhome = st.number_input("Kids at Home", min_value=0, format="%d")
    teenhome = st.number_input("Teens at Home", min_value=0, format="%d")
    age = st.number_input("Age", min_value=0, format="%d")
    partner = st.selectbox("Partner", ["Yes", "No"])
    education_level = st.selectbox("Education Level", ["Basic", "Master", "PhD", "Other"])

    # Convert categorical input to numeric
    partner_binary = 1 if partner == "Yes" else 0
    education_level_encoded = {
        "Basic": 1,
        "Master": 2,
        "PhD": 3,
        "Other": 4
    }[education_level]

    if st.button("Segment Customer"):
        result = segment_customers(
            [[income, kidhome, teenhome, age, partner_binary, education_level_encoded]],
            preprocessor,
            kmeans_model
        )
        st.success(result)
        
        # Display offers
        offer = suggest_offers(result)
        st.write(f"Suggested Offer: {offer}")

    # Show cluster plot
    if st.checkbox("Show Cluster Visualization"):
        show_cluster_plot()

if __name__ == '__main__':
    main()
