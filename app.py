import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("AI-ML Retail Customer Intelligence System")
st.info("You can upload your own retail CSV dataset from the sidebar to run customer segmentation.")
# Sidebar file upload
st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)
# Load data
if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    customer_data = pd.read_csv("QVI_purchase_behaviour.csv")

transaction_data = pd.read_excel("QVI_transaction_data.xlsx")

# Merge datasets
data = pd.merge(transaction_data, customer_data, on="LYLTY_CARD_NBR")

# Feature engineering
customer_features = data.groupby("LYLTY_CARD_NBR").agg({
    "TOT_SALES": "sum",
    "PROD_QTY": "sum"
}).reset_index()

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_features["Cluster"] = kmeans.fit_predict(
    customer_features[["TOT_SALES","PROD_QTY"]]
)

st.subheader("Customer Segmentation Data")
st.dataframe(customer_features.head())

# Plot
fig, ax = plt.subplots()
sns.scatterplot(
    data=customer_features,
    x="PROD_QTY",
    y="TOT_SALES",
    hue="Cluster",
    palette="Set1",
    ax=ax
)

ax.set_title("Customer Segmentation (ML Clusters)")
st.pyplot(fig)
import streamlit as st

st.subheader("Key Business Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Customers", customer_features.shape[0])

with col2:
    st.metric("Total Sales", round(customer_features["TOT_SALES"].sum(),2))
st.subheader("Customer Cluster Distribution")

cluster_counts = customer_features["Cluster"].value_counts()

st.bar_chart(cluster_counts)

customer_id = st.selectbox(
    "Select Customer ID",
    customer_features["LYLTY_CARD_NBR"]
)

filtered_data = customer_features[
    customer_features["LYLTY_CARD_NBR"] == customer_id
]

st.write(filtered_data)

st.download_button(
    label="Download Customer Data",
    data=customer_features.to_csv(index=False),
    file_name="customer_segmentation.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("AI-ML Retail Customer Intelligence Dashboard | Built by Dev Balyan")

