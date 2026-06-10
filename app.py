import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.pdfgen import canvas
from io import BytesIO

# PAGE SETTINGS
st.set_page_config(
    page_title="Gene Expression Analyzer",
    layout="wide"
)

# TITLE
st.title("Gene Expression Clustering and Pattern Discovery")

st.markdown("""
### Alzheimer's Disease Gene Expression Analysis

This application performs:
- Data normalization
- K-Means clustering
- PCA visualization
- Heatmap analysis
- Hierarchical clustering
- Biological interpretation
- PDF report generation
""")

# SIDEBAR
st.sidebar.title("Gene Expression Analyzer")

st.sidebar.info(
    """
    Upload a gene expression CSV file
    to perform clustering analysis
    and pattern discovery.
    """
)

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# MAIN PROGRAM
if uploaded_file:

    # LOAD DATA
    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # DATASET PREVIEW
    st.subheader("Dataset Preview")
    st.write(data.head())

    # HANDLE MISSING VALUES
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # GENE NAMES
    genes = data.iloc[:, 0]

    # EXPRESSION VALUES
    X = data.iloc[:, 1:]

    # KEEP ONLY NUMERIC COLUMNS
    X = X.select_dtypes(include=[np.number])
    # LIMIT DATA SIZE FOR FAST PROCESSING
    data = data.iloc[:200]

    genes = data.iloc[:, 0]

    X = data.iloc[:, 1:]

    # KEEP ONLY NUMERIC COLUMNS
    X = X.select_dtypes(include=[np.number])

    # NORMALIZATION
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    st.success("Normalization Completed")

    # PCA
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    # K-MEANS CLUSTERING
    kmeans = KMeans(
        n_clusters=2,
        random_state=42
    )

    clusters = kmeans.fit_predict(X_scaled)

    data['Cluster'] = clusters

    # CLUSTER RESULTS
    st.subheader("Cluster Results")

    st.write(data[['Cluster']].head(20))

    # PCA CLUSTER VISUALIZATION
    st.subheader("K-Means Cluster Visualization")

    fig1, ax1 = plt.subplots(figsize=(8,6))

    ax1.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters
    )

    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")

    st.pyplot(fig1, use_container_width=True)

    st.write("""
    INTERPRETATION:

    • Each point represents a gene

    • Genes with the same color belong
      to the same cluster

    • Genes located close together
      show similar expression patterns
    """)

    # GENE EXPRESSION HEATMAP
    st.subheader("Gene Expression Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        X.iloc[:30],
        cmap='coolwarm',
        ax=ax2
    )

    st.pyplot(fig2, use_container_width=True)

    st.write("""
    COLOR INTERPRETATION:

    • Red indicates higher gene expression

    • Blue indicates lower gene expression

    • Similar color patterns suggest
      co-expressed genes
    """)

    # GENE CORRELATION HEATMAP
    st.subheader("Gene Correlation Heatmap")

    TOP_N_CORR = 25

    top25 = X.var().nlargest(TOP_N_CORR).index

    corr_mat = X[top25].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(12,10))

    sns.heatmap(
        corr_mat,
        cmap='RdBu_r',
        center=0,
        ax=ax_corr
    )

    st.pyplot(fig_corr, use_container_width=True)

    st.write("""
    COLOR INTERPRETATION:

    • Red indicates positive correlation

    • Blue indicates negative correlation

    • Darker shades represent stronger
      relationships between genes
    """)

    # DENDROGRAM
    st.subheader("Hierarchical Clustering Dendrogram")

    sample_data = X_scaled[:50]

    linked = linkage(
        sample_data,
        method='ward'
    )

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    dendrogram(
        linked,
        no_labels=True,
        ax=ax3
    )

    st.pyplot(fig3, use_container_width=True)

    st.write("""
    DENDROGRAM INTERPRETATION:

    • Shorter branches indicate
      stronger similarity between genes

    • Longer branches indicate
      weaker similarity

    • Closely connected branches
      represent gene clusters
    """)

    # BIOLOGICAL INTERPRETATION
    st.subheader("Biological Interpretation")

    st.write("""
    Genes within the same cluster
    showed similar expression patterns.

    These co-expression patterns may
    indicate related biological pathways
    in Alzheimer's disease.
    """)

    # REPORT SECTION
    st.subheader("Generated Project Report")

    report = """
PROJECT TITLE:
Gene Expression Clustering and Pattern Discovery

OBJECTIVE:
To analyze Alzheimer's gene expression
data using clustering techniques and
identify co-expression patterns.

METHODOLOGY:
• Data preprocessing
• Data normalization
• PCA dimensionality reduction
• K-Means clustering
• Hierarchical clustering
• Heatmap visualization

RESULTS:
Genes were grouped into clusters
based on similar expression patterns.
Heatmaps and dendrograms revealed
co-expression relationships among genes.

CONCLUSION:
The project successfully identified
gene expression patterns associated
with Alzheimer's disease using
bioinformatics and machine learning
approaches.
"""

    st.text_area(
        "Project Report",
        report,
        height=400
    )

    # PDF GENERATION
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    text = pdf.beginText(40, 800)

    for line in report.split('\n'):
        text.textLine(line)

    pdf.drawText(text)

    pdf.save()

    buffer.seek(0)

    st.download_button(
        label="Download Report as PDF",
        data=buffer,
        file_name="Gene_Expression_Report.pdf",
        mime="application/pdf"
    )

    # FOOTER
    st.markdown("---")

    st.caption(
        "Mini Project — Gene Expression Clustering and Pattern Discovery"
    )
