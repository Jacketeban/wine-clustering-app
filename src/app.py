# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from clustering import perform_clustering

# Título de la aplicación
st.title('App de Clustering de Vinos')

# Sidebar para cargar el archivo CSV
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

# Verificar si se ha cargado un archivo
if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)

    # Definir características relevantes para el clustering
    features = ['Alcohol', 'Color_Intensity', 'Flavanoids']

    # Realizar el clustering
    cluster_labels = perform_clustering(df, features)

    # Calcular el Silhouette Score
    silhouette_avg = silhouette_score(df[features], cluster_labels)

    # Mostrar una pequeña vista previa del archivo cargado
    st.subheader("Vista previa del archivo cargado:")
    st.write(df.head())

    # Mostrar el Silhouette Score
    st.subheader("Silhouette Score:")
    st.write(f"**{silhouette_avg:.4f}**")

    # Mostrar visualización de resultados
    st.subheader("Visualización de resultados:")

    # Crear una figura para la gráfica de dispersión
    fig, ax = plt.subplots()

    # Graficar la dispersión de los datos con colores según los clusters
    sns.scatterplot(data=df, x='Alcohol', y='Color_Intensity', hue=cluster_labels, palette='viridis', ax=ax)

    # Configurar etiquetas y leyenda
    ax.set_title('Distribución de Alcohol por Intensidad de color')
    ax.set_xlabel('Alcohol')
    ax.set_ylabel('Intensidad de color')
    ax.legend(title='Cluster')

    # Mostrar la gráfica en la aplicación
    st.pyplot(fig)
