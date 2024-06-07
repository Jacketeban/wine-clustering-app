# clustering.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(df, features, n_clusters=4, random_state=0):
    """
    Realiza el clustering de los datos.

    Parameters:
        df (DataFrame): DataFrame que contiene los datos.
        features (list): Lista de características relevantes para el clustering.
        n_clusters (int): Número de clusters para K-Means (default=4).
        random_state (int): Semilla aleatoria para reproducibilidad (default=0).

    Returns:
        labels (array): Etiquetas de los clusters asignadas por K-Means.
    """
    # Preprocesar los datos (escalar)
    x = df[features]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Aplicar el algoritmo de clustering (K-Means)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(x_scaled)

    # Devolver las etiquetas de los clusters
    labels = kmeans.labels_
    return labels
