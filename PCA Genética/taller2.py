import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Carga de Datos ---
# Cargamos el CSV
df = pd.read_csv('datos_genotipo.csv', index_col=0)
G = df.values # Matriz G
n, m = G.shape # n=30 individuos, m=100 SNPs

# --- Estimación de Frecuencias y Limpieza ---
# Calculamos la frecuencia p_j para cada SNP (columna)
p_j = np.sum(G, axis=0) / (2 * n)

# Filtramos singularidades
# Conservamos solo las columnas donde la frecuencia no sea 0 ni 1
valid_snps_mask = (p_j > 0) & (p_j < 1)

# Aplicamos la máscara para obtener la matriz y frecuencias limpias
G_clean = G[:, valid_snps_mask]
p_j_clean = p_j[valid_snps_mask]

# Mostramos los resultados en consola para validar
print(f"Dimensiones originales de G: {G.shape}")
print(f"Dimensiones de G despues de la limpieza: {G_clean.shape}")
print(f"SNPs (columnas) descartados por ser constantes: {m - G_clean.shape[1]}")

# --- Normalización de Patterson ---
n_clean, m_clean = G_clean.shape
print(f"\nNúmero de individuos: {n_clean}")
print(f"Número de SNPs válidos: {m_clean}")

# matriz inicializada en ceros y con valores flotantes
G_normalized = np.zeros((n_clean, m_clean), dtype=float)

print("\nIniciando normalización...")

for i in range(n_clean): # individuos
    for j in range(m_clean): # columna (SNP)

        # Se extrae el genotipo del individuo y su frecuencia alélica
        genotipo = G_clean[i, j]
        p = p_j_clean[j]

        # Aplicamos la normalización
        # (G_ij - 2p) / sqrt(2p(1-p))

        # 2*p*(1-p) es la varianza, osea es la dispersión que hay en los genotipos
        # por lo que el denominador termina siendo la desviación estándar
        # como tal esto ayuda a que cada snp contribuya equitativamente
        # G_ij - 2p no es mas que la diferencia entre el genotipo observado y el esperado

        G_normalized[i, j] = (genotipo - (2 * p)) / np.sqrt(2 * p * (1 - p))

print("Normalización completada")

# --- Factorización y Proyección ---

# Primero la SVD de la matriz normalizada
U, S, VT = np.linalg.svd(G_normalized, full_matrices=False)
print("\nSVD Calculada")
# La matriz U representa los individuos (vectores singulares izq)
# cada fila es un individuo
# La matriz S representa cuánta varianza explica cada componente
# La matriz VT guarda los SNP en sus columnas
print(f"Dimensiones de U: {U.shape}")
print(f"Dimensiones de S: {S.shape}")
print(f"Dimensiones de VT: {VT.shape}")

# Ahora obtenemos las coordenadas de los individuos en el espacio de las componentes principales
PC = U * S
PC1 = PC[:, 0]
PC2 = PC[:, 1]

# Crear la figura
plt.figure(figsize=(10, 8))

# Hacer el scatter plot (gráfico de puntos)
plt.scatter(PC1, PC2, s=100, alpha=0.6, color='blue', edgecolors='black', linewidth=1)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('Gráfico PCA', fontsize=14)
# Esto agrega lineas de referencia
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.savefig('pca_grafico.png', dpi=100, bbox_inches='tight')
print("\nGráfico guardado como 'pca_grafico.png'")
plt.show()
