# Imagina que las páginas están conectadas de la siguiente 
# manera:
# Página 1 enlaza a la 2 y a la 4.
# Página 2 enlaza a la 3 y a la 5.
# Página 3 enlaza a la 1 Aquí cerramos el ciclo principal 1 2 3 1!) y a la 6.
# Página 4 enlaza a la 5 y a la 7.
# Página 5 enlaza a la 6 y a la 8.
# Página 6 enlaza a la 7 y a la 8.
# Página 7 enlaza a la 4 (formando otro pequeño ciclo 4 7 4) y a la 8.
# Página 8 no enlaza a ninguna.

import numpy as np

def calcular_pagerank(M, d=0.85, tol=1e-8):
    """
    Calcula el vector PageRank utilizando el método de las potencias.
    
    Parámetros:
    M (numpy.ndarray): Matriz de transición original (nxn).
    d (float): Factor de amortiguamiento (por defecto 0.85).
    tol (float): Tolerancia para el criterio de convergencia (por defecto 1e-8).
    
    Retorna:
    v (numpy.ndarray): Vector de importancia (PageRank).
    iteraciones (int): Número de iteraciones hasta la convergencia.
    """
    n = M.shape[0]
    
    # 1. Construimos la matriz J (matriz de unos de tamaño nxn)
    J = np.ones((n, n))
    
    # 2. Construimos la Matriz de Google G
    # Fórmula: G = d*M + ((1-d)/n)*J
    G = d * M + ((1 - d) / n) * J
    
    # 3. Inicializamos el vector v (reparto equitativo inicial)
    v = np.ones(n) / n
    
    iteraciones = 0
    error = 1.0 # Valor inicial alto para entrar al bucle
    
    # 4. Método de las Potencias
    while error > tol:
        # Multiplicación matriz-vector: v^(k+1) = G * v^(k)
        v_siguiente = np.dot(G, v)
        
        # Calculamos el error como la norma (distancia) entre iteraciones
        error = np.linalg.norm(v_siguiente - v, 1)
        
        # Actualizamos v para la siguiente iteración
        v = v_siguiente
        iteraciones += 1
        
    return v, iteraciones

# --- CASO DE PRUEBA: Grafo de 8 nodos---

# Matriz de transicion
M = np.array([
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], # Nodo 1 recibe de 3
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Nodo 2 recibe de 1
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Nodo 3 recibe de 2
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], # Nodo 4 recibe de 1 y 7
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], # Nodo 5 recibe de 2 y 4
    [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], # Nodo 6 recibe de 3 y 5
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0], # Nodo 7 recibe de 4 y 6
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]  # Nodo 8 (sumidero) recibe de 5, 6 y 7
])

# Ejecución del algoritmo
pagerank, iters = calcular_pagerank(M)

# Resultados
print(f"El algoritmo convergió en {iters} iteraciones con una tolerancia de 10^-8.\n")
print("Vector de PageRank resultante (Orden de importancia):")
for i, pr in enumerate(pagerank):
    print(f"Página {i+1}: {pr:.6f}")