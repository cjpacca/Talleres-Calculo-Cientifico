import numpy as np

def calcular_pagerank(M, nombre_grafo, d=0.85, tol=1e-8):
    """
    Calcula el vector PageRank utilizando el método de las potencias.
    
    Parámetros:
    M (numpy.ndarray): Matriz de transición original (nxn).
    nombre_grafo (str): Etiqueta para imprimir resultados.
    d (float): Factor de amortiguamiento (por defecto 0.85).
    tol (float): Tolerancia para el criterio de convergencia (por defecto 1e-8).
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
        
        # Calculamos el error como la norma 1 (distancia) entre iteraciones
        error = np.linalg.norm(v_siguiente - v, 1)
        
        # Actualizamos v para la siguiente iteración
        v = v_siguiente
        iteraciones += 1
        
    # Imprimir los resultados
    print(f"--- Resultados para: {nombre_grafo} ---")
    print(f"Convergencia alcanzada en {iteraciones} iteraciones.")
    print("Vector de PageRank resultante:")
    for i, pr in enumerate(v):
        print(f"  Nodo {i}: {pr:.6f}")
    print("-" * 40 + "\n")

# DEFINICIÓN DE LOS CASOS DE PRUEBA

casos_de_prueba = {
    "Grafo 1 (3 Nodos, con sumidero)": np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ]),
    
    "Grafo 2 (4 Nodos, anillo)": np.array([
        [0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0]
    ]),
    
    "Grafo 3 (8 Nodos, sub-redes desconectadas)": np.array([
        [0.0, 0.5, 1/3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 1/3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1/3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1/3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1/3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1/3, 0.0]
    ]),
    
    "Grafo 4 (5 Nodos, ciclo filtrando a otro ciclo)": np.array([
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ]),
    
    "Grafo 5 (3 Nodos, fuente hacia ciclo)": np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]),
    
    # Página 1 enlaza a la 2 y a la 4.
    # Página 2 enlaza a la 3 y a la 5.
    # Página 3 enlaza a la 1, ciclo 1-2-3-1, y enlaza a la 6.
    # Página 4 enlaza a la 5 y a la 7.
    # Página 5 enlaza a la 6 y a la 8.
    # Página 6 enlaza a la 7 y a la 8.
    # Página 7 enlaza a la 4, pequeño ciclo 4-7-4, y enlaza a la 8.
    # Página 8 no enlaza a ninguna.
    "Grafo Propio (8 Nodos, ciclo y sumidero)": np.array([
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], 
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]  
    ])
}

# EJECUCIÓN

if __name__ == "__main__":
    for nombre, matriz in casos_de_prueba.items():
        calcular_pagerank(matriz, nombre)