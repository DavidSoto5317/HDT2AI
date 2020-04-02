## Refencia: Codigo de Samuel Chavez
## Codigo analizado y modificado por David Soto 

import numpy as np
import pandas as pd
import pickle
from functools import reduce
from scipy import optimize as op

flatten_list_of_arrays = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

def inflate_matrixes(flat_thetas, shapes):
    layers = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(layers, dtype=int)

    for i in range(layers - 1):
        steps[i + 1] = steps[i] + sizes[i]

    return [
        flat_thetas[steps[i]: steps[i + 1]].reshape(*shapes[i])
        for i in range(layers - 1)
    ]

## Funcion para Paso 1 del algoritmo
def set_delta_size(thetas):
    arregloDeltas = []
    for i in range(len(thetas)):
        x, y = thetas[i].shape
        a = np.ones(x*y).reshape(x, y)
        arregloDeltas.append(a)
    return arregloDeltas

## Paso 2 del algoritmo
## Funcion para Paso 2.1 y 2.2
def feed_propagation(thetas, X):
    ## Paso 2.1
    listaDeMatricesA = [np.asarray(X)] # Agregamos la Matriz a un lista de matrices
    ## Paso 2.2
    for i in range(len(thetas)):
        ## Aqui agregamos la siguiente a^i+1 al arreglo de las "a" calculadas
        listaDeMatricesA.append(
            ## Se aplica la funcion sigmoide sobre z^i para obtener a^i+1
            sigmoid(
                ## Aqui se hace la multiplicacion de a^i * Theta.T para obtener z^i
                np.matmul(
                    np.hstack((
                        ## Aqui se le agrega el Bias a la matriz a^i
                        np.ones(len(X)).reshape(len(X), 1),
                        listaDeMatricesA[i]
                    )),
                    thetas[i].T
                )
            )
        )
    return listaDeMatricesA

## Funcion sigmoide para el paso 2.2
def sigmoid(z):
    a = []
    aValue = 0
    for i in np.nditer(z):
        aValue = 1/(1 + np.exp(-i))
        a.append(aValue)
    a = np.asarray(a).reshape(z.shape) 
    return a

def cost_function(flat_thetas, shapes, X, Y):
    a = feed_propagation(
        inflate_matrixes(flat_thetas, shapes),
        X
    )

    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X) 

## Unifica el mecanismo
def cost_bayesian_neural_network(flat_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes) + 1
    thetas = inflate_matrixes(flat_thetas, shapes)
    a = feed_propagation(thetas, X)
    deltas = [*range(layers - 1), a[-1] - Y]

    # Paso 2.4 (V2.0)
    # for i in range(layers - 2, 0, -1):
    #     delta =  (deltas[i + 1] @ thetas[i]) * np.hstack((
    #         ## Aqui se le agrega el Bias a la matriz a^i
    #         np.ones(len(X)).reshape(len(X), 1),
    #         (a[i] * (1 - a[i]))
    #     ))
    #     deltas[i] = np.delete(delta, 0, 1)
    # Paso 2.4 (V1.0)
    for i in range(layers - 2, 0, -1):
        deltas[i] =  (deltas[i + 1] @ np.delete((thetas[i]), 0, 1)) * (a[i] * (1 - a[i]))

    # Paso 2.5 (V2.0)
    # arregloDeltas = []
    # for n in range(layers - 1):
    #     f1, c1 = thetas[n].shape
    #     deltaMayuscula = np.zeros(f1*c1).reshape(f1,c1) 
    #     f2, _ = a[n].shape
    #     for k in range(f2): 
    #         f3 = len(deltas[n + 1][k])
    #         f4 = len(a[n][k])
    #         deltaMayuscula = deltaMayuscula + ((deltas[n + 1][k]).reshape(f3,1) * ((np.append([1],a[n][k])).reshape(f4 + 1,1)).T)

    #     arregloDeltas.append(deltaMayuscula) 

    # Paso 2.5 (V1.0)
    arregloDeltas = []
    for n in range(layers - 1):
        f1, c1 = thetas[n].shape
        deltaMayuscula = np.zeros(f1*c1).reshape(f1,c1) 
        f2, _ = a[n].shape
        for k in range(f2): 
            f3 = len(deltas[n + 1][k])
            f4 = len(a[n][k])
            deltaMayuscula = deltaMayuscula + ((deltas[n + 1][k]).reshape(f3,1) @ ((np.append([1],a[n][k])).reshape(f4 + 1,1)).T)

        arregloDeltas.append(deltaMayuscula) 

    arregloDeltas = np.asarray(arregloDeltas)

    # Paso 3
    return flatten_list_of_arrays([
        np.true_divide(arreglo, m)
        for arreglo in arregloDeltas
    ])


## Cargamos datos
## Iniciamos a probar con datos de training de fashion
# Se cargan los datoss  
datos = pd.read_csv('data/train/fashion-mnist_train.csv')

# Se procesa el dataset
X = datos.iloc[:, 1:] / 1000.0 #Normalizacion de los datos
m, n = X.shape
y = np.asarray(datos.iloc[:, 0])
y = y.reshape(m, 1)
Y = (y == np.array(range(10))).astype(int)

# # Se hace un set de la arquitectura de la red neuronal
NETWORK_ARCHITECTURE = np.array([
    n,
    130,
    10
])


theta_shapes = np.hstack((
    NETWORK_ARCHITECTURE[1:].reshape(len(NETWORK_ARCHITECTURE) - 1, 1),
    (NETWORK_ARCHITECTURE[:-1] + 1).reshape(len(NETWORK_ARCHITECTURE) - 1, 1)
))

flat_thetas = flatten_list_of_arrays([
    np.random.rand(*theta_shape) 
    for theta_shape in theta_shapes
    ])

# cost_bayesian_neural_network(flat_thetas, theta_shapes, X, Y)
print("Optimazing...")
result = op.minimize(
    fun=cost_function,
    x0=flat_thetas,
    args=(theta_shapes, X, Y),
    method='L-BFGS-B',
    jac=cost_bayesian_neural_network,
    options={'disp': True, 'maxiter': 3000}
)

print("Optimized")


# Se imprimen los resultados
print(result)
print(result.x)

# Se escribe el resultado en un archivo
outfile = open("model_trained", "wb")
pickle.dump(result.x, outfile)
outfile.close()

with (open("model_trained", "rb")) as openfile:
    while True:
        try:
            thetasOptimized = pickle.load(openfile)
        except EOFError:
            break

print(np.asarray(thetasOptimized))