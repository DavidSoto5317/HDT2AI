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

# Funcion Sigmoide
def sigmoid(z):
    a = [(1 / (1 + np.exp(-i))) for i in z]
    return np.asarray(a).reshape(z.shape)

# Funcion de costo entre la prediccion y el valor esperado
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

    # Paso 2.4
    for i in range(layers - 2, 0, -1):
        deltas[i] =  (deltas[i + 1] @ np.delete(thetas[i], 0, 1)) * (a[i] * (1 - a[i]))

    # Paso 2.5
    arregloDeltas = []
    for n in range(layers - 1):
        arregloDeltas.append(
            (deltas[n + 1].T 
            @ 
            np.hstack((
                np.ones(len(a[n])).reshape(len(a[n]), 1),
                a[n]
            ))) / m
        )

    arregloDeltas = np.asarray(arregloDeltas)

    # Paso 3
    return flatten_list_of_arrays(
        arregloDeltas
    )

## Realiza la prediccion del algoritmo
def prediction(flat_thetas, shapes, X, Y):
    thetas = inflate_matrixes(flat_thetas, shapes)
    a = feed_propagation(thetas, X)

    return [a[-1] , Y]