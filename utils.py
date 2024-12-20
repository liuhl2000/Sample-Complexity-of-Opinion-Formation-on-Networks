import numpy as np
import random
import cvxpy as cp
import networkx as nx
import copy
from networkx.generators.random_graphs import random_regular_graph
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib.pyplot as plt


#Solve the primal problem by reformulating it to SOCP
def optimal(sq_W, n, m):
    x = cp.Variable(n*2, pos = True)
    b = 1/m *  np.ones(n)
    # c = 1/k * np.ones(n)
    obj = cp.Minimize(cp.sum(x[n:2*n]))
    soc_constraints = [cp.SOC(x[i+n] + x[i], cp.vstack((x[i+n] - x[i], 2))) for i in range(n)]
    con = [sq_W @ x[:n] <= b] + soc_constraints
    prob = cp.Problem(obj, con)
    prob.solve(solver = cp.ECOS, max_iters = 150, verbose = True, abstol = 1e-6, reltol = 1e-6)
    ##This is used for ego-Facebook with α = 10 only
    # prob.solve(solver = cp.CVXOPT, verbose = True)
    optimal_solution = prob.value
    local_samples = x.value[n:2*n]

    return optimal_solution, local_samples

#Solve the dual problem
#This is optimization (D) in overleaf
def dual(sq_W, n, m):
    x = cp.Variable(n, nonneg = True)
    function = 0
    for i in range(n):
        sum_1 = 0
        for j in range(n):
            sum_1 = sum_1 + x[j] * sq_W[i][j]
        sqrt_sum_1 = cp.sqrt(sum_1)
        function = function + sqrt_sum_1
    function = function*2 - 1/m * cp.sum(x)
    
    obj = cp.Maximize(function)
    con = []
    prob = cp.Problem(obj, con)
    prob.solve()

    optimal_solution = prob.value
    lambda_value = x.value

    return optimal_solution, lambda_value

#get the matrices we need given graph G with uniform influence factor \alpha
def get_matrix(G, n, alpha):
    L = np.array(nx.laplacian_matrix(G).todense())
    GM = alpha * L + np.identity(n)
    W = np.linalg.inv(GM)
    sq_W = W*W
    return L, W, sq_W
    
#Get the useful matrix we need for general weights(weights are given when generating graphs)
def get_matrix_general(G, n):
    L = np.array(nx.laplacian_matrix(G).todense())
    GM = L + np.identity(n)
    W = np.linalg.inv(GM)
    sq_W = W*W
    return L, W, sq_W
