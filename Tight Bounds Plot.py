import numpy as np
import copy
import matplotlib.pyplot as plt

import cvxpy as cp
import networkx as nx
import random
from networkx.generators.random_graphs import random_regular_graph
from networkx.generators.random_graphs import erdos_renyi_graph

from utils import *

#Define global constant
#m is the sample one agent need to learn a good model
#m is set to 1 to solve εm_i*/k, which is a fixed value
m = 1
#Iter_num is defined as the rounds of generating random graphs and influence factors
##Can be set to 1 if you want get a quick result, actually there is little difference
Iter_num = 5
#Define an alpha_list to choose uniform influence fatcor \alpha
alpha_list = [0.01, 0.1, 1, 5, 10]


#get upper bound
def upper_bound(sq_W):
    B = np.linalg.inv(sq_W)
    return np.sum(1/np.sum(B, axis = 1))

#auxilary function for getting lower bound
def check_condition(sq_W):
    n = len(sq_W)
    B = np.linalg.inv(sq_W)
    row_sum = np.sum(B, axis = 1)

    condition_value = []
    for i in range(n):
        condition_value.append(np.sum(B[i]/(row_sum * row_sum)))
    return condition_value

#get lower bound
def lower_bound(sq_W):
    n = len(sq_W)
    condition_v = check_condition(sq_W)
    lambda_set = np.zeros(n) * m * m
    for k in range(n):
        if condition_v[k] > 0:
            lambda_set[k] = condition_v[k]
        else:
            lambda_set[k] = 0
    function = 0
    for i in range(n):
        sum_1 = 0
        for j in range(n):
            sum_1 = sum_1 + lambda_set[j] * sq_W[i][j]
        sqrt_sum_1 = np.sqrt(sum_1)
        function = function + sqrt_sum_1
    function = function*2 - 1/m * np.sum(lambda_set)
    return function


def to_even(n):
    if n % 2 == 0:
        return n
    else:
        return n + 1

#get plot of the optimal solution and bounds
def get_weighted_tightness_plot(name):
    #Write the final relative error to error file, current error file is given
    # err_file =  open("tight_weighted_" + name + "/" + "max_error.txt", "a")
    for true_alpha in alpha_list:
        #Creat plot
        fig, ax = plt.subplots()
        
        n_list = [100, 200, 400, 600, 800, 1000]
       
        # Generate random graphs and weights
        upp_err_list = []
        low_err_list = []

        for iter in range(Iter_num):
            for n in n_list:
                for i in range(5):
                    if name == "PL":
                        if i <= 2:
                            G = nx.barabasi_albert_graph(n, 2)
                        else:
                            G = nx.barabasi_albert_graph(n, 3)
                    elif name == "RR":
                        G = nx.random_regular_graph(to_even(int(n**(0.25 + 0.1 * i))), n)
                        now_power = max(0.5 - 0.2 * i, 0)
            
                    elif name == "ER":
                        G = erdos_renyi_graph(n, 0.25)

                    else:
                        print("INVALID NAME")
                        return

                    ###Add random weights for each edge
                    for (u,v,w) in G.edges(data=True):
                        w['weight'] = random.uniform(0, true_alpha)
                    
                    L,_,sq_W = get_matrix_general(G, n)
                    

                    TSC,_ = optimal(sq_W, n ,1)
                    upper_tight = upper_bound(sq_W)
                    lower_tight = max(lower_bound(sq_W), 1)


                    ## The abs is added because for small alpha, the solutions are very close (< 1e-7). There are computational error
                    upp_err_list.append(abs((upper_tight - TSC)/TSC))
                    low_err_list.append(abs((TSC - lower_tight)/TSC))


        min_ul = min(min(low_err_list), min(upp_err_list))
        max_ul = max(max(low_err_list), max(upp_err_list))

        bins = np.linspace(min_ul * 0.9, max_ul * 1.1, 50)

        # Record the relative error
        # err_file.write(str(true_alpha) + "\n")
        # for term in upp_err_list:
        #     err_file.write(str(term) + " ")
        # err_file.write("\n")
        # for term in low_err_list:
        #     err_file.write(str(term) + " ")
        # err_file.write("\n")
    

        ax.set_title("Bound Tightness when vij in [0," + str(true_alpha) + "]")
        ax.set_xlabel("Relative Error")
        ax.set_ylabel("Frequency")

        ax.hist(upp_err_list, bins, alpha=0.5, label="upper bound error")
        ax.hist(low_err_list, bins, alpha=0.5, label="lower bound error")

        ax.legend(fontsize="21", loc ="upper right")

        ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='x')
        fig.savefig("main_text" + name + "/IF = " + str(true_alpha) + ".pdf")
        # plt.show()
    # err_file.close()


## Plot the frequency distribution of relative error from the recorded error generated by function "get_weighted_tightness_plot"
def plot_from_error(name, alpha):
    fig, ax = plt.subplots()
    alpha_index = alpha_list.index(alpha)
    err_file =  open("tight_weighted_" + name + "/" + "max_error.txt").readlines()

    upp = err_file[alpha_index * 3 + 1].split(' ')[:-1]
    low = err_file[alpha_index * 3 + 2].split(' ')[:-1]
    upp_err_list = [float(x) for x in upp]
    low_err_list = [float(x) for x in low]

    min_ul = min(min(low_err_list), min(upp_err_list))
    max_ul = max(max(low_err_list), max(upp_err_list))

    bins = np.linspace(min_ul * 0.9, max_ul * 1.1, 50)

    ax.set_title("Bound Tightness when vij in [0," + str(alpha) + "]")
    ax.set_xlabel("Relative Error")
    ax.set_ylabel("Frequency")

    ax.hist(upp_err_list, bins, alpha=0.5, label="upper bound error")
    ax.hist(low_err_list, bins, alpha=0.5, label="lower bound error")

    ax.legend(fontsize="21", loc ="upper right")

    ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='x')
    fig.savefig("main_text/IF=" + str(alpha) + "_" + name + ".pdf")



## Verify the network gain of random regular networks
def gain_RR():
    fig,ax = plt.subplots()
    ax.set_xlabel("Square of degree")
    ax.set_ylabel("Network Gain")
    ax.set_title("Network Gain and Degree")

    true_alpha = 1
    n_list = [100, 200, 400, 600, 800, 1000]
    degree_list = [2,3,4,5,6,7,8,9,10]
    # degree_list = [11]
    M = 500
    k = 5
    square_deg = np.square(degree_list)
    for n in n_list:
        TSC_list = []
        for degree in degree_list:
            G = nx.random_regular_graph(degree, n)
            L,_,sq_W = get_matrix(G, n, true_alpha)
            # M(epsilon) = 50 , k = 5, epsilon = 0.1
            TSC, samples = optimal(sq_W, n, M) 
            TSC_list.append((n * M - n * k) / (TSC + n + 0.000001))

        ax.plot(square_deg, TSC_list, marker = "o", markersize = 2, label = "n = " + str(n))

    ax.legend(fontsize="16", loc ="upper left")
    fig.savefig("RR_Gain" + str(true_alpha) + ".pdf")
    plt.show()


    
## Plot the relative error for real-world networks
def real_network_bound_weighted(g, name):
    file = open("tight_weighted_RN/" + name + ".txt", "a")
    n = len(g.nodes)
    for alpha in alpha_list:
        err_u = 0
        err_l = 0
        for iter in range(50):
            for (u,v,w) in g.edges(data=True):
                w['weight'] = random.uniform(0,alpha)
                
    
            _, _, sq_W = get_matrix_general(g, len(g.nodes))
            TSC,_ = optimal(sq_W, n ,1)
            upper_tight = upper_bound(sq_W)
            lower_tight = max(lower_bound(sq_W), 1)

            curr_err_u = (upper_tight - TSC)/TSC
            if curr_err_u > err_u: err_u = curr_err_u
            curr_err_l = (TSC - lower_tight)/TSC
            if curr_err_l > err_l: err_l = curr_err_l

        file.write(str(alpha) + "\n")
        file.write(str(err_u) + "\n")
        file.write(str(err_l) + "\n")
    file.close()
    

if __name__ == "__main__":
    ##Get the plot for optimal solution and bounds for sythesized network
    # get_weighted_tightness_plot("PL")
    # get_weighted_tightness_plot("RR")
    # get_weighted_tightness_plot("ER")
    plot_from_error("PL", 1)
    plot_from_error("RR", 1)
    plot_from_error("ER", 1)

    ##Verify network gain for random regular graphs
    gain_RR()

    ##Get relative error for real-world networks
    email = nx.read_edgelist("email-Eu-core-temporal.txt", create_using=nx.Graph(), nodetype=int, data=(("time", float),))
    real_network_bound_weighted(email, "email")

    econ =  nx.read_edgelist("econ-mahindas.txt", create_using=nx.Graph(), nodetype=int, data=(("time", float),))
    real_network_bound_weighted(econ, "Econ")

    facebook = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)
    real_network_bound_weighted(facebook, "facebook")

    bit = nx.read_edgelist("130bit.txt", create_using=nx.Graph(), nodetype=int)
    real_network_bound_weighted(bit, "130bit")

