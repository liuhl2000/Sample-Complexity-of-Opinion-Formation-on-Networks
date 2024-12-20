import numpy as np
import copy
import matplotlib.pyplot as plt

import cvxpy as cp
import networkx as nx
import random
from networkx.generators.random_graphs import random_regular_graph
from networkx.generators.random_graphs import erdos_renyi_graph

from utils import *

#Define an alpha_list to choose uniform influence fatcor \alpha
alpha_list = [0.01, 0.1, 1, 5, 10]

#Plot the degree-sample dependence for sythesize networks
#Type can be "PL" and "ER", denoting power law network and erdos_renyi random network respectively
def sample_plot_for_sythesized_network(type):
    for true_alpha in alpha_list:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("Average Number of Samples divided by k/ε")
        ax1.set_title("Average Sample Assignment and Degree when α = " + str(true_alpha))

        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Variance of Samples")
        ax2.set_title("Sample Variance and Degree when α = " + str(true_alpha))

        if type == "PL":
            n_list = [100, 200, 400, 600, 800, 1000]
        elif type == "ER":
            n_list = [600, 620, 650, 680, 700]

        for n in n_list:
            if type == "PL":
                g = nx.barabasi_albert_graph(n, 2)
            if type == "ER": 
                g  = erdos_renyi_graph(n, 0.3)
            degree_list = g.degree
            L, W, sq_W = get_matrix(g, n, true_alpha)
            samples, distri = optimal(sq_W, n, 1)
            d_s = []
            for i in range(n):
                true_node_label = list(g.nodes)[i]
                d_s.append((degree_list[true_node_label], distri[i]))

            value_dict = {}
            for item in d_s:
                degree = item[0]
                value_s = item[1]
                if degree in value_dict:
                    value_dict[degree].append(value_s)
                else:
                    value_dict[degree] = [value_s]

            value_dict = dict(sorted(value_dict.items(), key= lambda x: x[0]))
            #print(value_dict)

            mean_dict_p = {}
            var_dict_p = {}
            for value in value_dict:
                mean_p = np.mean(value_dict[value])
                var = np.var(value_dict[value])
                mean_dict_p[value] = mean_p
                var_dict_p[value] = var


            ax1.plot(list(mean_dict_p.keys()), list(mean_dict_p.values()), marker = "o", markersize = 2, label = "n = " + str(n))
            ax2.plot(list(var_dict_p.keys()), list(var_dict_p.values()), marker = "o", markersize = 2, label = "n = " + str(n))
        ax1.legend(fontsize="20", loc ="upper right")
        ax2.legend(fontsize="15", loc ="upper right")
        #plt.show()
        if type == "PL":
            fig1.savefig("Degree_TSC_PL/sample_" + str(true_alpha) + ".pdf")
            fig2.savefig("Degree_TSC_PL/variance_" + str(true_alpha) + ".pdf")
        if type == "ER":
            fig1.savefig("Degree_TSC_ER/sample_" + str(true_alpha) + ".pdf")
            fig2.savefig("Degree_TSC_ER/variance_" + str(true_alpha) + ".pdf")



#Get the data of the degree-sample dependence for real-world networks
#This function will be used together with plot_TSC_real
def sample_degree_for_real_network(g, name):
    n = len(g.nodes)
    for true_alpha in alpha_list:
        degree_list = g.degree
        L, W, sq_W = get_matrix(g, n, true_alpha)
        samples, distri = optimal(sq_W, n, 1)
        d_s = []
        for i in range(n):
            true_node_label = list(g.nodes)[i]
            d_s.append((degree_list[true_node_label], distri[i]))

        value_dict = {}
        for item in d_s:
            degree = item[0]
            value_s = item[1]
            if degree in value_dict:
                value_dict[degree].append(value_s)
            else:
                value_dict[degree] = [value_s]

        value_dict = dict(sorted(value_dict.items(), key= lambda x: x[0]))
        #print(value_dict)

        mean_dict_p = {}
        var_dict_p = {}
        for value in value_dict:
            mean_p = np.mean(value_dict[value])
            var = np.var(value_dict[value])
            mean_dict_p[value] = mean_p
            var_dict_p[value] = var


        name1 =  name + "_degree" + str(true_alpha) + ".txt"
        name2 =  name +  "_avg" + str(true_alpha) + ".txt"
        name3 =  name + "_var" + str(true_alpha) + ".txt"
        name4 =  name + "sample" + str(true_alpha) + ".txt"
        file1 = open(name + "/" + name1, "a")
        file2 = open(name + "/" + name2, "a")
        file3 = open(name + "/" + name3, "a")
        file4 = open(name + "/" + name4, "a")
        for i in range(len(mean_dict_p.keys())):
            file1.write(str(list(mean_dict_p.keys())[i]) + "\n")
            file2.write(str(list(mean_dict_p.values())[i]) + "\n")
            file3.write(str(list(var_dict_p.values())[i]) + "\n")
        file4.write(str(samples)+ "\n")
        file1.close()
        file2.close()
        file3.close()
        file4.close()

def preprocess(data):
    return [float(x.strip("\n")) for x in data]

def plot_TSC_real(alpha):
    email_degree = preprocess(open("email/email_degree" +  str(alpha) + ".txt").readlines())
    email_avg = preprocess(open("email/email_avg" +  str(alpha) + ".txt").readlines())
    email_var = preprocess(open("email/email_var" +  str(alpha) + ".txt").readlines())

    facebook_degree = preprocess(open("facebook/facebook_degree" +  str(alpha) + ".txt").readlines())
    facebook_avg = preprocess(open("facebook/facebook_avg" +  str(alpha) + ".txt").readlines())
    facebook_var = preprocess(open("facebook/facebook_var" +  str(alpha) + ".txt").readlines())

    bit_degree = preprocess(open("130bit/130bit_degree" +  str(alpha) + ".txt").readlines())
    bit_avg = preprocess(open("130bit/130bit_avg" +  str(alpha) + ".txt").readlines())
    bit_var = preprocess(open("130bit/130bit_var" +  str(alpha) + ".txt").readlines())

    econ_degree = preprocess(open("Econ/Econ_degree" +  str(alpha) + ".txt").readlines())
    econ_avg = preprocess(open("Econ/Econ_avg" +  str(alpha) + ".txt").readlines())
    econ_var = preprocess(open("Econ/Econ_var" +  str(alpha) + ".txt").readlines())

    
    #plot for avg
    fig1,ax1 = plt.subplots()
    #plot for var
    fig2,ax2 = plt.subplots()
    #plot for middle degree
    fig3,ax3 = plt.subplots()
    ax1.plot(email_degree, email_avg, marker = "o", markersize = 2, label = "email")
    ax1.plot(facebook_degree, facebook_avg, marker = "o", markersize = 2, label = "facebook")
    ax1.plot(bit_degree, bit_avg, marker = "o", markersize = 2, label = "130bit")
    ax1.plot(econ_degree, econ_avg, marker = "o", markersize = 2, label = "econ")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Average Number of Samples divided by k/ε")
    ax1.set_title("Average Sample Assignment and Degree when α = " + str(alpha))
    ax1.legend(fontsize="20", loc ="upper right")

    ax2.plot(email_degree, email_var, marker = "o", markersize = 2, label = "email")
    ax2.plot(facebook_degree, facebook_var, marker = "o", markersize = 2, label = "facebook")
    ax2.plot(bit_degree, bit_var, marker = "o", markersize = 2, label = "130bit")
    ax2.plot(econ_degree, econ_var, marker = "o", markersize = 2, label = "econ")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Variance of Samples")
    ax2.set_title("Sample Variance and Degree when α = " + str(alpha))
    ax2.legend(fontsize="20", loc ="upper right")

    ax3.plot(email_degree[8:], email_avg[8:], marker = "o", markersize = 2, label = "email")
    ax3.plot(facebook_degree[8:225], facebook_avg[8:225], marker = "o", markersize = 2, label = "facebook")
    ax3.plot(bit_degree, bit_avg, marker = "o", markersize = 2, label = "130bit")
    ax3.plot(econ_degree[5:], econ_avg[5:], marker = "o", markersize = 2, label = "econ")
    ax3.set_xlabel("Degree")
    ax3.set_ylabel("Average Number of Samples divided by k/ε")
    ax3.set_title("Average Sample Assignment and Degree(>8) when α = " + str(alpha))
    ax3.legend(fontsize="20", loc ="upper right")
    # fig1.show()
    # fig2.show()
    # fig3.show()
    fig1.savefig("Degree_TSC_REAL/avg_" + str(alpha) + ".pdf")
    fig2.savefig("Degree_TSC_REAL/var_" + str(alpha) + ".pdf")
    fig3.savefig("Degree_TSC_REAL/middle_" + str(alpha) + ".pdf")
    #plt.show()
    return


if __name__ == "__main__":
    #Plot the sample-degree dependance for sythesized networks
    sample_plot_for_sythesized_network("PL")
    sample_plot_for_sythesized_network("ER")
    
    #Read real-world networks and get their data
    ## This step is finished and the record error is given in the corresponding folder named after the real-world network

    # email = nx.read_edgelist("email-Eu-core-temporal.txt", create_using=nx.Graph(), nodetype=int, data=(("time", float),))
    # sample_degree_for_real_network(email, "email")

    # econ =  nx.read_edgelist("econ-mahindas.txt", create_using=nx.Graph(), nodetype=int, data=(("time", float),))
    # sample_degree_for_real_network(econ, "Econ")

    # facebook = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)
    # sample_degree_for_real_network(facebook, "facebook")

    # bit = nx.read_edgelist("130bit.txt", create_using=nx.Graph(), nodetype=int)
    # sample_degree_for_real_network(bit, "130bit")

    #Plot the degree-sample figure for real-world networks
    for alpha in alpha_list:
        plot_TSC_real(alpha)