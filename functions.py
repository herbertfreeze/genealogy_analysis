import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Transforms Simplex Plane to XY Plane
def SimplexTo2D(A):
    """
    Takes in a Nx3 list of points that satisfy x+y+z = 1
    Then transforms it to the 2D Simplex
    """

    # Translate plane to pass through origin
    A -= np.array([0,0,1])

    # Rotate plane to xy plane
    cos_thet = 1/np.sqrt(3)
    sin_thet = np.sqrt(2/3) 
    u1       = 1/np.sqrt(2) # the fix (a^2 + b^2)
    u2       = -1/np.sqrt(2)
    rot = np.array([[cos_thet+(1-cos_thet)*u1**2, u1*u2*(1-cos_thet),  u2*sin_thet],
                    [u1*u2*(1-cos_thet), cos_thet+u2**2*(1-cos_thet), -u1*sin_thet],
                    [-u2*sin_thet,u1*sin_thet, cos_thet]]) 
    A = np.matmul( rot, A.T).T

    # Subset on XY plane, then rotate to align x-axis, normalized so simplex side is 1
    A = A[:,:2]
    rot2 = np.array([[ 1.3660254,  0.3660254  ],
                     [-0.3660254,  1.3660254  ]]) / ( np.sqrt(2)**2) # one for hypotneuse one to normalize
    A = np.matmul(rot2,A.T).T
    return A


class condition(object):
    def __init__(self,name,count_M, count_G, ratio, power_ratio):
        self.name = name
        self.cM = count_M
        self.cG = count_G
        self.ratio = ratio
        self.PR = power_ratio

def extract_possible_keys(ID2smthing,G):
    G_distrib = [ID2smthing[n] for n in G.nodes()]
    G_count = Counter(G_distrib)
    return list(G_count.keys())

def conditional_prob_K(condition, dictionary,G):
    count_G = 0
    count_M = 0
    for n in G.nodes():
        if dictionary[n] == condition:
            count_G += 1
            
    for n in K.nodes():
        if dictionary[n] == condition:
            count_M += 1
    return count_M, count_G, count_M/count_G

def show_stats_K(dictionary,G):
    keys = extract_possible_keys(dictionary,G)
    stat_objects = []
    for k in keys:
        count_M, count_G, ratio = conditional_prob_K(k, dictionary,G)
        power_ratio = ratio / P_M
        print(k, count_M, count_G, ratio, sep = ", ")
        print("Power Ratio", power_ratio)
        
        COND = condition(k, count_M, count_G, ratio, power_ratio)
        stat_objects.append(COND)
        
    return stat_objects

def extract_dict(G, dictionary):
    countries = defaultdict(int)
    counter = 0
    for n in G.nodes():
        try:
            C = dictionary(n)
            countries[C] += 1
        except KeyError:
            counter += 1
            pass
    sorted_D = sorted(countries.items(), key=lambda x: x[1],reverse = True)
    return sorted_D

def simplify_hist(sorted_d,thresh, group):
    hist_dict = {}
    for i in sorted_d[:thresh]:
        hist_dict[i[0]] = i[1]
    other_count = 0
    if group:
        for i in sorted_d[thresh:]:
            other_count += i[1]
        hist_dict["Other"] = other_count
    return hist_dict

def extract_hist(G, dictionary, thresh = 20, group = True):
    sorted_d = extract_dict(G, dictionary)
    hist_dict = simplify_hist(sorted_d,thresh, group)
    return hist_dict

def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})
