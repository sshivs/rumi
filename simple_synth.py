
# Setup
import numpy as np
import networkx as nx
import scipy.sparse
import estimators as ests
import copy

import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-p', '--pro', help='Prob of edge addition/deletion', type=float, default=1.0)
parser.add_argument('-r', '--rwt', help='Offdiag/Diag ratio', type=float, default=2.0)
args = vars(parser.parse_args())

n = 5000
T = 50
deg = 10

diag =1
r = args['rwt']
offdiag = r*diag

rand_wts = np.random.rand(n,3)
alpha = rand_wts[:,0].flatten()






G = nx.Graph()

# Add nodes
G.add_nodes_from(range(n))

# Add edges to ensure each node has 5 neighbors
for i in range(n):
  neighbors = [(i + j) % n for j in range(1, 10)]
  G.add_edges_from((neighbor, i) for neighbor in neighbors)

A = nx.adjacency_matrix(G)
A = copy.deepcopy(A)

G.clear()

G1 = nx.Graph()

# Add nodes
G1.add_nodes_from(range(n))


prob_to_adrem = args['pro']


# Add edges to ensure each node has 5 neighbors


final_index = 10 + int(prob_to_adrem*10)
#print("FI", final_index)
list_to_include = list(range(1,final_index))

for i in range(n):
  neighbors = [(i + j) % n for j in list_to_include]
  G1.add_edges_from((neighbor, i) for neighbor in neighbors)



def simpleWeights(A, diag=5, offdiag=5, rand_diag=np.array([]), rand_offdiag=np.array([])):
    n = A.shape[0]
    C_offdiag = offdiag*rand_offdiag
    in_deg = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten(),0)
    C = in_deg.dot(A - scipy.sparse.eye(n))
    col_sum = np.array(C.sum(axis=0)).flatten()
    col_sum += 1
    temp = scipy.sparse.diags(C_offdiag/col_sum)
    C = C.dot(temp)
    C_diag = diag*rand_diag
    C.setdiag(C_diag)
    return C


A.setdiag(1)
C = simpleWeights(A, diag, offdiag, rand_wts[:,1].flatten(), rand_wts[:,2].flatten())

#print (C)
A1 = nx.adjacency_matrix(G1)
A1.setdiag(1)

#print ("PER", np.percentile(A.dot(np.ones(n)), [1,10,25,50,75,90,99]))
#print ("PER", np.percentile(A1.dot(np.ones(n)), [1,10,25,50,75,90,99]))


beta = 1





if beta == 1:
  fy = lambda z: C.dot(z) + alpha
else:
  fy = lambda z: C.dot(z)*( ((C != 0).astype('float')).dot(z)) + alpha

TTE = 1./n * np.sum((fy(np.ones(n)) - fy(np.zeros(n))))


p= 0.2

outcom = C.diagonal()

mugraph = A1

results = [[],[], [],[],[],[],[],[],[],[]]

for i in range(T):
  z = (np.random.rand(n) < p)
  z = z + 0
  y = fy(z)

