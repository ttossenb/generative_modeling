import numpy as np
from networkx.algorithms import bipartite
from scipy.sparse import coo_matrix
from annoy import AnnoyIndex
import time


def normalize(a, order=2, axis=-1):
    #print(a[a == np.zeros(a[0].shape)])
    #a[a == np.zeros(a[0].shape)] = np.ones(a[0].shape)
    a_norms = np.expand_dims(np.linalg.norm(a, order, axis), axis)
    return a / a_norms


def binMatrix(d):
    binaries = np.array([[0], [1]], dtype=int)
    for i in range(1, d):
        binaries = np.vstack((np.concatenate((np.zeros((2 ** i, 1), dtype=int), binaries), axis=-1),
                              np.concatenate((np.ones((2 ** i, 1), dtype=int), binaries), axis=-1)))
    return binaries

#number of traing inputs
n = 50000
#latent dimension
d = 10
#number of trees in the LSH-forest
n_trees = 60
#number of closest neighbourers to check
n_nbrs = 10

#number of points to put in each space segment
k = n // (2 ** d)

start1 = time.clock()

#generating target points, normalizing them, taking the absolute value of each
targetPoints = np.absolute(normalize(np.random.normal(0, 1, (2 ** d, k, d))))

binaries = np.expand_dims(binMatrix(d), axis=1)

#0 -> 1 (don't flip sign), 1 -> -1 (flip sign)
targetPoints = np.reshape(((-2) * binaries + 1) * targetPoints, ((2 ** d) * k, d))

#placeholder for input
latentPoints = normalize(np.random.normal(0, 1, (n, d)))

#create AnnoyIndex in R^d
targetIndex = AnnoyIndex(d)
#add each of the target points
for i in range(targetPoints.shape[0]):
    targetIndex.add_item(i, targetPoints[i])

#build the LSH-forest with the target points
targetIndex.build(n_trees)

#save and load with memory map
targetIndex.save("LSHForest.ann")
loadedIndex = AnnoyIndex(d)
loadedIndex.load("LSHForest.ann")

end1 = time.clock()
start2 = time.clock()

#find the closest neighbours for each latent point and their distances,
#and also create the biadjacency sparse matrix of the desired bipartite graph
#initialize with an empty sparse matrix
B = coo_matrix((latentPoints.shape[0], targetPoints.shape[0]), dtype=np.float16)
#initialize closeIndices and closeDistances
closeIndices = np.zeros((latentPoints.shape[0], n_nbrs), dtype=int)
closeDistances = np.zeros((latentPoints.shape[0], n_nbrs), dtype=np.float16)

for i in range(latentPoints.shape[0]):
    tempTuple = loadedIndex.get_nns_by_vector(latentPoints[i], n_nbrs, include_distances=True)
    closeIndices[i] = np.array(tempTuple[0])
    closeDistances[i] = np.array(tempTuple[1])

end2 = time.clock()
start3 = time.clock()

#create the biadjacency matrix
#arg1=data=(data, (rows, cols))
B = coo_matrix((closeDistances.flatten(), (np.repeat(np.arange(latentPoints.shape[0]), n_nbrs), closeIndices.flatten())))

#create the bipartite graph in networkx
G = bipartite.from_biadjacency_matrix(B)

end3 = time.clock()

print("Elapsed time during the initialization of the targets: ", end1-start1)
print("Elapsed time during the initialization of the LSH-forest: ", end2-start2)
print("Elapsed time during the creation of the bipartite graph: ", end3-start3)
