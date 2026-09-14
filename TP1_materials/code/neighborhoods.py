# ------------------------------------------------------------------------------------------
#      |    TP1 Basic structures and operations on point clouds    |
#
#   Third script of the practical session. Neighborhoods in a point cloud
#   Hugues THOMAS - 13/12/2017
# ------------------------------------------------------------------------------------------




# Imports and global variables
#----------------------------

# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import math

import matplotlib.pyplot as plt




# Functions
# (Here you can define useful functions to be used in the main)
#----------------------------
def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    neighborhoods = np.array([])
    i = 0
    for pointSearch in queries :
        i += 1
        temp = []
        for pointCloud in supports :
            if math.dist(pointSearch, pointCloud) < radius :
                temp.append(pointSearch)
        neighborhoods = np.append(neighborhoods, temp)

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = np.array([])
    i = 0
    for pointSearch in queries :
        i += 1
        dist = np.array([])
        j = 0
        temp= np.ones((k,2))
        for pointCloud in supports :          
            dist =  np.array([math.dist(pointSearch, pointCloud), j])
            j+=1
            if(dist[0] < temp[k-1][0]) :
                temp[k-1] = dist
                temp = temp[np.argsort(temp[:, 0])]
            
        neighborhoods = np.append(neighborhoods, temp)
        

    return neighborhoods


# Main
# (Here you can define the instructions that are called when you execute this file)
#----------------------------
if __name__ == '__main__':

    # Load point cloud
    # (Load the file '../data/indoor_scan.ply' - See read_ply function)
    # ****************

    file_path = '../data/indoor_scan.ply' # Path of the file
    data = read_ply(file_path) # Load point cloud

    # Concatenate data
    xyz = np.vstack((data['x'], data['y'], data['z'])).T


    # Brute force neighborhoods
    # ****************

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(xyz.shape[0], num_queries, replace=False)
        queries = xyz[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, xyz, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, xyz, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = xyz.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = xyz.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 
    # KDTree neighborhoods
    # ****************

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000
        neighbors_num = 100
        radius = 0.2

        # YOUR CODE
        random_indices = np.random.choice(xyz.shape[0], num_queries, replace=False)
        queries = xyz[random_indices, :]

        tree = KDTree(xyz, leaf_size=80)

        #KNN KDTree
        t0 = time.time()
        _, neighborhoodsKNN = tree.query(queries, k=neighbors_num)
        t1 = time.time()

        #spherical KDTree
        neighborhoodsSpherical = tree.query_radius(queries, r=radius)  

        t2 = time.time()
        minR = 0.01
        maxR = 1

        pas = minR

        timeList = np.zeros(101)
        radiusList = np.zeros(101)
        for i in range (0,101) :
            t = i/100
            rad = minR * (1 - t) + maxR * t 
            t3 = time.time()      
            _ = tree.query_radius(queries, r=rad)          
            t4 = time.time()
            timeList[i] = t4-t3
            radiusList[i] = rad * 100

        plt.figure(figsize=(8, 5))
        plt.plot(radiusList, timeList, marker='o', linestyle='-', color='tab:blue')
        plt.xlabel("Rayon en cm")
        plt.ylabel("Temps en secondes")
        plt.title("Temps en fonction du rayon")
        plt.grid(True)
        plt.show()



        # Print timing results
        print('{:d} KNN neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} Spherical computed in {:.3f} seconds'.format(num_queries, t2 - t1))
    
        # Time to compute all neighborhoods in the cloud
        total_spherical_time = xyz.shape[0] * (t2 - t1) / num_queries
        total_KNN_time = xyz.shape[0] * (t1 - t0) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))