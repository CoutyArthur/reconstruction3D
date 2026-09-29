# ------------------------------------------------------------------------------------------
#      |    TP1 Iterative Closest Point    |
#
#   Second script of the practical session
#   Hugues THOMAS - 17/01/2018
# ------------------------------------------------------------------------------------------




# Imports and global variables
#----------------------------

# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys




# Functions
# (Here you can define useful functions to be used in the main)
#----------------------------
def best_rigid_transform(source, target):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        source = (d x N) matrix where "N" is the number of points and "d" the dimension
        target = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * source + T is aligned on target
    '''

    # YOUR CODE
    x = 0
    y = 0
    z = 0
    countSource = 0
    for point in source.T :
        x += point[0]
        y += point[1]
        z += point[2]
        countSource += 1
    source_barycenter = np.array([x/countSource, y/countSource, z/countSource])
    #source_barycenter = np.array([x/countSource, y/countSource])

    x = 0
    y = 0
    z = 0
    countTarget = 0
    for point in target.T :
        x += point[0]
        y += point[1]
        z += point[2]
        countTarget += 1
    target_barycenter = np.array([x/countTarget, y/countTarget, z/countTarget])
    #target_barycenter = np.array([x/countTarget, y/countTarget])

    source_centered = source - source_barycenter[:, np.newaxis]  # (3,N)
    target_centered = target - target_barycenter[:, np.newaxis]

    H = np.dot(source_centered, target_centered.T)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    T = target_barycenter - np.dot(R,source_barycenter)
    T = T[:, np.newaxis]

    return R, T


def icp_point_to_point(source, target, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        source = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        target = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        source_aligned = source aligned on target cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each source point in
        the target cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    source_aligned = np.copy(source)

    tree = KDTree(target.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    Rtot = np.eye(source.shape[0])
    Ttot = np.zeros((source.shape[0], 1))

    # YOUR CODE

    for i in range (0, max_iter) :

        dist, indices = tree.query(source_aligned.T, k=1)
        indices = indices.ravel() 

        neighbors_list.append(indices)

        target_matched = target[:, indices]

        R, T = best_rigid_transform(source_aligned, target_matched)

        Rtot = np.dot(Rtot, R)
        Ttot = np.dot(R,Ttot) + T

        R_list.append(Rtot)
        T_list.append(Ttot)

        source_aligned = R.dot(source_aligned) + T
            
        RMS = np.sqrt(np.mean(dist**2))
        RMS_list.append(RMS)

        if RMS < RMS_threshold:
            break

    return source_aligned, R_list, T_list, neighbors_list, RMS_list



def icp_point_to_point_fast(source, target, max_iter, RMS_threshold, sampling_limit):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        source = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        target = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        source_aligned = source aligned on target cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each source point in
        the target cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    source_aligned = np.copy(source)

    tree = KDTree(target.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    Rtot = np.eye(source.shape[0])
    Ttot = np.zeros((source.shape[0], 1))

    # YOUR CODE

    for i in range (0, max_iter) :

        print(i)
        random_indices = np.random.choice(source_aligned.shape[1], sampling_limit, replace=False)
        source_sampled = source_aligned[:, random_indices]
 
        dist, indices = tree.query(source_sampled.T, k=1)
        indices = indices.ravel() 

        neighbors_list.append(indices)

        target_matched = target[:, indices]

        R, T = best_rigid_transform(source_sampled, target_matched)

        Rtot = np.dot(R, Rtot)
        Ttot = np.dot(R,Ttot) + T

        R_list.append(Rtot)
        T_list.append(Ttot)

        source_aligned = R.dot(source_aligned) + T
            
        new_dist, _ = tree.query(source_aligned.T, k=1)
        RMS = np.sqrt(np.mean(new_dist**2))
        RMS_list.append(RMS)

        if RMS < RMS_threshold:
            break

    return source_aligned, R_list, T_list, neighbors_list, RMS_list





# Main
# (Here you can define the instructions that are called when you execute this file)
#----------------------------
if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))      

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-6)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()


    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        ND_o_path = '../data/Notre_Dame_Des_Champs_1.ply'
        ND_p_path = '../data/Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        ND_o_ply = read_ply(ND_o_path)
        ND_p_ply = read_ply(ND_p_path)
        ND_o = np.vstack((ND_o_ply['x'], ND_o_ply['y'], ND_o_ply['z']))
        ND_p = np.vstack((ND_p_ply['x'], ND_p_ply['y'], ND_p_ply['z']))

        # Apply ICP
        ND_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(ND_p, ND_o, 25, 1e-4, 10000)
        
        write_ply('../Notre_Dame_Des_Champs_r_opt.ply', [ND_p_opt.T], ['x', 'y', 'z'])
        
        # Show ICP
        #show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
