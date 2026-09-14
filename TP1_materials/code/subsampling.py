# ------------------------------------------------------------------------------------------
#      |    TP1 Basic structures and operations on point clouds    |
#
#   Second script of the practical session. Subsampling of a point cloud
#   Hugues THOMAS - 13/12/2017
# ------------------------------------------------------------------------------------------




# Imports and global variables
#----------------------------

# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import math

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Functions
# (Here you can define useful functions to be used in the main)
#----------------------------
def cloud_decimation(xyz, rgb, labels, factor):

    # YOUR CODE
    decimated_xyz = xyz[0:np.size(xyz):factor]
    decimated_rgb = rgb[0:np.size(rgb):factor]
    decimated_labels = labels[0:np.size(labels):factor]

    return decimated_xyz, decimated_rgb, decimated_labels

def grid_subsampling(xyz, rgb, labels, voxelSize):

    decimated_xyz = []
    decimated_rgb = []
    decimated_labels = []

    minx = xyz[0][0]
    miny = xyz[0][1]
    minz = xyz[0][2]

    maxx = xyz[0][0]
    maxy = xyz[0][1]
    maxz = xyz[0][2]

    for point in xyz:
        if point[0] > maxx :
            maxx = point[0]
        if point[1] > maxy :
            maxy = point[1]
        if point[2] > maxz :
            maxz = point[2]

        if point[0] < minx :
            minx = point[0]
        if point[1] < miny :
            miny = point[1]
        if point[2] < minz :
            minz = point[2]

    width = maxx - minx
    height = maxy - miny
    depth = maxz - minz

    nbVoxel = 100
    
    widthVoxelSize = width/nbVoxel
    heightVoxelSize = height/nbVoxel
    depthVoxelSize = depth/nbVoxel

    tree = KDTree(xyz, leaf_size=80)

    for i in range(0, nbVoxel) :
        print("x =", i)
        tx = i/nbVoxel
        x = minx*(1-tx) + maxx*tx
        for j in range(0, nbVoxel) :
            ty = j/nbVoxel
            y = miny*(1-ty) + maxy*ty
            for k in range(0, nbVoxel) :
                tz = k/nbVoxel
                z = minz*(1-tz) + maxz*tz
                index = -1
                barycentre = np.array([x + widthVoxelSize/2, y + heightVoxelSize/2, z + depthVoxelSize/2])
                distanceMax = math.dist(barycentre, np.array([x,y,z]))
                distance, index = tree.query(np.array([barycentre]), k=1)
                distance = distance[0][0]
                index = index[0][0]
                if(distance < distanceMax):
                    decimated_xyz.append(xyz[index])
                    decimated_rgb.append(rgb[index])
                    decimated_labels.append(labels[index])
    return np.array(decimated_xyz), np.array(decimated_rgb), np.array(decimated_labels) 
                



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
    rgb = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    


    # Decimate the point cloud
    # ************************
    
    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_xyz, decimated_rgb, decimated_labels = cloud_decimation(xyz, rgb, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    t0 = time.time()
    grid_xyz, grid_rgb, grid_labels = grid_subsampling(xyz, rgb, labels, factor)
    t1 = time.time()
    print('grid decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_xyz, decimated_rgb, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../grid_subSampling.ply', [grid_xyz, grid_rgb, grid_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    print('Done')
