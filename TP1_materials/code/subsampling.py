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




# Functions
# (Here you can define useful functions to be used in the main)
#----------------------------
def cloud_decimation(xyz, rgb, labels, factor):

    # YOUR CODE
    decimated_xyz = xyz[0:np.size(xyz):factor]
    decimated_rgb = rgb[0:np.size(rgb):factor]
    decimated_labels = labels[0:np.size(labels):factor]

    return decimated_xyz, decimated_rgb, decimated_labels




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

    # Save
    write_ply('../decimated.ply', [decimated_xyz, decimated_rgb, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    print('Done')
