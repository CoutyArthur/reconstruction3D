# ------------------------------------------------------------------------------------------
#      |    TP1 Basic structures and operations on point clouds    |
#
#   First script of the practical session. Transformation of a point cloud
#   Hugues THOMAS - 13/12/2017
# ------------------------------------------------------------------------------------------




# Imports and global variables
#----------------------------

# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply




# Functions
# (Here you can define useful functions to be used in the main)
#----------------------------



# Main
# (Here you can define the instructions that are called when you execute this file)
#----------------------------
if __name__ == '__main__':

    # Load point cloud
    # ****************

    file_path = '../data/bunny.ply' # Path of the file
    data = read_ply(file_path) # Load point cloud
    
    # Concatenate x, y, and z in a (N*3) point matrix
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    
    # Concatenate R, G, and B channels in a (N*3) color matrix
    rgb = np.vstack((data['red'], data['green'], data['blue'])).T


    # Transform point cloud
    # (Follow the instructions step by step)
    # *********************

    # Replace this line by your code
    meanX = 0
    meanY = 0
    meanZ = 0

    for points in xyz :
        meanX += points[0]
        meanY += points[1]
        meanZ += points[2]
    
    meanX /= np.size(xyz)
    meanY /= np.size(xyz)
    meanZ /= np.size(xyz)

    transformed_xyz = xyz
    print(np.size(transformed_xyz))

    for points in xyz :
        points[0] -= meanX
        points[1] -= meanY
        points[2] -= meanZ

        points[0] /= 2
        points[1] /= 2
        points[2] /= 2

        points[0] += meanX
        points[1] += meanY
        points[2] += meanZ

        points[1] -= 0.1




    # Save point cloud
    # (Save your result file - See write_ply function)
    # *********************
    transformed_xyz = xyz
    # Save point cloud
    write_ply('../little_bunny.ply', [transformed_xyz, rgb], ['x', 'y', 'z', 'red', 'green', 'blue'])
    print('Done')
