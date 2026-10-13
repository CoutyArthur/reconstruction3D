#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh



# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    
    tree = KDTree(points, leaf_size=80)
    N = grid_resolution
    idx = np.arange(N)

    # Indices (i, j, k) de chaque nœud, dans l'ordre C (k varie le plus vite),
    # ce qui correspond au reshape (N, N, N) de scalar_field
    i = np.repeat(idx, N * N)           # 0,0,0,...,1,1,1,...
    j = np.tile(np.repeat(idx, N), N)   # 0..0,1..1,...,N-1 répétés N fois
    k = np.tile(idx, N * N)             # 0,1,...,N-1 répétés N*N fois

    # Position de chaque nœud : x = min_grid + size_voxel * (i, j, k)
    nodes = min_grid + size_voxel * np.stack((i, j, k), axis=1)

    # Point le plus proche de chaque nœud (un seul appel pour tous les nœuds)
    _, nn = tree.query(nodes, k=1)
    nn = nn[:, 0]

    # f(x) = n_i . (x - p_i)
    f = np.sum(normals[nn] * (nodes - points[nn]), axis=1)

    scalar_field[:] = f.reshape(N, N, N)
    

# IMLS surface reconstruction
def compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn):
    h = 0.01                      # paramètre de l'énoncé (bruit du nuage)
    N = grid_resolution
    tree = KDTree(points, leaf_size=80)

    # 1. Positions de tous les nœuds de la grille (comme pour Hoppe)
    idx = np.arange(N)
    i = np.repeat(idx, N * N)
    j = np.tile(np.repeat(idx, N), N)
    k = np.tile(idx, N * N)
    nodes = min_grid + size_voxel * np.stack((i, j, k), axis=1)

    f = np.zeros(N ** 3, dtype=np.float32)
    chunk = 50000                 # nœuds traités à la fois (limite la mémoire)

    for s in range(0, len(nodes), chunk):
        x = nodes[s:s + chunk]                      # (m, 3)

        # 2. k plus proches voisins de chaque nœud
        _, nn = tree.query(x, k=knn)                # (m, knn), triés du plus proche au plus loin
        p = points[nn]                              # (m, knn, 3)
        n = normals[nn]                             # (m, knn, 3)

        # 3. Vecteurs x - p_i et distances au carré
        diff = x[:, None, :] - p                    # (m, knn, 3)
        dist2 = np.sum(diff * diff, axis=2)         # (m, knn)

        # 4. Poids gaussien theta(x - p_i) = exp(-||x - p_i||^2 / h^2)
        theta = np.exp(-dist2 / h ** 2)             # (m, knn)

        # 5. Distance signée au plan de chaque voisin : n_i . (x - p_i)
        proj = np.sum(n * diff, axis=2)             # (m, knn)

        # 6. Moyenne pondérée : sum(proj * theta) / sum(theta)
        num = np.sum(proj * theta, axis=1)          # (m,)
        den = np.sum(theta, axis=1)                 # (m,)

        # 7. Nœuds trop loin du nuage : den ~ 0 donc 0/0. On retombe sur Hoppe
        ok = den > 1e-12
        f[s:s + chunk] = np.where(ok, num / np.where(ok, den, 1.0), proj[:, 0])

    # 8. Remplissage de la grille 3D
    scalar_field[:] = f.reshape(N, N, N)



if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid
    #compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel, size_voxel, size_voxel))
    verts += min_grid
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj=f"../bunny_mesh_imls_128.ply", file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	


