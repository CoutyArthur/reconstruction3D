#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import random

import os

from descriptors import compute_local_PCA



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    
    point_plane = points[0]
    

    vec1 = points[1]- points[0]
    vec2 = points[2] - points[0]

    normal_plane = np.cross(vec1, vec2)
    norm = np.linalg.norm(normal_plane)
    if norm < 1e-12:
        return None, None
    return point_plane, normal_plane / norm




def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    
    distances = np.abs((points - pt_plane) @ normal_plane)
    return distances <= threshold_in

def compute_normals(points, radius):
    # Normale = vecteur propre de la plus petite valeur propre (colonne 0, eigh trie par ordre croissant)
    _, all_eigenvectors = compute_local_PCA(points, points, radius)
    return all_eigenvectors[:, :, 0]

def in_plane_normals(points, normals, pt_plane, normal_plane, threshold_in=0.1, threshold_angle=25):
    # Condition 1 : distance au plan
    close = np.abs(np.dot(points, normal_plane) - np.dot(pt_plane, normal_plane)) <= threshold_in
    # Condition 2 : écart angulaire entre normale du point et normale du plan (signe ignoré)
    aligned = np.abs(np.dot(normals, normal_plane)) >= np.cos(np.radians(threshold_angle))
    return close & aligned



def RANSAC(points, nb_draws=100, threshold_in=0.1):
    
    best_vote = 0
    best_pt_plane = np.zeros((3))
    best_normal_plane = np.zeros((3))

    for i in range(nb_draws):
        idx = random.sample(range(len(points)), 3)          # 3 indices distincts
        point_plane, normal_plane = compute_plane(points[idx])
        if normal_plane is None:                            # tirage dégénéré
            continue

        vote = np.count_nonzero(in_plane(points, point_plane, normal_plane, threshold_in))
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = point_plane
            best_normal_plane = normal_plane

                
    return best_pt_plane, best_normal_plane, best_vote

def RANSAC_fast(points, nb_draws=100, threshold_in=0.1, sample_ratio=0.05, nb_candidates=5):
    n = len(points)
    m = min(n, max(2000, int(sample_ratio * n)))
    sub = points[np.random.choice(n, m, replace=False)]      # sous-échantillon pour le vote

    candidates = []                                          # (votes sur sub, point, normale)
    for _ in range(nb_draws):
        idx = random.sample(range(m), 3)
        pt, nor = compute_plane(sub[idx])
        if nor is None:
            continue
        vote = np.count_nonzero(in_plane(sub, pt, nor, threshold_in))
        candidates.append((vote, pt, nor))

    # Validation des meilleures hypothèses sur le nuage complet
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_vote, best_pt, best_nor = 0, np.zeros(3), np.zeros(3)
    for _, pt, nor in candidates[:nb_candidates]:
        vote = np.count_nonzero(in_plane(points, pt, nor, threshold_in))
        if vote > best_vote:
            best_vote, best_pt, best_nor = vote, pt, nor

    return best_pt, best_nor, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, ransac_fn=RANSAC):
    
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)

    # Cas de base : plus de plan à chercher ou plus assez de points
    if nb_planes == 0 or nb_points < 3:
        return plane_inds, remaining_inds, plane_labels

    # Meilleur plan du nuage courant
    pt_plane, normal_plane, vote = ransac_fn(points, nb_draws, threshold_in)
    if vote == 0:                           # aucun plan valide trouvé
        return plane_inds, remaining_inds, plane_labels

    # Points de ce plan (label 0) et points restants
    in_mask = in_plane(points, pt_plane, normal_plane, threshold_in)
    plane_inds = in_mask.nonzero()[0]
    plane_labels = np.zeros(len(plane_inds), dtype=int)
    remaining_inds = (~in_mask).nonzero()[0]

    # Appel récursif sur les points restants, avec un plan de moins à chercher
    sub_plane_inds, sub_remaining_inds, sub_plane_labels = recursive_RANSAC(
        points[remaining_inds], nb_draws, threshold_in, nb_planes - 1, ransac_fn)

    # Retour aux indices du nuage courant, et décalage des labels
    plane_inds = np.concatenate((plane_inds, remaining_inds[sub_plane_inds]))
    plane_labels = np.concatenate((plane_labels, sub_plane_labels + 1))
    remaining_inds = remaining_inds[sub_remaining_inds]

    return plane_inds, remaining_inds, plane_labels



def RANSAC_normals(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=25):

    best_vote = 0
    best_pt_plane = np.zeros(3)
    best_normal_plane = np.zeros(3)

    for i in range(nb_draws):
        idx = random.sample(range(len(points)), 3)
        point_plane, normal_plane = compute_plane(points[idx])
        if normal_plane is None:
            continue

        vote = np.count_nonzero(in_plane_normals(points, normals, point_plane, normal_plane,
                                                 threshold_in, threshold_angle))
        if vote > best_vote:
            best_vote = vote
            best_pt_plane = point_plane
            best_normal_plane = normal_plane

    return best_pt_plane, best_normal_plane, best_vote

def RANSAC_normals_fast(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=25,
                        sample_ratio=0.05, nb_candidates=5, min_sample=20000, max_tries_factor=50):
    n = len(points)
    m = min(n, max(min_sample, int(sample_ratio * n))) 
    sel = np.random.choice(n, m, replace=False)
    sub_pts, sub_nor = points[sel], normals[sel]            # mêmes indices pour points et normales
    cos_t = np.cos(np.radians(threshold_angle))

    candidates = []
    tries, max_tries = 0, nb_draws * max_tries_factor
    while len(candidates) < nb_draws and tries < max_tries:  # on ne compte que les tirages acceptés
        tries += 1
        idx = random.sample(range(m), 3)
    for _ in range(nb_draws):
        idx = random.sample(range(m), 3)
        pt, nor = compute_plane(sub_pts[idx])
        if nor is None:
            continue
        # Pré-filtre : les 3 points tirés doivent avoir une normale alignée avec celle du plan
        if np.any(np.abs(sub_nor[idx] @ nor) < cos_t):
            continue
        vote = np.count_nonzero(in_plane_normals(sub_pts, sub_nor, pt, nor, threshold_in, threshold_angle))
        candidates.append((vote, pt, nor))

    if not candidates:                                       # repli : version de référence
        return RANSAC_normals(points, normals, nb_draws, threshold_in, threshold_angle)
    # Validation des meilleures hypothèses sur le nuage complet
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_vote, best_pt, best_nor = 0, np.zeros(3), np.zeros(3)
    for _, pt, nor in candidates[:nb_candidates]:
        vote = np.count_nonzero(in_plane_normals(points, normals, pt, nor, threshold_in, threshold_angle))
        if vote > best_vote:
            best_vote, best_pt, best_nor = vote, pt, nor

    return best_pt, best_nor, best_vote


def recursive_RANSAC_normals(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=25, nb_planes=5, ransac_fn=RANSAC_normals):

    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)

    if nb_planes == 0 or nb_points < 3:
        return plane_inds, remaining_inds, plane_labels

    pt_plane, normal_plane, vote = ransac_fn(points, normals, nb_draws, threshold_in, threshold_angle)
    if vote == 0:
        return plane_inds, remaining_inds, plane_labels

    in_mask = in_plane_normals(points, normals, pt_plane, normal_plane, threshold_in, threshold_angle)
    plane_inds = in_mask.nonzero()[0]
    plane_labels = np.zeros(len(plane_inds), dtype=int)
    remaining_inds = (~in_mask).nonzero()[0]

    # Appel récursif sur les points (et leurs normales) restants
    sub_plane_inds, sub_remaining_inds, sub_plane_labels = recursive_RANSAC_normals(
        points[remaining_inds], normals[remaining_inds],
        nb_draws, threshold_in, threshold_angle, nb_planes - 1, ransac_fn)

    plane_inds = np.concatenate((plane_inds, remaining_inds[sub_plane_inds]))
    plane_labels = np.concatenate((plane_labels, sub_plane_labels + 1))
    remaining_inds = remaining_inds[sub_remaining_inds]

    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'


    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    nb_points = len(points)
    if 'label' in data.dtype.names :
        labels = data['label']
    else :
        labels = np.zeros(nb_points, dtype=np.int32)

    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    #threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    #threshold_in = 0.10
    nb_planes = 5
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    # print('\n--- 5) normals variant ---\n')

    radius_normals = 0.3        # valeur de départ, à ajuster à l'échelle du nuage
    nb_draws = 100
    threshold_in = 0.10
    threshold_angle = 25        # degrés
    nb_planes = 5

    # Les normales sont longues à calculer : on les met en cache
    normals_path = '../indoor_scan_normals.npy'
    if os.path.exists(normals_path):
        normals = np.load(normals_path)
    else:
        t0 = time.time()
        normals = compute_normals(points, radius_normals)
        np.save(normals_path, normals)
        print('normals computed in {:.3f} seconds'.format(time.time() - t0))

    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC_normals(
        points, normals, nb_draws, threshold_in, threshold_angle, nb_planes)
    print('recursive RANSAC with normals done in {:.3f} seconds'.format(time.time() - t0))

    for l in range(nb_planes):
        print('plane {}: {} points'.format(l, np.count_nonzero(plane_labels == l)))
    print('remaining: {} points'.format(len(remaining_inds)))

    write_ply('../best_planes_normals.ply',
              [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes_normals.ply',
              [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    print('\n--- Bonus : accélération (avec normales) ---\n')
    nb_draws = 100
    threshold_in = 0.10
    threshold_angle = 25
    nb_planes = 5

    fast_fn = lambda p, n, nd, th, ang: RANSAC_normals_fast(p, n, nd, th, ang, sample_ratio=0.05, nb_candidates=5)

    times = {}
    for name, tag, fn in [('sans accélération', 'ref', RANSAC_normals), ('avec accélération', 'fast', fast_fn)]:
        t0 = time.time()
        p_inds, r_inds, p_labels = recursive_RANSAC_normals(points, normals, nb_draws, threshold_in,
                                                            threshold_angle, nb_planes, fn)
        times[tag] = time.time() - t0
        print('{} : {:.3f} s'.format(name, times[tag]))
        for l in range(nb_planes):
            print('   plan {} : {} points'.format(l, np.count_nonzero(p_labels == l)))
        print('   restants : {} points'.format(len(r_inds)))
        write_ply('../best_planes_normals_{}.ply'.format(tag),
                [points[p_inds], colors[p_inds], labels[p_inds], p_labels.astype(np.int32)],
                ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])

    print('gain : x{:.1f}'.format(times['ref'] / times['fast']))
    print('Done')
    