'''
author: Elko Gerville-Reache
date Created: 2025-02-01
last Modified: 2025-02-27
purpose: generates an octree data structure from a set of particles
usage:
    - call recursive_oct_divide() to generate an octree to a specified depth
    - call traverse_octree() to traverse each node of the octree with respect to the other nodes and
    check how far the nodes are from each other
notes:
- requires the installation of numpy, and numba
- ensure input particle data are numpy arrays
'''

@njit(parallel = True)
def recursive_oct_divide(pos, vel, mass, origin, box_length, depth):
    '''
    generates an octree data structure for a set of particles up to a specified depth
    Parameters
    ----------
    pos: np.ndarray[np.float64]
        Nx3 array of particle positions where each row represents the [x, y, z] position coordinates of a particle in 3D space
        N is the total number of particles
    vel: np.ndarray[np.float64]
        Nx3 array of particle velocities where each row represents the [x, y, z] velocity components for a particle in 3D space
        N is the total number of particles
    mass: np.ndarray[np.float64]
        Nx1 array where each row represents the mass of a particle
        N is the number of particles
    origin: list[float]
        list containing the [x,y,z] coordinates of the center point of the simulation bounds
    box_length: int
        length of each side of the bounding cube that contains all particles, centered at the origin.
        sets the extent of the simulation volume
    depth: int
        specifies how many subdivisions when creating the octree, with higher numbers resulting in finer grids
        this controls how many times to loop over the entire tree when subdividing cells
    Returns
    -------
    coms: np.ndarray[np.float64]
        Mx3 array where each row represents the [x, y, z] coordinates of the center of mass for a cell in the octree
        M is the total number of cells in the octree.
    cell_masses: np.ndarray[np.float64]
        Mx1 array where each row represents the total mass of a cell in the octree
        M is the total number of cells in the octree
    centers_list: np.ndarray[np.float64]
        Mx3 array where each row contains the [x, y, z] coordinates of the center of a cell in the octree
        M is the total number of cells in the octree
    lengths_list: np.ndarray[np.float64]
        Mx1 array where each row contains the length of a cell cube in the octree
        M is the total number of cells in the octree
    Examples
    --------
    N = 1000
    box_length = 100
    pos = np.random.rand(N,3) * box_length
    vel = np.random.rand(N,3) * box_length
    mass = np.ones((N,1))/N
    origin = [0.0, 0.0, 0.0]
    depth = 7
    coms, cell_masses, centers_list, lengths_list = recursive_oct_divide(pos, vel, mass, origin, box_length, depth)
    Notes
    -----
    tolerance: int
        specifies the max number of particles in a cell, by default is set to 1
    '''
    # number of subdivisions
    depth = int(depth)
    # max number of particles per cell; cells are not subdivided if Nparticles < tolerance
    tolerance = 1
    # initial subdivision
    coms, cell_masses, centers_list, lengths_list = subdivide_cell(pos, vel, mass, origin, box_length)
    # number of tree depths
    for level in range(depth-1):
        # sum COM of each cell to create a unique tag
        com_sums = np.sum(coms, axis = 1)
        sums_list = np.unique(com_sums)
        # loop through each COM sum
        for i in prange(sums_list.shape[0]):
            # access all particles sharing a com (particles in the same cell)
            com_sum = sums_list[i]
            #idx = np.where(com_sums == com_sum)[0]
            mask = com_sums == com_sum
            # if number of particles > tolerance: subdivide into octants; else: skip
            if np.sum(mask) > tolerance:
                center = centers_list[mask][0]
                length = lengths_list[mask][0][0]
                octant_coms, octant_masses, octant_centers, octant_lengths = subdivide_cell(pos[mask], vel[mask],
                                                                                      mass[mask], center, length)
                # create new unique tags for subdivided cells
                octant_com_sums = np.sum(octant_coms, axis = 1)
                octant_sums_list = np.unique(octant_com_sums)
                # loop through new cells and update tree variables
                for j in prange(octant_sums_list.shape[0]):
                    octant_com_sum = octant_sums_list[j]
                    new_mask = octant_com_sums == octant_com_sum
                    new_idx = np.where(mask)[0][new_mask]
                    coms[new_idx] = octant_coms[new_mask]
                    cell_masses[new_idx] = octant_masses[new_mask]
                    centers_list[new_idx] = octant_centers[new_mask]
                    lengths_list[new_idx] = octant_lengths[new_mask]

            else:
                continue

    return coms, cell_masses, centers_list, lengths_list

@njit()
def subdivide_cell(pos, vel, mass, origin, length):
    '''subdivide simulation phase space into 8 octants'''
    # octdivide tree node
    masks = [(pos[:,0] > origin[0]) & (pos[:,1] > origin[1]) & (pos[:,2] > origin[2]),
             (pos[:,0] > origin[0]) & (pos[:,1] > origin[1]) & (pos[:,2] < origin[2]),
             (pos[:,0] < origin[0]) & (pos[:,1] > origin[1]) & (pos[:,2] > origin[2]),
             (pos[:,0] < origin[0]) & (pos[:,1] > origin[1]) & (pos[:,2] < origin[2]),
             (pos[:,0] < origin[0]) & (pos[:,1] < origin[1]) & (pos[:,2] > origin[2]),
             (pos[:,0] < origin[0]) & (pos[:,1] < origin[1]) & (pos[:,2] < origin[2]),
             (pos[:,0] > origin[0]) & (pos[:,1] < origin[1]) & (pos[:,2] > origin[2]),
             (pos[:,0] > origin[0]) & (pos[:,1] < origin[1]) & (pos[:,2] < origin[2])]
    # compute new centers
    centers = np.array([[origin[0] + length/4, origin[1] + length/4, origin[2] + length/4],
                        [origin[0] + length/4, origin[1] + length/4, origin[2] - length/4],
                        [origin[0] - length/4, origin[1] + length/4, origin[2] + length/4],
                        [origin[0] - length/4, origin[1] + length/4, origin[2] - length/4],
                        [origin[0] - length/4, origin[1] - length/4, origin[2] + length/4],
                        [origin[0] - length/4, origin[1] - length/4, origin[2] - length/4],
                        [origin[0] + length/4, origin[1] - length/4, origin[2] + length/4],
                        [origin[0] + length/4, origin[1] - length/4, origin[2] - length/4]])
    # generate arrays for storing data
    coms = np.zeros((pos.shape[0], 3))
    cell_masses = np.zeros((pos.shape[0], 1))
    centers_list = np.zeros((pos.shape[0], 3))
    lengths_list = np.zeros((pos.shape[0], 1))
    # iterate through each mask
    for i in prange(centers.shape[0]):
        mask = masks[i]
        center = centers[i]
        # skip empty masks
        if np.any(mask):
            # compute center of mass and total mass of cell
            com, cell_mass = compute_com(pos[mask], mass[mask])
            coms[mask] = com
            cell_masses[mask] = cell_mass
            centers_list[mask] = center
            lengths_list[mask] = length/2.0

    return coms, cell_masses, centers_list, lengths_list

@njit()
def traverse_octree(sum_com_dict, sums_list, r_sq, theta_sq, lengths_list):
    '''NxN interaction tracker variant, works well i believe '''
    sums_list_shape = sums_list.shape[0]
    lengths_list_sq = lengths_list**2
    com_approximation_list = List()
    particle_particle_list = List()
    acceptance_ratio = theta_sq*r_sq
    interaction_tracker = np.zeros((sums_list_shape, sums_list_shape), dtype = np.uint8)

    # loop through each cell via list of unique COMs
    for i in prange(sums_list_shape):
        current_cell = sums_list[i]
        idx = sum_com_dict[current_cell]
        com_approximation_idx = np.empty(0, dtype=np.int64)
        particle_particle_idx = np.empty(0, dtype=np.int64)

        # loop through all other COMs
        for j in prange(sums_list_shape):
            # skip identical cells
            if (i == j):
                continue
            # skip precomputed pairs:
            if interaction_tracker[i,j] != 0:
                idx_neighbor = sum_com_dict[sums_list[j]]
                # if i,j is far enough for com approximation, so is j,i
                if interaction_tracker[i,j] == 1:
                    com_approximation_idx = np.concatenate((com_approximation_idx, idx_neighbor))
                    continue
                # if i,j is close enough for particle-particle computation, so is j,i
                if interaction_tracker[i,j] == 2:
                    particle_particle_idx = np.concatenate((particle_particle_idx, idx_neighbor))
                    continue

            idx_neighbor = sum_com_dict[sums_list[j]]
            # check if (length <= theta*distance)**2
            if lengths_list_sq[j] <= acceptance_ratio[i,j]:
                com_approximation_idx = np.concatenate((com_approximation_idx, idx_neighbor))
                interaction_tracker[i,j] = 1
                interaction_tracker[j,i] = 1

            else:
                particle_particle_idx = np.concatenate((particle_particle_idx, idx_neighbor))
                interaction_tracker[i,j] = 2
                interaction_tracker[j,i] = 2

        com_approximation_list.append(com_approximation_idx)
        particle_particle_list.append(particle_particle_idx)

    return com_approximation_list, particle_particle_list

@njit(inline = 'always')
def compute_com(pos, mass):
    '''computes center of mass for a set of particles'''
    total_mass = np.sum(mass)
    com = np.sum(pos*mass, axis = 0)/total_mass

    return com, total_mass

@njit(inline = 'always')
def compute_distance_sq(coms_list):
    x = coms_list[:, 0:1]
    y = coms_list[:, 1:2]
    z = coms_list[:, 2:3]
    delx = x.T-x
    dely = y.T-y
    delz = z.T-z
    return (delx**2+dely**2+delz**2)

def construct_numba_dict(com_sums, sums_list):
    sum_com_dict = Dict.empty(
        key_type=types.float64,
        value_type=types.int64[:],  # Array of int64 indices
    )

    for com_sum in sums_list:
        sum_com_dict[com_sum] = np.where(com_sums == com_sum)[0]

    return sum_com_dict

def compute_octree_params(coms):
    coms_sum = np.sum(coms, axis = 1)
    coms_list = np.unique(coms, axis = 0)
    sums_list = np.sum(coms_list, axis = 1)
    sum_com_dict = construct_numba_dict(coms_sum, sums_list)
    r_sq = compute_distance_sq(coms_list)

    return sum_com_dict, sums_list, r_sq
