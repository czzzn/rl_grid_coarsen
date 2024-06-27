from MG_Agent import Agent
#from utils import plot_learning_curves
import time
from  Unstructured import MyMesh, grid, rand_Amesh_gen, rand_grid_gen, structured
import fem 
import torch as T
from Scott_greedy import greedy_coarsening
import copy
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import pygmsh
import scipy.sparse as sp
from scipy.sparse.linalg import inv, eigs

PATH = "/content/rl_grid_coarsen/Model"


def my_test(K, dim, costum_grid, model_dir):

    """
    To built costum_grid:
    1. structured grid:
      n_row = 20
      n_col = 20
      theta = 0 # rotation param for generating structured grid
      epsilon = 100 # param for generating structured grid
      Theta = 0.90 # the param to determine diagonal dominance

      grid_ = structured(n_row, n_col, theta, epsilon, Theta)
    2. graded mesh
      mesh = gen_graded_mesh(20)
      grid_ = mesh_to_grid(mesh)
    """
    
    K= 4
    agent = Agent(dim = dim, K = K, gamma = 1, epsilon = 1, \
                  lr= 0.001, mem_size = 5000, batch_size = 32,\
                      eps_min = 0.01 , eps_dec = 1.25/5000, replace=10)

    agent.q_eval.load_state_dict(T.load(model_dir))

        

    agent.epsilon = 0
    
    Q_list = []
    Ahat_list = []
    A_list = []
    done = False
    
    
    if costum_grid!=None:
        grid_ = copy.deepcopy(costum_grid)
        grid_gr  = copy.deepcopy(grid_)
        
    else:
        
        grid_ = rand_grid_gen(None)
        grid_gr  = copy.deepcopy(grid_)
        
    while not done:
        
        observation = grid_.data
        #action = agent.choose_action(observation, grid_.viol_nodes()[0])
        with T.no_grad():
                
            Q, advantage = agent.q_eval.forward(observation)

            A_list.append(advantage)
            Q_list.append(Q)
            Ahat_list.append(advantage-advantage.max())
            viol_nodes = grid_.viol_nodes()[0]
            action = viol_nodes[T.argmax(Q[viol_nodes]).item()]
            
        print ("VIOLS",  len(grid_.viol_nodes()[0]))
        # print (agent.q_eval.forward(grid_.data))
        grid_.coarsen_node(action)
        done = True if grid_.viol_nodes()[2] == 0 else False
        
    print ("RL result", sum(grid_.active)/grid_.num_nodes)
    #grid_.plot()
    
    grid_gr = greedy_coarsening(grid_gr)
    
    return  grid_, grid_gr, Q_list, A_list, Ahat_list

# Step 3: Define a mesh size function for graded mesh
def mesh_size_function(x, y):
    center = np.array([0.5, 0.5])
    point = np.array([x, y])
    distance_to_center = np.linalg.norm(point - center)
    return 0.01 + 0.09 * distance_to_center  # Increased coefficient for more variation  

def gen_graded_mesh(num_nodes):
    # Step 1: Generate random points and compute the convex hull
    points = np.random.rand(num_nodes, 2)
    hull = ConvexHull(points)

    # Step 2: Perform Delaunay triangulation on the points
    tri = Delaunay(points)

    # Step 4: Use pygmsh to create the mesh based on the triangulation with graded mesh sizes
    with pygmsh.geo.Geometry() as geom:
        # Add points to pygmsh geometry with graded mesh sizes
        geom_points = [geom.add_point([p[0], p[1], 0], mesh_size=mesh_size_function(p[0], p[1])) for p in points]

        # Add lines and surfaces from Delaunay triangulation
        for simplex in tri.simplices:
            p1, p2, p3 = simplex
            line1 = geom.add_line(geom_points[p1], geom_points[p2])
            line2 = geom.add_line(geom_points[p2], geom_points[p3])
            line3 = geom.add_line(geom_points[p3], geom_points[p1])
            loop = geom.add_curve_loop([line1, line2, line3])
            surface = geom.add_plane_surface(loop)
    
        # Generate the mesh
        mesh = geom.generate_mesh()
        
    return mesh

def mesh_to_grid(mesh):
    # generate grid object from a customized mesh
    # input mesh: mesh object from geom.generate_mesh

    msh = MyMesh(mesh)
    A, b = fem.gradgradform(msh, kappa=None, f=None, degree=1)
    
    fine_nodes = [i for i in range(A.shape[0])]
    
    #set_of_edge = set_edge_from_msh(mymsh)
    mygrid = grid(A,fine_nodes,[],msh,0.56)

    return mygrid

def grid_quality(A,grid):
    """
    Calculate the grid quality measure for an algebraic multigrid method.

    This function computes the measure as defined in "On Generalizing the Algebraic Multigrid Framework"
    by Robert D. Falgout and Panayot S. Vassilevski. 

    Parameters:
    ----------
    A : csr_matrix
        The coefficient matrix of the linear system, expected to be sparse.
    grid : Grid class instance from Unstructured.py
        The grid object containing the mesh details like fine and coarse nodes.

    Returns:
    ------- 
    mu : float
        The computed grid quality. 

    Raises:
    ------
    LinAlgError
        If the matrix inversion fails or the eigenvalue solver does not converge.

    Examples:
    --------
    >>> grid = Grid(...)  # assuming an appropriate Grid class and constructor
    >>> A = grid.A # A is a sparse matrix in csr format
    >>> cr_value = cr_measure(A, grid)
    >>> print(f"CR Measure: {cr_value}")

    Notes:
    -----
    The function uses sparse matrix operations to ensure efficiency on large matrices. Ensure that the input matrix `A`
    is properly formatted as a csr_matrix for optimal performance.
    """
    nf = len(grid.fine_nodes)
    nc = grid.num_nodes - nf

    # Smoother S
    S = sp.vstack([sp.eye(nf, format='csr'), sp.csr_matrix((nc, nf))])

    # Matrix M is the diagonal part of A
    M = sp.diags(A.diagonal())

    # Compute X = M(M+M^T-A)^{-1}M^T
    inner_matrix = M + M.T - A
    M_inv = inv(inner_matrix)
    X = M.dot(M_inv).dot(M.T)

    # Compute S^T A S and S^T X S
    ST_A_S = S.T.dot(A.dot(S))
    ST_X_S = S.T.dot(X.dot(S))

    # Compute the generalized eigenvalues
    vals, _ = eigs(ST_X_S.dot(ST_A_S), k=1, which='SM', tol=1e-6, maxiter=10000)
    lambda_min = vals.real[0]  # Smallest eigenvalue

    # Compute the CR measure
    mu = 1 / lambda_min if lambda_min != 0 else np.inf

    return mu






    