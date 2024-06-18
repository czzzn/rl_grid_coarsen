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

PATH = "/content/rl_grid_coarsen/Model"


def run_test(K, dim, costum_grid, model_dir):
    
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
    points = np.random.rand(10, 2)
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

def mesh_to_grid(mesh):
    # generate grid object from a customized mesh
    # input mesh: mesh object from geom.generate_mesh

    msh = MyMesh(mesh)
    A, b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1)
    
    fine_nodes = [i for i in range(A.shape[0])]
    
    #set_of_edge = set_edge_from_msh(mymsh)
    grid = grid(A,fine_nodes,[],mymsh,0.56)

    return grid






    