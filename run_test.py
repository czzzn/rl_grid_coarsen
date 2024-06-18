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

    