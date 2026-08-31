#This is my implementation of Ant Colony Optimisation based on the following paper:

#G. A. Sgarro and L. Grilli, “Ant colony optimization for
#Chinese postman problem,” Neural Computing and Applications,
#vol. 36, no. 6, pp. 2901–2920, 11 2023. [Online]. Available:
#https://doi.org/10.1007/s00521-023-09195-4

import random
from typing import Tuple
import time
import matplotlib.pyplot as plt
from cppsolver.helper_functions import to_odd_subproblem, pairs_to_sol, pairs_to_score

def ACO(
        graph: list[list[int]],
        odd_info: list[list[list[int]], list[int], list[list[list[int]]]] = None,
        score_only: bool = False,
        edge_total: int = None,
        n_ants: int = 10,
        N: int = 1000,
        evap_rate: float = 0.25,
        reinforce_amount: float = 1,
        alpha: float = 1,
        beta: float = 4.5,
        time_limit: float = 120,
        stag_limit: int = 250,
        make_plot: bool = False
        ) -> Tuple[list[int], int]:
    """
    ACO implementation for CPP.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_info (list[list[list[int]], list[int], list[list[list[int]]]]): odd_subgraph, odd_indexes and subgraph_of_paths from to_odd_subproblem.
        score_only (bool): if True then only the score of the solution will be returned, faster.
        edge_total (int): can optionally be precomputed if using score_only.
        n_ants (int): number of ants.
        N (int): number of iterations.
        evap_rate (float): pheromone evapouration rate.
        reinforce_amount (float): pheromone reinforcement.
        alpha (float): pheromone matrix scalar.
        beta (float): heuristic matrix scalar.
        time_limit (float): time limit in seconds.
        stag_limit (int): terminates if this many iterations pass without improvement.
        make_plot: prints convergence graph.

    Returns:
        Tuple[list[int], int]:
        - list[int]: solution tour of the original graph.
        - int: cost of the tour.
    """
    start_time = time.time()
    end_time = start_time + time_limit

    if make_plot:
        best_tracking = []

    if not odd_info: #this allows to_odd_subproblem to be precomputed for large experiments
        odd_subgraph, odd_indexes, subgraph_of_paths = to_odd_subproblem(graph)
    else:
        odd_subgraph, odd_indexes, subgraph_of_paths = odd_info

    #for row in odd_subgraph:
    #    print(row)
    #print(odd_indexes)
    n_odds = len(odd_indexes)

    h_matrix = [[1/j if j != 0 else 0 for j in odd_subgraph[i]] for i in range(n_odds)] #heuristic matrix, represents desirability of an edge based on its distance

    
    #print("Odd subgraph:")
    #for row in odd_subgraph:
    #    print(row)

    #print("H matrix:")
    #for row in h_matrix:
    #    print(row)


    epsilon = 1e-4
    p_matrix = [[epsilon if i != j else 0 for j in range(n_odds)] for i in range(n_odds)] #pheromone matrix initialised with some epsilon

    #print("P matrix:")
    #for row in p_matrix:
    #    print(row)

    global_best_cost = float("inf")
    global_best_sol = []

    stagnation = 0
    for iteration in range(N):

        iteration_best_cost = float("inf")
        iteration_best_sol = []

        ant_pair_sets = []
        for ant in range(n_ants):
            unvisited = [i for i in range(n_odds)]

            pairs = []
            while len(unvisited) > 0:
                start = random.choice(unvisited)
                unvisited.remove(start)

                weights = [p_matrix[start][next_node]**alpha * h_matrix[start][next_node]**beta for next_node in unvisited] #weight the desirability of the potential next nodes using p_matrix and h_matrix
                total_w = sum(weights)
                if total_w == 0:
                    next_node = random.choice(unvisited) #random choice failsafe to avoid zero division
                else:
                    probs = [w/total_w for w in weights] #convert weights to probabilities
                    next_node = random.choices(unvisited, weights = probs, k = 1)[0] #select next node with relevant probilities
                unvisited.remove(next_node)
                pairs.append([start, next_node])
            ant_pair_sets.append(pairs)


        #pheromone updates
        for i in range(n_odds): #evapouration
            for j in range(n_odds):
                p_matrix[i][j] *= 1 - evap_rate

        for ant in range(n_ants): #reinforcement
            pairs = ant_pair_sets[ant]
            total_cost = 0
            for pair in pairs:
                pair_cost = odd_subgraph[pair[0]][pair[1]]
                total_cost += pair_cost

            if total_cost < iteration_best_cost: #track iteration best
                iteration_best_cost = total_cost
                iteration_best_sol = pairs
            
            for pair in pairs: #undirected edges: update both
                p_matrix[pair[0]][pair[1]] += reinforce_amount / total_cost
                p_matrix[pair[1]][pair[0]] += reinforce_amount / total_cost

        #if (iteration+1)%100 == 0:    
        #    print("Iteration", iteration+1, global_best_cost)

        if iteration_best_cost < global_best_cost:
            stagnation = 0
            global_best_cost = iteration_best_cost
            global_best_sol = iteration_best_sol

        else:
            stagnation += 1

        if make_plot:
            best_tracking.append(global_best_cost)

        if time.time() > end_time or stagnation > stag_limit:
            break

    #print("Global best sol:", global_best_sol)
    if score_only:
        solution = "n/a"
        total = pairs_to_score(graph, global_best_cost=global_best_cost, edge_total=edge_total)
    else:
        solution, total = pairs_to_sol(graph, odd_subgraph, odd_indexes, subgraph_of_paths, global_best_sol)
    
    if make_plot:
        plt.plot(best_tracking, label="Global Best")
        plt.xlabel("Iteration")
        plt.ylabel("Odd Pairing Cost")
        plt.title("ACO Convergence Plot")
        plt.legend()
        plt.grid()
        plt.show()

    return solution, total