#Simulated annealing implementation for CPP

import math
import random
from typing import Tuple
import time
from cppsolver.helper_functions import to_odd_subproblem, pairs_to_sol, pairs_to_score

def simulated_annealing(
        graph: list[list[int]],
        odd_info: list[list[list[int]], list[int], list[list[list[int]]]] = None,
        score_only: bool = False,
        edge_total: int = None,
        start_temp: int = 1000,
        iterations: int = 50000,
        cooling_rate: float = 0.995,
        time_limit: float = 120,
        stag_limit: int = 1000
        ) -> Tuple[list[int], int]:
    """
    Simulated Annealing implementation for CPP.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_info (list[list[list[int]], list[int], list[list[list[int]]]]): odd_subgraph, odd_indexes and subgraph_of_paths from to_odd_subproblem.
        score_only (bool): if True then only the score of the solution will be returned, faster.
        edge_total (int): can optionally be precomputed if using score_only.
        start_temp (int): starting temperature.
        iterations (int): number of iterations.
        cooling_rate (float): cooling rate.
        time_limit (float): time limit in seconds.
        stag_limit (int): terminates if this many iterations pass without improvement.

    Returns:
        Tuple[list[int], int]:
        - list[int]: solution tour of the original graph.
        - int: cost of the tour.
    """
    start_time = time.time()
    end_time = start_time + time_limit

    if not odd_info: #this allows to_odd_subproblem to be precomputed for large experiments
        odd_subgraph, odd_indexes, subgraph_of_paths = to_odd_subproblem(graph)
    else:
        odd_subgraph, odd_indexes, subgraph_of_paths = odd_info
    n_odds = len(odd_indexes)

    def fitness(x): #aim to choose pairing with minimal fitness
        total = 0
        for pair in x:
            total += odd_subgraph[pair[0]][pair[1]]
        return total
    
    def to_sol(x):
        sol = [[x[i], x[i+1]] for i in range(0, n_odds, 2)]
        return sol
    
    def swap(x):
        n = len(x)
        move = random.choice(["swap", "reverse", "insert"]) #three swap moves, represent different random pertubations to a pairing solution
        u, v = random.sample(range(n), 2)
        if move == "swap":
            x[u], x[v] = x[v], x[u]
        elif move == "reverse":
            l, r = min(u, v), max(u, v)
            x[l:r] = reversed(x[l:r])
        else:
            temp = x.pop(u)
            x.insert(v, temp)
        return x
    
    global_best_sol = list(range(n_odds))
    random.shuffle(global_best_sol)
    global_best_fitness = fitness(to_sol(global_best_sol))

    current_sol = global_best_sol.copy()
    current_fitness = global_best_fitness

    stagnation = 0 #optional stagnation early termination condition
    for i in range(iterations):

        T = start_temp * (cooling_rate**i)

        temp_sol = current_sol.copy()
        temp_sol = swap(temp_sol)
        temp_fitness = fitness(to_sol(temp_sol))

        delta = temp_fitness - current_fitness

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-20)): #accept improved solutions (and worse solutions sometimes, see: "annealing")
            current_sol = temp_sol
            current_fitness = temp_fitness
        
            if current_fitness < global_best_fitness:
                stagnation = 0
                global_best_sol = current_sol.copy()
                global_best_fitness = current_fitness
        
        if time.time() > end_time or stagnation > stag_limit:
            break

        stagnation += 1

    global_best_sol = to_sol(global_best_sol)



    if score_only: #this is an optimisation for large performance comparison experiments
        solution = "n/a"
        total = pairs_to_score(graph, global_best_cost=global_best_fitness, edge_total=edge_total)
    else:
        solution, total = pairs_to_sol(graph, odd_subgraph, odd_indexes, subgraph_of_paths, global_best_sol)
    
    return solution, total