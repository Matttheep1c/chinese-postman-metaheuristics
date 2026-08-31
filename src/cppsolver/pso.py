#This is my Particle Swarm Optimisation for CPP algorithm.
#The key discretisation concepts were inspired by the following PSO for TSP implementation:

# K.-P. Wang, L. Huang, C.-G. Zhou, and W. Pang, “Particle swarm
#optimization for traveling salesman problem,” in Proc. Int. Conf.
#Machine Learning and Cybernetics, vol. 3, 2003, pp. 1583–1585.

import random
from typing import Tuple
import time
from cppsolver.helper_functions import to_odd_subproblem, pairs_to_sol, pairs_to_score

def PSO(
        graph: list[list[int]],
        odd_info: list[list[list[int]], list[int], list[list[list[int]]]] = None,
        score_only: bool = False,
        edge_total: int = None,
        n_particles: int = 20,
        iterations: int = 1000,
        alpha: float = 0.27,
        beta: float = 0.82,
        p_mutate: float = 0.52,
        time_limit: float = 120,
        stag_limit: int = 250
        ) -> Tuple[list[int], int]:
    """
    PSO implementation for CPP with local search and mutation enhancements.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_info (list[list[list[int]], list[int], list[list[list[int]]]]): odd_subgraph, odd_indexes and subgraph_of_paths from to_odd_subproblem.
        score_only (bool): if True then only the score of the solution will be returned, faster.
        edge_total (int): can optionally be precomputed if using score_only.
        n_particles (int): number of particles.
        iterations (int): number of iterations.
        alpha (float): (0, 1), weight towards persuing current particle best.
        beta (float): (0, 1), weight towards persuing current swarm best.
        p_mutate (float): (0, 1), probability of a particle mutating during a given iteration.
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

    def rand_sol(n_odds):
        nodes = list(range(n_odds))
        random.shuffle(nodes)
        return nodes
    
    def swaps(u, v): #swaps to go from u to v
        pos = {val: i for i, val in enumerate(u)}
        swaps = []

        if v == []:
            return []
        
        for i in range(n_odds):
            if u[i] != v[i]:
                j = pos[v[i]]
                
                u[i], u[j] = u[j], u[i]

                swaps.append((i, j))

                pos[u[j]] = j
                pos[u[i]] = i

        return swaps
    
    def apply_swaps(x, swaps, frac): #applies a fraction of the found swaps (e.g.: a discrete version of "moving 10% of the way from u to v")
        total_swaps = len(swaps)
        n_swaps = max(0, round(total_swaps*frac))
        for i in range(n_swaps):
            current_swap = swaps[i]
            u = current_swap[0]
            v = current_swap[1]
            x[u], x[v] = x[v], x[u]
        return x
    
    def to_sol(x):
        sol = [[x[i], x[i+1]] for i in range(0, n_odds, 2)]
        return sol
    
    def fitness(x): #obtain fitness of the proposed pairing, aim to minimise this
        total = 0
        for pair in x:
            total += odd_subgraph[pair[0]][pair[1]]
        return total
    
    def local_search(x_new):
        x_test = x_new.copy()
        u, v = random.sample(range(len(x_new)), 2)
        x_test[u], x_test[v] = x_test[v], x_test[u]
        return x_test
    
    def mutate(x_new): #mutate function for increased stochastic exploration, found to be effective during testing
        n = len(x_new)
        i = random.randint(0, n-2)
        j = random.randint(i+1, n)
        sub = x_new[i:j]
        random.shuffle(sub)
        x_new[i:j] = sub
        return x_new

    particles = [rand_sol(n_odds) for i in range(n_particles)]
    p_bests = [[] for i in range(n_particles)]
    p_best_scores = [float("inf") for i in range(n_particles)]
    gbest = []
    gbest_score = float("inf")

    stagnation = 0
    for iteration in range(iterations):
        for i in range(n_particles):
            x_current = particles[i].copy()
            pbest = p_bests[i]
            r1 = random.random()
            r2 = random.random()
            pbest_swaps = swaps(x_current, pbest)
            x_new = apply_swaps(x_current, pbest_swaps, alpha*r1)

            
            gbest_swaps = swaps(x_new, gbest)
            
            x_new = apply_swaps(x_new, gbest_swaps, beta*r2)

            x_new = local_search(x_new)

            if random.random() < p_mutate:
                x_new = mutate(x_new)

            particles[i] = x_new

            sol = to_sol(x_new)
            sol_score = fitness(sol)

            if sol_score < p_best_scores[i]:
                p_bests[i] = x_new
                p_best_scores[i] = sol_score
        
        for i in range(n_particles):
            p_best_score = p_best_scores[i]
            if p_best_score < gbest_score:
                stagnation = 0
                gbest_score = p_best_score
                gbest = p_bests[i]
        
        if time.time() > end_time or stagnation > stag_limit:
            break
        
        stagnation += 1
    
    global_best_sol = to_sol(gbest)

    if score_only:
        solution = "n/a"
        total = pairs_to_score(graph, global_best_cost=gbest_score, edge_total=edge_total)
    else:
        solution, total = pairs_to_sol(graph, odd_subgraph, odd_indexes, subgraph_of_paths, global_best_sol)

    return solution, total