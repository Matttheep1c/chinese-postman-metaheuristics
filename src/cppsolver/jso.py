#My Jellyfish Search Optimiser implementation
#This code is a discretised version of the continuous algorithm,
#based on the pseudocode from the following paper:

# J.-S. Chou and A. Molla, “Recent advances in use of bio
#inspired jellyfish search algorithm for solving optimization
#problems,” Scientific Reports, vol. 12, no. 1, p. 19157, 11 2022.
#[Online]. Available: https://www.nature.com/articles/s41598-022-23121-z

import random
from typing import Tuple
import time
from cppsolver.helper_functions import to_odd_subproblem, pairs_to_sol, pairs_to_score

def JSO(
        graph: list[list[int]],
        odd_info: list[list[list[int]], list[int], list[list[list[int]]]] = None,
        score_only: bool = False,
        edge_total: int = None,
        nJellyfish: int = 10,
        MaxIt: int = 100,
        beta: float = 3,
        gamma: float = 0.1,
        time_limit: float = 120,
        stag_limit: int = 250
        ) -> Tuple[list[int], int]:
    """
    Jellyfish Search Optimiser implementation.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_info (list[list[list[int]], list[int], list[list[list[int]]]]): odd_subgraph, odd_indexes and subgraph_of_paths from to_odd_subproblem.
        score_only (bool): if True then only the score of the solution will be returned, faster.
        edge_total (int): can optionally be precomputed if using score_only.
        nJellyfish (int): number of jellyfish.
        MaxIt (int): maximum number of iterations.
        beta (float): distribution coefficient.
        gamma (float): motion coefficient.
        time_limit (float): time limit in seconds.
        stag_limit (int): terminates if this many iterations pass without improvement.

    Returns:
        Tuple[list[int], int]:
        - list[int]: solution tour of the original graph.
        - int: cost of the tour.
    """
    start_time = time.time()
    end_time = start_time + time_limit

    c_0 = 0.5

    if not odd_info: #this allows to_odd_subproblem to be precomputed for large experiments
        odd_subgraph, odd_indexes, subgraph_of_paths = to_odd_subproblem(graph)
    else:
        odd_subgraph, odd_indexes, subgraph_of_paths = odd_info
    n_odds = len(odd_indexes)

    def list_to_pairs(lst):
        #len(lst) must be even
        return [lst[i:i+2] for i in range(0, len(lst), 2)]

    def fitness(sol):
        pairs = list_to_pairs(sol)
        total = 0
        for pair in pairs:
            total += odd_subgraph[pair[0]][pair[1]]
        return total
    
    def swaps(u, v): #swaps to go from u to v
        pos = {val: i for i, val in enumerate(u)}
        u = u[:]
        swaps = []
        
        for i in range(len(u)):
            if u[i] != v[i]:
                j = pos[v[i]]
                
                u[i], u[j] = u[j], u[i]

                swaps.append((i, j))

                pos[u[j]] = j
                pos[u[i]] = i

        return swaps
    
    def apply_swaps(x, swaps, frac): #apply a fraction "frac" of the swaps found in "swaps" to "x"
        x = x[:]
        total_swaps = len(swaps)
        n_swaps = max(0, round(total_swaps*frac))
        for i in range(n_swaps):
            current_swap = swaps[i]
            u = current_swap[0]
            v = current_swap[1]
            x[u], x[v] = x[v], x[u]
        return x

    
    jellyfish = []
    jellyfish_fitnesses = []

    for i in range(nJellyfish): #initialise jellyfish population randomly
        temp = list(range(n_odds))
        random.shuffle(temp)
        temp_fitness = fitness(temp)
        jellyfish.append(temp)
        jellyfish_fitnesses.append(temp_fitness)

    best_jellyfish_cost = min(jellyfish_fitnesses)
    best_jellyfish = jellyfish[jellyfish_fitnesses.index(best_jellyfish_cost)]
    
    stagnation = 0
    for iteration in range(MaxIt):
        iteration_best_cost = float("inf")
        iteration_best_sol = None
        for j in range(nJellyfish):
            c_t = abs((1-(iteration/MaxIt))*(2*random.random()-1))

            if c_t >= c_0: #jellyfish follow ocean current
                
                av = [0 for _ in range(n_odds)] #discrete approximation of average jellyfish
                for pos in range(n_odds): #take most common value in each position
                    counts = [0 for _ in range(n_odds)]
                    for jf in jellyfish:
                        counts[jf[pos]] += 1
                    av[pos] = max(range(n_odds), key=lambda x: counts[x])
                
                missing = set(range(n_odds))
                av_jellyfish = []
                used = set()

                for pos in range(n_odds): #repair missing/duplicates
                    v = av[pos]
                    if v not in used:
                        used.add(v)
                        missing.discard(v)
                        av_jellyfish.append(v)
                    else:
                        replacement = missing.pop()
                        av_jellyfish.append(replacement)
                        used.add(replacement)
                


                alpha = min(1, beta*random.random())
                trend = apply_swaps(av_jellyfish, swaps(av_jellyfish, best_jellyfish), alpha)

                jellyfish[j] = apply_swaps(jellyfish[j], swaps(jellyfish[j], trend), random.random())

            else: #jellyfish moves inside swarm
                if random.random() > (1-c_t): #jellyfish exhibits passive motion, i.e. random movement

                    bound_range = n_odds - 1 #Ub-Lb is represented as the maximum number of swaps to turn one solution to another in this discrete space

                    k = int(gamma*random.random()*bound_range)

                    for _ in range(k): #apply k random swaps
                        a, b = random.sample(range(n_odds), 2)
                        jellyfish[j][a], jellyfish[j][b] = jellyfish[j][b], jellyfish[j][a]
                
                else: #jellyfish exhibits active motion, i.e. moving towards/away from a randomly chosen jellyfish
                    step = random.random()
                    #randomly choose another jellyfish for comparison
                    k = random.randint(0, nJellyfish-2)
                    if k >= j:
                        k += 1
                    
                    if fitness(jellyfish[j]) <= fitness(jellyfish[k]): #move away from k, approximated as making swaps that DON'T move towards k
                        swps = {tuple(sorted(s)) for s in swaps(jellyfish[j], jellyfish[k])}
                        num_to_swap = int(len(swps)*step)

                        for _ in range(num_to_swap):
                            for _ in range(100): #max number of tries to find a swap away from k
                                a, b = random.sample(range(n_odds), 2)
                                s = tuple(sorted((a, b)))
                                if s not in swps:
                                    jellyfish[j][a], jellyfish[j][b] = jellyfish[j][b], jellyfish[j][a]
                                    break
                            

                    else: #move towards k
                        jellyfish[j] = apply_swaps(jellyfish[j], swaps(jellyfish[j], jellyfish[k]), step)

            current_fitness = fitness(jellyfish[j])
            jellyfish_fitnesses[j] = current_fitness

            if current_fitness < iteration_best_cost:
                iteration_best_cost = current_fitness
                iteration_best_sol = jellyfish[j][:]
        
        if iteration_best_cost < best_jellyfish_cost:
            stagnation = 0
            best_jellyfish_cost = iteration_best_cost
            best_jellyfish = iteration_best_sol[:]

        else:
            stagnation += 1

        if time.time() > end_time or stagnation > stag_limit:
            break

    global_best_sol = list_to_pairs(best_jellyfish)

    if score_only: #this is an optimisation for large performance comparison experiments
        solution = "n/a"
        total = pairs_to_score(graph, global_best_cost=best_jellyfish_cost, edge_total=edge_total)
    else:
        solution, total = pairs_to_sol(graph, odd_subgraph, odd_indexes, subgraph_of_paths, global_best_sol)
    
    return solution, total