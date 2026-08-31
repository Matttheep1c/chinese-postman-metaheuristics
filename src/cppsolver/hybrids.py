#This .py file contains 2 hybrids of ACO and SA
#ACO-SA uses SA to improve solutions and apply extra pheromone reinforcement during convergence
#ACO-SA2 uses SA to simply refine the final solution from ACO

import math
import random
from typing import Tuple
import time
from cppsolver.helper_functions import to_odd_subproblem, pairs_to_sol, pairs_to_score

def ACO_SA(
        graph: list[list[int]],
        odd_info: list[list[list[int]], list[int], list[list[list[int]]]] = None,
        score_only: bool = False,
        edge_total: int = None,
        n_ants: int = 10,
        ACO_it: int = 1000,
        evap_rate: float = 0.01,
        reinforce_amount: float = 1,
        alpha: float = 1.9,
        beta: float = 5,
        start_temp: int = 1000,
        SA_it: int = 100,
        cooling_rate: float = 0.96,
        interval: int = 10,
        time_limit: float = 120,
        stag_limit: int = 250
        ) -> Tuple[list[int], int]:
    """
    Hybrid ACO-SA algorithm. ACO solutions are improved with SA.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_info (list[list[list[int]], list[int], list[list[list[int]]]]): odd_subgraph, odd_indexes and subgraph_of_paths from to_odd_subproblem.
        score_only (bool): if True then only the score of the solution will be returned, faster.
        edge_total (int): can optionally be precomputed if using score_only.
        n_ants (int): number of ants.
        ACO_it (int): number of iterations of ACO.
        evap_rate (float): pheromone evapouration rate.
        reinforce_amount (float): pheromone reinforcement.
        alpha (float): pheromone matrix scalar.
        beta (float): heuristic matrix scalar.
        start_temp (int): starting temperature.
        SA_it (int): number of iterations of simulated annealing.
        cooling_rate (float): cooling rate.
        interval (int): how often local SA is applied to iteration best.
        time_limit (float): time limit in seconds.
        stag_limit (int): terminates if this many iterations pass without improvement.

    Returns:
        Tuple[list[int], int]:
        - list[int]: solution tour of the original graph.
        - int: cost of the tour.
    """
    start_time = time.time()
    end_time = start_time + time_limit

    def local_SA(iteration_best_sol, start_temp, SA_it, cooling_rate): #simulated annealing as in sa.py. As this is applied many times, I recommend fewer iterations and a more aggressive cooling schedule
        def fitness(x):
            total = 0
            for pair in x:
                total += odd_subgraph[pair[0]][pair[1]]
            return total
        
        def to_sol(x):
            sol = [[x[i], x[i+1]] for i in range(0, n_odds, 2)]
            return sol
        
        def swap(x):
            n = len(x)
            move = random.choice(["swap", "reverse", "insert"])
            if move == "swap":
                u, v = random.sample(range(n), 2)
                x[u], x[v] = x[v], x[u]
            elif move == "reverse":
                u, v = sorted(random.sample(range(n), 2))
                x[u:v] = reversed(x[u:v])
            else:
                u, v = random.sample(range(n), 2)
                temp = x.pop(u)
                x.insert(v, temp)
            return x
        
        global_best_sol = [x for pair in iteration_best_sol for x in pair]
        global_best_fitness = fitness(to_sol(global_best_sol))

        current_sol = global_best_sol.copy()
        current_fitness = global_best_fitness

        for i in range(SA_it):

            T = start_temp * (cooling_rate**i)

            temp_sol = current_sol.copy()
            temp_sol = swap(temp_sol)
            temp_fitness = fitness(to_sol(temp_sol))

            delta = temp_fitness - current_fitness

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-20)): #simulated annealing acceptance rule
                current_sol = temp_sol
                current_fitness = temp_fitness
            
                if current_fitness < global_best_fitness:
                    global_best_sol = current_sol.copy()
                    global_best_fitness = current_fitness

            if time.time() > end_time:
                break

        return to_sol(global_best_sol)

    if not odd_info: #this allows to_odd_subproblem to be precomputed for large experiments
        odd_subgraph, odd_indexes, subgraph_of_paths = to_odd_subproblem(graph)
    else:
        odd_subgraph, odd_indexes, subgraph_of_paths = odd_info

    n_odds = len(odd_indexes)

    h_matrix = [[1/j if j != 0 else 0 for j in odd_subgraph[i]] for i in range(n_odds)]

    epsilon = 1e-4
    p_matrix = [[epsilon if i != j else 0 for j in range(n_odds)] for i in range(n_odds)]


    global_best_cost = float("inf")
    global_best_sol = []

    stagnation = 0
    for iteration in range(ACO_it):

        iteration_best_cost = float("inf")
        iteration_best_sol = []

        ant_pair_sets = []
        for ant in range(n_ants):
            unvisited = [i for i in range(n_odds)]

            pairs = []
            while len(unvisited) > 0:
                start = random.choice(unvisited)
                unvisited.remove(start)

                weights = [p_matrix[start][next_node]**alpha * h_matrix[start][next_node]**beta for next_node in unvisited]
                total_w = sum(weights)
                if total_w == 0:
                    next_node = random.choice(unvisited)
                else:
                    probs = [w/total_w for w in weights]
                    next_node = random.choices(unvisited, weights = probs, k = 1)[0]
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
                iteration_best_sol = [pair[:] for pair in pairs]
            
            for pair in pairs: #undirected edges: update both
                p_matrix[pair[0]][pair[1]] += reinforce_amount / total_cost
                p_matrix[pair[1]][pair[0]] += reinforce_amount / total_cost

        if (iteration+1)%interval == 0: #apply SA every n iterations to the iteration best. This enhanced solution is allowed to reinforce pheromones once more
            iteration_best_sol = local_SA(iteration_best_sol, start_temp=start_temp, SA_it=SA_it, cooling_rate=cooling_rate)    
            iteration_best_cost = 0
            for pair in iteration_best_sol:
                pair_cost = odd_subgraph[pair[0]][pair[1]]
                iteration_best_cost += pair_cost

            for pair in iteration_best_sol: #extra p update for refined solution
                p_matrix[pair[0]][pair[1]] += reinforce_amount / iteration_best_cost
                p_matrix[pair[1]][pair[0]] += reinforce_amount / iteration_best_cost



        if iteration_best_cost < global_best_cost:
            stagnation = 0
            global_best_cost = iteration_best_cost
            global_best_sol = [pair[:] for pair in iteration_best_sol]
        else:
            stagnation += 1
        
        if time.time() > end_time or stagnation > stag_limit:
            break
    
    #print("Global best cost:", global_best_cost)

    if score_only:
        solution = "n/a"
        total = pairs_to_score(graph, global_best_cost=global_best_cost, edge_total=edge_total)
    else:
        solution, total = pairs_to_sol(graph, odd_subgraph, odd_indexes, subgraph_of_paths, global_best_sol)
    
    return solution, total

def ACO_SA2(
        graph: list[list[int]],
        odd_info: list[list[list[int]], list[int], list[list[list[int]]]] = None,
        score_only: bool = False,
        edge_total: int = None,
        n_ants: int = 10,
        ACO_it: int = 1000,
        evap_rate: float = 0.07,
        reinforce_amount: float = 1,
        alpha: float = 1.4,
        beta: float = 3.4,
        start_temp: int = 1000,
        SA_it: int = 50000,
        cooling_rate: float = 0.995,
        time_limit: float = 120,
        stag_limit: int = 250
        ) -> Tuple[list[int], int]:
    """
    Hybrid ACO-SA algorithm. ACO final solution is improved with SA.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_info (list[list[list[int]], list[int], list[list[list[int]]]]): odd_subgraph, odd_indexes and subgraph_of_paths from to_odd_subproblem.
        score_only (bool): if True then only the score of the solution will be returned, faster.
        edge_total (int): can optionally be precomputed if using score_only.
        n_ants (int): number of ants.
        ACO_it (int): number of iterations of ACO.
        evap_rate (float): pheromone evapouration rate.
        reinforce_amount (float): pheromone reinforcement.
        alpha (float): pheromone matrix scalar.
        beta (float): heuristic matrix scalar.
        start_temp (int): starting temperature.
        SA_it (int): number of iterations of simulated annealing.
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
    def local_SA(iteration_best_sol, start_temp, SA_it, cooling_rate):
        def fitness(x):
            total = 0
            for pair in x:
                total += odd_subgraph[pair[0]][pair[1]]
            return total
        
        def to_sol(x):
            sol = [[x[i], x[i+1]] for i in range(0, n_odds, 2)]
            return sol
        
        def swap(x):
            n = len(x)
            move = random.choice(["swap", "reverse", "insert"])
            if move == "swap":
                u, v = random.sample(range(n), 2)
                x[u], x[v] = x[v], x[u]
            elif move == "reverse":
                u, v = sorted(random.sample(range(n), 2))
                x[u:v] = reversed(x[u:v])
            else:
                u, v = random.sample(range(n), 2)
                temp = x.pop(u)
                x.insert(v, temp)
            return x
        
        global_best_sol = [x for pair in iteration_best_sol for x in pair]
        global_best_fitness = fitness(to_sol(global_best_sol))

        current_sol = global_best_sol.copy()
        current_fitness = global_best_fitness

        for i in range(SA_it):

            T = start_temp * (cooling_rate**i)

            temp_sol = current_sol.copy()
            temp_sol = swap(temp_sol)
            temp_fitness = fitness(to_sol(temp_sol))

            delta = temp_fitness - current_fitness

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-20)):
                current_sol = temp_sol
                current_fitness = temp_fitness
            
                if current_fitness < global_best_fitness:
                    global_best_sol = current_sol.copy()
                    global_best_fitness = current_fitness
            
            if time.time() > end_time:
                break

        return to_sol(global_best_sol)

    if not odd_info: #this allows to_odd_subproblem to be precomputed for large experiments
        odd_subgraph, odd_indexes, subgraph_of_paths = to_odd_subproblem(graph)
    else:
        odd_subgraph, odd_indexes, subgraph_of_paths = odd_info

    n_odds = len(odd_indexes)

    h_matrix = [[1/j if j != 0 else 0 for j in odd_subgraph[i]] for i in range(n_odds)]




    epsilon = 1e-4
    p_matrix = [[epsilon if i != j else 0 for j in range(n_odds)] for i in range(n_odds)]

    global_best_cost = float("inf")
    global_best_sol = []

    stagnation = 0
    for iteration in range(ACO_it):

        iteration_best_cost = float("inf")
        iteration_best_sol = []

        ant_pair_sets = []
        for ant in range(n_ants):
            unvisited = [i for i in range(n_odds)]

            pairs = []
            while len(unvisited) > 0:
                start = random.choice(unvisited)
                unvisited.remove(start)

                weights = [p_matrix[start][next_node]**alpha * h_matrix[start][next_node]**beta for next_node in unvisited]
                total_w = sum(weights)
                if total_w == 0:
                    next_node = random.choice(unvisited)
                else:
                    probs = [w/total_w for w in weights]
                    next_node = random.choices(unvisited, weights = probs, k = 1)[0]
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
                iteration_best_sol = [pair[:] for pair in pairs]
            
            for pair in pairs: #undirected edges: update both
                p_matrix[pair[0]][pair[1]] += reinforce_amount / total_cost
                p_matrix[pair[1]][pair[0]] += reinforce_amount / total_cost


        if iteration_best_cost < global_best_cost:
            stagnation = 0
            global_best_cost = iteration_best_cost
            global_best_sol = [pair[:] for pair in iteration_best_sol]
        
        else:
            stagnation += 1

        if time.time() > end_time - time_limit*0.1 or stagnation > stag_limit:
            break
    
    global_best_sol = local_SA(global_best_sol, start_temp=start_temp, SA_it=SA_it, cooling_rate=cooling_rate) #simply applies simulated annealing to the final solution
    global_best_cost = 0
    for pair in global_best_sol:
        pair_cost = odd_subgraph[pair[0]][pair[1]]
        global_best_cost += pair_cost

    #print("Global best cost:", global_best_cost)

    if score_only:
        solution = "n/a"
        total = pairs_to_score(graph, global_best_cost=global_best_cost, edge_total=edge_total)
    else:
        solution, total = pairs_to_sol(graph, odd_subgraph, odd_indexes, subgraph_of_paths, global_best_sol)
    
    return solution, total