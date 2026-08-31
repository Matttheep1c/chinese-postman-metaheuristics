#helper functions involved in breaking down the problem and constructing solutions

import heapq
from typing import Tuple
import numpy as np

def load_graph(filename: str) -> Tuple[list[list[int]], int, int, int, int]:
    """
    Loads graphX from the graphs folder.

    Args:
        filename (str): "graphs/graphX.txt"
    Returns:
        Tuple[list[list[int]], int, int, int, int]:
        - list[list[int]]: graph[i][j] represents the edge from i to j.
        - int: total number of nodes.
        - int: number of odd valency nodes.
        - int: number of edges.
        - int: maximum edge weight.
    """
    with open(filename) as f:
        lines = f.readlines()

    N_nodes = int(lines[0])
    N_odds = int(lines[1])
    edge_count = int(lines[2])
    W_max = int(lines[3])

    graph = [list(map(int, line.split())) for line in lines[4:]]

    return(graph, N_nodes, N_odds, edge_count, W_max)


def edge_sum(graph: list[list[int]]) -> int:
    """
    Helper function for pairs_to_score. Optionally can be precomputed and passed to pairs_to_score.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.

    Returns:
        int: sum of all edges in the graph.
    """
    g = np.array(graph)
    edge_total = g[np.triu_indices_from(g, k=1)].sum()
    return edge_total

def dijkstra(graph: list[list[int]], total_nodes: int, start: int) -> Tuple[list[int], list[list[int]]]:
    """
    Shortest path from start to all.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        total_nodes (int): total number of nodes.
        start (int): the starting node.

    Returns:
        Tuple[list[int], list[list[int]]]:
        - list[int]: list of minimum distances from start node to i.
        - list[list[int]]: list of shortest paths from start node to i.
    """
    distance = [float("inf")]*total_nodes
    prev = [None]*total_nodes
    distance[start] = 0
    
    queue = [(0, start)] #(min_distance, node)
    it_check = 0

    while len(queue) > 0:
        #print("Here is the queue on while loop iteration",it_check,queue)
        current_dist, current_node = heapq.heappop(queue)
        
        if current_dist > distance[current_node]:
            continue #?

        for new_node in range(total_nodes):
            weight = graph[current_node][new_node]
            if weight > 0:
                new_dist = distance[current_node] + weight
                if new_dist < distance[new_node]:
                    distance[new_node] = new_dist
                    prev[new_node] = current_node
                    heapq.heappush(queue, (new_dist, new_node))
        it_check += 1

        def build_path(start, end, prev):
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = prev[current]
            path.reverse()
            
            return path if path[0] == start else []

        paths = []
        for node in range(total_nodes):
            if distance[node] == float("inf") or distance[node] == 0:
                paths.append([])
            else:
                paths.append(build_path(start, node, prev))

    return distance, paths

def to_odd_subproblem(graph: list[list[int]]) -> Tuple[list[list[int]], list[int], list[list[list[int]]]]:
    """
    Simplifies the problem by converting the graph into a subgraph of odd nodes to be paired up.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
    
    Returns:
        Tuple[list[list[int]], list[int], list[list[list[int]]]]:
        - list[list[int]]: Odd node subgraph with odd_subgraph[i][j] representing the minimum distance from node odd_index[i] to odd_index[j], see below.
        - list[int]: Odd node indexes. Node i in the odd subgraph corresponds to odd_indexes[i] in the orginal graph.
        - list[list[list[int]]]: subgraph_of_paths[i][j] contains the shortest path between the two nodes in list form.
    """
    total_nodes = len(graph)
    odd_indexes = [i for i in range(total_nodes) if sum(1 for w in graph[i] if w != 0) % 2 == 1]
    
    odd_subgraph = [[] for i in range(len(odd_indexes))]
    subgraph_of_paths = [[] for i in range(len(odd_indexes))]

    for i in range(len(odd_indexes)):
        i_dijkstras, i_paths = dijkstra(graph, total_nodes, odd_indexes[i]) #get dijkstras from i to all (i is odd)
        odd_paths = []
        odd_dijkstras = []
        for j in odd_indexes: #filter to i to j (where both are odd)
            odd_paths.append(i_paths[j])
            odd_dijkstras.append(i_dijkstras[j])
        odd_subgraph[i] = odd_dijkstras
        subgraph_of_paths[i] = odd_paths
    
    #print("odd subgraph")
    #for row in odd_subgraph:
    #    print(row)

    #print("subgraph of paths")
    #for row in subgraph_of_paths:
    #    print(row)

    return odd_subgraph, odd_indexes, subgraph_of_paths

def pairs_to_sol(graph: list[list[int]], odd_subgraph: list[list[int]], odd_indexes: list[int], subgraph_of_paths: list[list[list[int]]], pair_sol: list[list[int]]) -> Tuple[list[int], int]:
    """
    Connects a set of odd node pairings to full graph to get a full solution via Hierholzer's algorithm.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        odd_subgraph (list[list[int]]): Odd node subgraph with odd_subgraph[i][j] representing the minimum distance from node odd_index[i] to odd_index[j], see below.
        odd_indexes (list[int]): Odd node indexes. Node i in the odd subgraph corresponds to odd_indexes[i] in the orginal graph.
        subgraph_of_paths (list[list[list[int]]]): subgraph_of_paths[i][j] contains the shortest path between the two nodes in list form.
        pair_sol (list[list[int]]): list of odd node pairs.

    Returns:
        Tuple[list[int], int]:
        - list[int]: solution tour of the original graph.
        - int: cost of the tour.
    """
    augmented_graph = [[[j, 0] if j == 0 else [j, 1] for j in graph[i]] for i in range(len(graph))]

    for pair in pair_sol:
        paths_to_add = subgraph_of_paths[pair[0]][pair[1]]
        for i in range(len(paths_to_add)-1):
            u = paths_to_add[i]
            v = paths_to_add[i+1]
            augmented_graph[u][v][1] += 1
            augmented_graph[v][u][1] += 1
    
    #hierholzer's algorithm:
    node_count = len(augmented_graph)
    stack = [0] #start node
    circuit = []
    total = 0
    while len(stack) > 0:
        v = stack[-1]
        edge_found = False
        for u in range(node_count):
            dist, count = augmented_graph[u][v]
            if count > 0:
                augmented_graph[u][v][1] -= 1 #use the edge
                augmented_graph[v][u][1] -= 1
                total += augmented_graph[u][v][0] #track total weight
                stack.append(u)
                edge_found = True
                break

        if edge_found == False:
            circuit.append(stack.pop())

    return circuit[::-1], total

def pairs_to_score(graph: list[list[int]], global_best_cost: int, edge_total: int = None) -> int:
    """
    Lightweight alternative to pairs_to_sol. Only returns the final tour cost.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        global_best_cost (int): the combined cost of the pairing of odd nodes.
        edge_total (int): the sum of all edges in the graph, optional.
    Returns:
        int: the total cost of the tour.
    """
    if not edge_total:
        edge_total = edge_sum(graph)
    return edge_total + global_best_cost