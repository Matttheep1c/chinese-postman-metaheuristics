#Metaheuristics for CPP graph generator
import random
import os

def connectivity(graph): #DFS based connectivity checker
    nodes = len(graph)
    unvisited = [i for i in range(nodes)]
    current_point = 0
    unvisited.remove(current_point)

    def explore(current_point):
        for i in range(nodes):
            if graph[current_point][i] != 0 and i in unvisited:
                unvisited.remove(i)
                explore(i)

    explore(current_point)
    
    if len(unvisited) == 0:
        return True
    else:
        return False

def gen_graph(N_nodes, N_odds, W_max):
    print("Generating graph...")
    G = [] #initialise graph with 0s representing no edge
    for i in range(N_nodes):
        row = []
        for j in range(N_nodes):
            row.append(0)
        G.append(row)
    
    odd_count = 0
    iteration = 0
    connected = False
    while odd_count != N_odds:
        i = 0
        j = 0
        while i == j or G[i][j] == 1: #makes sure we don't have self loops
            i = random.randrange(0, N_nodes)
            j = random.randrange(0, N_nodes)
        G[i][j] = 1 #randomly add an edge
        G[j][i] = 1 #in both directions

        if iteration > N_nodes - 1 and connected == False: #optimisation to avoid computing connectivity every iteration
            connected = connectivity(G)

        if connected == True: #use DFS to check connectivity and then check odd nodes if connected
            odd_count = 0 #check how many odd nodes
            for i in range(N_nodes):
                if sum(G[i]) % 2 == 1:
                    odd_count += 1
    
        iteration += 1

    valency_count = 0 #count edges
    for i in range(N_nodes):
        valency_count += sum(G[i])
    
    global edge_count
    edge_count = valency_count // 2

    for i in range(N_nodes):
        for j in range(N_nodes):
            if G[i][j] == 1:
                w = random.randrange(1, W_max+1)
                G[i][j] = w
                G[j][i] = w

    print(f"Graph generated with {N_nodes} total nodes {odd_count} odd nodes and {edge_count} edges.")
    return G

####################################
N_nodes = 500
N_odds = 250
W_max = 1000
####################################

graph = gen_graph(N_nodes, N_odds, W_max)

def save_graph(graph, N_nodes, N_odds, W_max, edge_count):
    folder = "graphs"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"graph{N_nodes}.txt")
    with open(filename, "w") as f:
        f.write(f"{N_nodes}\n")
        f.write(f"{N_odds}\n")
        f.write(f"{edge_count}\n")
        f.write(f"{W_max}\n")
        for row in graph:
            f.write(" ".join(str(i) for i in row) + "\n")
    print("Graph saved in", filename)

save_graph(graph, N_nodes, N_odds, W_max, edge_count)