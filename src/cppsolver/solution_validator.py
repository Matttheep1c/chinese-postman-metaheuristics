#validator to check solution correctness

from typing import Tuple
import copy

def solution_check(graph: list[list[int]], solution: list, total: int) -> None:
    """
    CPP solution validator.

    Args:
        graph (list[list[int]]): graph[i][j] represents the edge from i to j.
        solution (list[int]): solution tour of the original graph.
        total (int): cost of the tour.

    Side Effects:
        Prints result and reasons.
    """
    result = True
    tally = 0
    check_graph = copy.deepcopy(graph)
    reasons = []
    for i in range(len(solution)-1):
        u = solution[i]
        v = solution[i+1]
        if graph[u][v] != 0:
            tally += graph[u][v]
            check_graph[u][v] = 0
            check_graph[v][u] = 0
        else:
            result = False
            reasons.append("You tried to traverse a non-existent edge.")

    weight_missed = 0
    for row in check_graph:
        weight_missed += sum(row)

    if solution[0] != solution [-1]:
        result = False
        reasons.append("It didn't start/end in the same place.")
    if tally != total:
        result = False
        reasons.append("The weight was calculated incorrectly.")
    if weight_missed != 0:
        result = False
        reasons.append("You missed edge(s).")

    if result == True:
        print("Congratulations. You have produced a valid tour of length", total)
    else:
        print("Your tour is invalid for the following reason(s):", reasons)