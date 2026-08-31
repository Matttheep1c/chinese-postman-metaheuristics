#Hyperparameter tuning using Optuna (Bayesian Optimisation)

#hyperparameters are tuned to 2sf where appropriate with a 20 round study
#the study minimises the 5-run mean on graph100

########################################################################

from cppsolver.helper_functions import load_graph, edge_sum, to_odd_subproblem
from cppsolver.aco import ACO
from cppsolver.pso import PSO
from cppsolver.sa import simulated_annealing
from cppsolver.jso import JSO
from cppsolver.hybrids import ACO_SA, ACO_SA2
from cppsolver.solution_validator import solution_check
import optuna
import numpy as np
from pathlib import Path

########################################################################

#the study is currently set up to tune PSO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def objective(trial):
    p_mutate = trial.suggest_float("p_mutate", 0.01, 0.99, step = 0.01)
    alpha = trial.suggest_float("alpha", 0.01, 0.99, step=0.01)
    beta = trial.suggest_float("beta", 0.01, 0.99, step=0.01)
    
    results = []
    for i in range(5):
        _, total = PSO(
            graph=graph,
            odd_info=odd_info,
            score_only=score_only,
            edge_total=edge_total,
            n_particles=particle_actors,
            iterations=budget//particle_actors,
            alpha=alpha,
            beta=beta,
            p_mutate=p_mutate,
            time_limit=time_limit,
            stag_limit=stag_limit
        )
        results.append(total)

    results = np.array(results)
    
    return np.mean(results)

filename = PROJECT_ROOT / "graphs/graph100.txt"

#predefine non-tuning hyperparameters for all algorithms
budget = 20000
ant_actors = 10
particle_actors = 50
population_actors = 100

interval = 25
aco_sa_local_prop = 0.9
aco_sa_global_prop = 0.9

time_limit = float("inf")
stag_limit = float("inf")

ACO_it_hybrid_local = int(aco_sa_local_prop*budget)//ant_actors
SA_it_hybrid_local = (budget - ACO_it_hybrid_local*ant_actors)//(ACO_it_hybrid_local//interval)
ACO_it_hybrid_global = int(aco_sa_global_prop*budget)//ant_actors
SA_it_hybrid_global = budget - ACO_it_hybrid_global*ant_actors

graph, N_nodes, N_odds, edge_count, W_max = load_graph(filename)
odd_info = to_odd_subproblem(graph)
edge_total = edge_sum(graph)
score_only = True



print("Starting optuna parameter study...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials = 20)
print("Study complete!")
print(study.best_params)