import os
import sys
import copy
import numpy as np
from paretoset import paretoset
from joblib import Parallel, delayed
from functools import reduce
from pyroaring import BitMap
from typing import Optional

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from .rule_utils import eval_ruleset_coverage, eval_ruleset


def bitmap_to_array(bm: BitMap,
                    data_size: int):
    """
    convert BitMap to 1-D boolean array
    """
    assert data_size >= 1
    l = [False] * data_size
    for i in bm:
        l[i] = True
    return l


def array_to_bitmap(array):
    """
    convert 1-D boolean array to BitMap
    """
    bm = BitMap()
    for i, v in enumerate(array):
        if v == 1:
            bm.add(i)
    return bm


def sort_solutions(solutions):
    """
    sort solutions based on recall in ascending order
    """
    rank = np.argsort(solutions[:, -1], axis=0, kind='stable')
    sorted_solutions = solutions[rank]
    return sorted_solutions, rank


def index_of_solutions(solution,
                       solutions):
    """
    find the index of a solution in sorted solutions
    """
    sorted_solutions, _ = sort_solutions(solutions)
    idx = 0
    if len(sorted_solutions) == 0:
        return idx
    for sol in sorted_solutions:
        if solution[-1] <= sol[-1]:
            assert(solution[-2] >= sol[-2])
            return idx
        idx += 1
    return idx


def pareto_solutions(solutions,
                     objectives_num=2,
                     sense=["max", "max"],
                     ruleset_coverage: Optional[list]=None,
                     positive_indices: Optional[BitMap]=None,
                     ):
    """
    find pareto optimal solutions
    """
    paretos = np.copy(solutions)
    if ruleset_coverage and positive_indices:
        for i in range(paretos.shape[0]):
            c = ruleset_coverage[i]
            [p, r] = eval_ruleset_coverage(c, positive_indices)
            paretos[i][-2] = p
            paretos[i][-1] = r
    pareto_mask = paretoset(paretos[:, -objectives_num:], sense)
    paretos = paretos[pareto_mask]
    return paretos, pareto_mask


def compute_hv2d(solutions):
    """
    compute the hv-indicator of the given pareto optimal solutions in 2-D precision-recall space referenced by (0, 0)
    """
    solutions, _ = sort_solutions(solutions)
    hv = 0
    last_recall = 0
    for i, v in enumerate(solutions):
        prec, recall = v[-2:]
        hv += prec * (recall - last_recall)
        last_recall = recall
    return hv


def compute_hvc2d(new_solutions,
                  solutions):
    """
    compute the hv contribution of the new solutions to given pareto optimal solutions
    """
    if new_solutions.shape[0] == 1:
        return _fast_hvc2d(new_solutions, solutions)
    hv = compute_hv2d(solutions)
    new_solutions = np.vstack([solutions, new_solutions])
    new_solutions, _ = pareto_solutions(new_solutions)
    new_solutions, _ = sort_solutions(new_solutions)
    hvc = compute_hv2d(new_solutions) - hv
    return hvc


def _fast_hvc2d(new_solution,
                solutions):
    """
    an efficient way to compute the hvc of a single solution
    """
    assert new_solution.shape[0] == 1
    if solutions.shape[0] == 0:
        return compute_hv2d(new_solution)
    solutions, _ = sort_solutions(solutions)
    new_prec, new_recall = new_solution[0, -2:]
    dominated_indices = []
    for idx, sol in enumerate(solutions):
        prec, recall = sol[-2:]
        if (prec <= new_prec and recall < new_recall) or (recall <= new_recall and prec < new_prec):
            dominated_indices.append(idx)
    if len(dominated_indices) == 0:
        insert_idx = index_of_solutions(new_solution[0], solutions)
        prev_recall = 0 if insert_idx == 0 else solutions[insert_idx - 1][-1]
        next_prec = 0 if insert_idx == solutions.shape[0] else solutions[insert_idx][-2]
        hvc = (new_prec - next_prec) * (new_recall - prev_recall)
    else:
        min_index = dominated_indices[0]
        max_index = dominated_indices[-1]
        dominated_solutions = solutions[min_index:max_index+1, -2:]
        min_solution = [new_prec, 0] if min_index == 0 else [new_prec, solutions[min_index - 1][-1]]
        max_solution = [0, new_recall] if max_index == solutions.shape[0] - 1 else \
                       [solutions[max_index + 1][-2], new_recall]
        dominated_solutions = np.vstack([dominated_solutions, np.array([min_solution, max_solution])])
        hvc = new_prec * new_recall - compute_hv2d(dominated_solutions)
    return hvc


def _gen_next_solutions_helper(indices,
                               args):
    """
    expand current pareto front
    """
    (solutions, rule_coverage, ruleset_coverage, pos_sample_indices) = args
    candidates = []
    candidates_objectives = []
    next_ruleset_coverage = []

    for i in indices:
        ruleset = solutions[i, :-2]
        train_coverage, valid_coverage = ruleset_coverage[i]
        for j in range(len(ruleset)):
            if ruleset[j] == 0:
                candidate = np.copy(ruleset)
                candidate[j] = 1
                candidates.append(candidate)
                candidate_train_coverage = train_coverage | rule_coverage[j][0]
                candidate_valid_coverage = valid_coverage | rule_coverage[j][1]
                next_ruleset_coverage.append((candidate_train_coverage, candidate_valid_coverage))
                candidate_objectives = eval_ruleset_coverage(candidate_train_coverage, pos_sample_indices)
                candidates_objectives.append(candidate_objectives)
    candidates = np.array(candidates)
    candidates_objectives = np.array(candidates_objectives)
    next_solutions = np.hstack([candidates, candidates_objectives])
    return next_solutions, next_ruleset_coverage


def gen_next_solutions(solutions: np.array,
                       rule_coverage: list,
                       ruleset_coverage: list,
                       pos_sample_indices: BitMap,
                       n_process=4):
    """
    expand current pareto front
    """
    def merge_solutions(s1, s2):
        if s1.shape[0] == 0:
            return s2
        if s2.shape[0] == 0:
            return s1
        return np.vstack([s1, s2])

    sub_solutions_indices = [[] for _ in range(n_process)]
    for i in range(solutions.shape[0]):
        sub_solutions_indices[i % n_process].append(i)
    args = (solutions, rule_coverage, ruleset_coverage, pos_sample_indices)
    global_results = Parallel(n_jobs=n_process, backend='multiprocessing')(
        delayed(_gen_next_solutions_helper)(indexes, args) for indexes in sub_solutions_indices)
    global_solutions = [g[0] for g in global_results]
    next_solutions = reduce(lambda s1, s2: merge_solutions(s1, s2), global_solutions)
    next_ruleset_coverage = []
    for g in global_results:
        next_ruleset_coverage += g[1]
    return next_solutions, next_ruleset_coverage


def eval_solutions(solutions: np.array,
                   rule_id_list: list,
                   rule_coverage: dict,
                   positive_indices: BitMap):
    new_solutions = np.copy(solutions)
    for idx in range(solutions.shape[0]):
        s = solutions[idx]
        ruleset = []
        for i in array_to_bitmap(s[:-2]):
            ruleset.append(rule_id_list[i])
        [prec, recall] = eval_ruleset(ruleset, rule_coverage, positive_indices)
        new_solutions[idx][-2] = prec
        new_solutions[idx][-1] = recall
    new_solutions, _ = pareto_solutions(new_solutions)
    test_hv = compute_hv2d(new_solutions)
    return test_hv


def eval_single_solution(solution, rule_id_list, rule_coverage, positive_indices):
    ruleset = []
    for i in array_to_bitmap(solution[:-2]):
        ruleset.append(rule_id_list[i])
    [prec, recall] = eval_ruleset(ruleset, rule_coverage, positive_indices)
    return prec, recall


def crowding_distance_function(solutions,
                               ndim=2):
    """
    Function: Crowding Distance (Adapted from PYMOO)
    """
    infinity = 1e+11
    population = copy.deepcopy(solutions[:, -ndim:])
    population = population.reshape((solutions.shape[0], ndim))
    if population.shape[0] <= 2:
        return np.full(population.shape[0], infinity)
    else:
        arg_1 = np.argsort(population, axis=0, kind='mergesort')
        population = population[arg_1, np.arange(ndim)]
        dist = np.concatenate([population, np.full((1, ndim), np.inf)]) - np.concatenate(
            [np.full((1, ndim), -np.inf), population])
        idx = np.where(dist == 0)
        a = np.copy(dist)
        b = np.copy(dist)
        for i, j in zip(*idx):
            a[i, j] = a[i - 1, j]
        for i, j in reversed(list(zip(*idx))):
            b[i, j] = b[i + 1, j]
        norm = np.max(population, axis=0) - np.min(population, axis=0)
        norm[norm == 0] = np.nan
        a, b = a[:-1] / norm, b[1:] / norm
        a[np.isnan(a)] = 0.0
        b[np.isnan(b)] = 0.0
        arg_2 = np.argsort(arg_1, axis=0)
        crowding = np.sum(a[arg_2, np.arange(ndim)] + b[arg_2, np.arange(ndim)], axis=1) / ndim
    crowding[np.isinf(crowding)] = infinity
    crowding = crowding.reshape((-1, 1))
    return crowding


def crowded_operator(rank,
                     crowding_distance,
                     individual_1=0,
                     individual_2=1):
    """
     Function:Crowded Comparison Operator
    """
    selection = False
    if (rank[individual_1] < rank[individual_2]) or ((rank[individual_1] == rank[individual_2]) and (
            crowding_distance[individual_1] > crowding_distance[individual_2])):
        selection = True
    return selection


def fast_non_dominated_sorting(solutions,
                               topk):
    n = solutions.shape[0]
    topk_solutions = np.empty(shape=(0, solutions.shape[1]))
    rank_num = 0
    rank = []
    if n <= topk:
        rank += [[rank_num] for i in range(topk)]
        rank = np.array(rank)
        return solutions, rank
    while True:
        paretos, pareto_mask = pareto_solutions(solutions)
        if paretos.shape[0] >= topk:
            crowding_distance = crowding_distance_function(paretos, 2)
            sort_idx = np.argsort(crowding_distance[:, 0])[::-1]
            topk_solutions = np.vstack([topk_solutions, paretos[sort_idx][:topk]])
            rank += [[rank_num] for _ in range(topk)]
            return topk_solutions, rank
        else:
            topk -= paretos.shape[0]
            rank += [[rank_num] for i in range(paretos.shape[0])]
            topk_solutions = np.vstack([topk_solutions, paretos])
            non_pareto_mask = [not i for i in pareto_mask]
            solutions = solutions[non_pareto_mask]
        rank_num += 1
