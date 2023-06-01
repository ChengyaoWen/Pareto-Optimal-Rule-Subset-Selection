import numpy as np
from sklearn_extra.cluster import KMedoids
from tsp_solver.greedy import solve_tsp
from utils.moo_utils import *


def _hv_ss(solutions: np.array,
           ruleset_coverage: list,
           k=10,
           **kwargs):
    n = solutions.shape[0]
    solutions, rank = sort_solutions(solutions)
    ruleset_coverage = [ruleset_coverage[i] for i in rank]
    selected_solutions = np.empty(shape=(0, solutions.shape[1]))
    selected_ruleset_coverage = []

    c = np.zeros(n)
    for i in range(n):
        prec, recall = solutions[i][-2:]
        c[i] = prec * recall

    while selected_solutions.shape[0] < k:
        while True:
            cmax_idx = np.argmax(c)
            cmax_hvc = compute_hvc2d(solutions[cmax_idx:cmax_idx + 1, :], selected_solutions)
            c[cmax_idx] = cmax_hvc
            cmax_idx_new = np.argmax(c)
            if cmax_idx_new == cmax_idx:
                insert_idx = index_of_solutions(solutions[cmax_idx], selected_solutions)
                selected_solutions = np.insert(selected_solutions, insert_idx, [solutions[cmax_idx]], axis=0)
                selected_ruleset_coverage.insert(insert_idx, ruleset_coverage[cmax_idx])
                c[cmax_idx] = -1
                break
    return selected_solutions, selected_ruleset_coverage


def _igd_ss(solutions: np.array,
            ruleset_coverage: list,
            k=10,
            metric='igd',
            **kwargs):
    if metric not in ['igd', 'igd+']:
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s " % (metric, ['igd', 'igd+']))
    n = solutions.shape[0]
    solutions, rank = sort_solutions(solutions)
    ruleset_coverage = [ruleset_coverage[i] for i in rank]

    selected_indices = []
    candidate_indices = list(np.arange(n))
    dmatrix = []
    for s1 in solutions:
        s1_prec, s1_recall = s1[-2:]
        dlist = []
        for s2 in solutions:
            s2_prec, s2_recall = s2[-2:]
            prec_d = s1_prec - s2_prec
            recall_d = s1_recall - s2_recall
            if metric == 'igd+':
                prec_d = max(0, prec_d)
                recall_d = max(0, recall_d)
            d = np.sqrt(np.square(prec_d) + np.square(recall_d))
            dlist.append(d)
        dmatrix.append(dlist)
    avg_dlist = [np.mean(v) for v in dmatrix]
    min_index = np.argmin(avg_dlist)
    prev_dlist = dmatrix[min_index]
    selected_indices.append(min_index)
    candidate_indices.remove(min_index)

    while len(selected_indices) < k:
        gains = [-1] * n
        for idx in candidate_indices:
            curr_dlist = dmatrix[idx]
            min_dlist = [min(prev_dlist[i], curr_dlist[i]) for i in range(len(prev_dlist))]
            gains[idx] = np.mean(prev_dlist) - np.mean(min_dlist)
        max_gain_index = np.argmax(gains)
        curr_dlist = dmatrix[max_gain_index]
        prev_dlist = [min(prev_dlist[i], curr_dlist[i]) for i in range(len(prev_dlist))]
        selected_indices.append(max_gain_index)
        candidate_indices.remove(max_gain_index)

    selected_solutions = solutions[selected_indices]
    selected_ruleset_coverage = [ruleset_coverage[i] for i in selected_indices]
    return selected_solutions, selected_ruleset_coverage


def _k_medoids_ss(solutions: np.array,
                  ruleset_coverage: list,
                  k=10,
                  metric='euclidean',
                  **kwargs):
    if metric not in ['euclidean', 'jaccard']:
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s " % (metric, ['euclidean', 'jaccard']))
    n = solutions.shape[0]
    if metric == 'jaccard':
        dataset_size = kwargs.get("dataset_size")
        if dataset_size is None:
            raise ValueError("missing parameter data_size")
        datas = np.array([bitmap_to_array(ruleset_coverage[i][0], dataset_size) for i in range(n)])
    else:
        datas = solutions[:, -2:]
    k_medoids = KMedoids(n_clusters=k, metric=metric, method="alternate", init="k-medoids++", max_iter=300)
    res = k_medoids.fit(datas)
    selected_indices = res.medoid_indices_
    selected_solutions = solutions[selected_indices]
    selected_ruleset_coverage = [ruleset_coverage[i] for i in selected_indices]
    return selected_solutions, selected_ruleset_coverage


def _hvc_ss(solutions: np.array,
            ruleset_coverage: list,
            k=10,
            prev_solutions: np.array = None,
            **kwargs):
    n = solutions.shape[0]
    if prev_solutions is None or prev_solutions.shape[0] == 0:
        return _hv_ss(solutions, ruleset_coverage, k)
    prev_solutions, _ = sort_solutions(prev_solutions)

    candidate_indices = list(range(n))
    selected_indices = []
    while len(selected_indices) < k:
        hvc_list = [-1] * n
        for i in candidate_indices:
            hvc_list[i] = compute_hvc2d(solutions[i:i+1, :], prev_solutions)
        max_hvc_index = np.argmax(hvc_list)
        selected_indices.append(max_hvc_index)
        candidate_indices.remove(max_hvc_index)
        prev_solutions = np.row_stack((prev_solutions, solutions[max_hvc_index]))
        prev_solutions, _ = pareto_solutions(prev_solutions)
        prev_solutions, _ = sort_solutions(prev_solutions)

    selected_solutions = solutions[selected_indices]
    selected_ruleset_coverage = [ruleset_coverage[i] for i in selected_indices]
    return selected_solutions, selected_ruleset_coverage


def _equi_dist_ss(solutions: np.array,
                  ruleset_coverage: list,
                  k=10,
                  **kwargs):

    def euclidean_dist(idx1, idx2):
        return np.sqrt(np.square(solutions[idx1][-2] - solutions[idx2][-2]) +
                       np.square(solutions[idx1][-1] - solutions[idx2][-1]))

    solutions, rank = sort_solutions(solutions)
    ruleset_coverage = [ruleset_coverage[i] for i in rank]
    n = solutions.shape[0]
    selected_indices = []
    if k == 1:
        selected_indices.append(n // 2)
    elif k == 2:
        selected_indices += [0, n-1]
    else:
        distances = np.cumsum([euclidean_dist(i-1, i) for i in range(1, n)])
        ideal_step = distances[-1] / (k - 1)
        dmatrix = []
        for i in range(n-1):
            dlist = [i]
            for j in range(k-1):
                dlist.append(np.abs(distances[i] - ideal_step * j))
            dmatrix.append(dlist)
        dmatrix = np.array(dmatrix)

        for j in range(1, dmatrix.shape[1]):
            idx = np.argmin(dmatrix[:, j])
            selected_indices.append(int(dmatrix[idx][0]))
            dmatrix = np.delete(dmatrix, idx, axis=0)
        selected_indices.append(n - 1)

    selected_solutions = solutions[selected_indices]
    selected_ruleset_coverage = [ruleset_coverage[i] for i in selected_indices]
    return selected_solutions, selected_ruleset_coverage


def _equi_spaced_ss(solutions: np.array,
                    ruleset_coverage: list,
                    k=10,
                    **kwargs):
    def manhattan_dist(idx1, idx2):
        return np.abs(solutions[idx1][-2] - solutions[idx2][-2]) \
                + np.abs(solutions[idx1][-1] - solutions[idx2][-1])

    solutions, rank = sort_solutions(solutions)
    ruleset_coverage = [ruleset_coverage[i] for i in rank]
    n = solutions.shape[0]
    selected_indices = []
    if k == 1:
        selected_indices.append(n // 2)
    elif k == 2:
        selected_indices += [0, n-1]
    else:
        total_distance = manhattan_dist(0, -1)
        ideal_step = total_distance / (k - 1)
        dmatrix = []
        for i in range(n-1):
            d = manhattan_dist(0, i)
            dlist = [i]
            for j in range(k-1):
                dlist.append(np.abs(d - ideal_step * j))
            dmatrix.append(dlist)
        dmatrix = np.array(dmatrix)

        for j in range(1, dmatrix.shape[1]):
            idx = np.argmin(dmatrix[:, j])
            selected_indices.append(int(dmatrix[idx][0]))
            dmatrix = np.delete(dmatrix, idx, axis=0)
        selected_indices.append(n-1)
    selected_solutions = solutions[selected_indices]
    selected_ruleset_coverage = [ruleset_coverage[i] for i in selected_indices]
    return selected_solutions, selected_ruleset_coverage


def _equi_jaccard_ss(solutions: np.array,
                     ruleset_coverage: list,
                     k=10,
                     **kwargs):
    selected_ruleset_coverage = []
    dmatrix = []
    for i in range(solutions.shape[0]):
        dlist = []
        this_coverage = ruleset_coverage[i][0]
        for j in range(solutions.shape[0]):
            that_coverage = ruleset_coverage[j][0]
            inter = this_coverage.intersection_cardinality(that_coverage)
            union = len(this_coverage | that_coverage)
            jaccard_distance = 1 - (inter / union)
            dlist.append(jaccard_distance)
        dmatrix.append(dlist)
    dmatrix = np.array(dmatrix)
    path = solve_tsp(dmatrix)

    selected_index = [path[0]]
    if k > 1:
        step_size = (len(path) - 1) // (k-1)
        for i in range(k - 1):
            selected_index.append(path[(i+1) * step_size])
    for i in selected_index:
        selected_ruleset_coverage.append(ruleset_coverage[i])
    selected_solutions = solutions[selected_index]
    return selected_solutions, selected_ruleset_coverage


SSF_METHODS = {
    'hv-ss': _hv_ss,
    'igd-ss': _igd_ss,
    'igd+-ss': _igd_ss,
    'k-medoids-pr': _k_medoids_ss,
    'k-medoids-jaccard': _k_medoids_ss,
    'hvc-ss': _hvc_ss,
    'equi-dist': _equi_dist_ss,
    'equi-spaced': _equi_spaced_ss,
    'equi-jaccard': _equi_jaccard_ss
}


def ssf(solutions: np.array,
        ruleset_coverage: list,
        method: str = 'hvc-ss',
        k=10,
        **kwargs):
    if method not in SSF_METHODS.keys():
        raise ValueError("Unknown method %s. "
                         "Valid methods are %s " % (method, list(SSF_METHODS.keys())))
    assert k > 0
    n = solutions.shape[0]
    if n <= k:
        return solutions, ruleset_coverage
    ssf_method = SSF_METHODS[method]
    return ssf_method(solutions, ruleset_coverage, k, **kwargs)
