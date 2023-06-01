import numpy as np
import copy
from pyroaring import BitMap
from typing import Optional
import logging
import time

from utils.rule_utils import eval_ruleset_coverage
from utils.moo_utils import pareto_solutions, compute_hv2d, gen_next_solutions, eval_solutions
from ssf import ssf

logger = logging.getLogger(__name__)


def init_solutions(rule_id_list: list,
                   train_rule_coverage: dict,
                   train_positive_indices: BitMap,
                   valid_rule_coverage: dict):
    """
    initialize solutions based on rule list and rule coverage
    """
    assert len(rule_id_list) > 0
    solutions = np.identity(len(rule_id_list))
    objectives = []
    rule_coverage_list = []
    for rule_id in rule_id_list:
        rule_coverage = train_rule_coverage[rule_id]
        objective = eval_ruleset_coverage(rule_coverage, train_positive_indices)
        objectives.append(objective)
        rule_coverage_list.append((rule_coverage, valid_rule_coverage[rule_id] if
                                   rule_id in valid_rule_coverage.keys() else BitMap()))
    solutions = np.hstack([solutions, np.array(objectives)])
    return solutions, rule_coverage_list


def pors(solutions: np.array,
         rule_coverage_list: list,
         train_pos_indices: BitMap,
         valid_pos_indices: BitMap,
         dataset_size: int,
         ssf_method='hvc-ss',
         ssf_param: Optional[dict]=None,
         k=10,
         max_round=50,
         n_process=4,
         **kwargs):
    """
    run PORS algorithm and return optimal pareto solutions
    """
    eval_step = int(kwargs.get("eval_step", 0))
    test_rule_coverage = kwargs.get("test_rule_coverage")
    test_pos_indices = kwargs.get("test_pos_indices")
    rule_id_list = kwargs.get("rule_id_list")
    evaluations = []

    time_s = time.time()
    input_size = solutions.shape[0]
    assert input_size == len(rule_coverage_list)
    logger.info("Start PORS. Input size:{}  ssf_method:{}".format(input_size, ssf_method))

    n = 1
    coverages = copy.deepcopy(rule_coverage_list)
    solutions, mask = pareto_solutions(solutions)
    coverages = [coverages[i] for i in range(len(mask)) if mask[i]]
    prev_solutions = solutions
    prev_coverages = coverages

    train_hv = compute_hv2d(solutions)
    valid_solutions, _ = pareto_solutions(solutions,
                                          ruleset_coverage=[c[1] for c in coverages],
                                          positive_indices=valid_pos_indices)
    valid_hv = compute_hv2d(valid_solutions)
    if eval_step > 0 and (n - 1) % eval_step == 0:
        if test_pos_indices and test_rule_coverage and rule_id_list:
            test_hv = eval_solutions(solutions, rule_id_list, test_rule_coverage, test_pos_indices)
        else:
            test_hv = valid_hv
        evaluations.append([train_hv, valid_hv, test_hv, time.time() - time_s])

    opt_hv = valid_hv
    opt_solutions = solutions
    opt_coverages = coverages
    if not ssf_param:
        ssf_param = dict()

    logger.debug("Round={} train_hv={} valid_hv={}".format(n, train_hv, valid_hv))
    solutions, coverages = ssf(solutions, coverages, method=ssf_method, k=k, dataset_size=dataset_size, **ssf_param)
    while n < max_round:
        n += 1
        next_solutions, next_coverages = gen_next_solutions(solutions, rule_coverage_list,
                                                            coverages, train_pos_indices, n_process)
        if len(next_coverages) == 0:
            logger.info("Finish! PORS Converges")
            break
        merge_solutions = np.vstack([prev_solutions, next_solutions])
        merge_coverages = prev_coverages + next_coverages
        curr_solutions, merge_pareto_mask = pareto_solutions(merge_solutions)
        curr_coverages = [merge_coverages[i] for i in range(len(merge_pareto_mask)) if merge_pareto_mask[i]]
        new_pareto_mask = merge_pareto_mask[prev_solutions.shape[0]:]
        if len(new_pareto_mask) == 0:
            logger.info("Finish! PORS Converges")
            break

        if ssf_method == 'hvc-ss':
            new_solutions = next_solutions[new_pareto_mask]
            new_coverages = [next_coverages[i] for i in range(len(new_pareto_mask)) if new_pareto_mask[i]]
            solutions, coverages = ssf(new_solutions, new_coverages,
                                       method=ssf_method,
                                       k=k,
                                       prev_solutions=prev_solutions,
                                       dataset_size=dataset_size,
                                       **ssf_param)
        else:
            solutions, coverages = ssf(curr_solutions, curr_coverages,
                                       method=ssf_method,
                                       k=k,
                                       prev_solutions=prev_solutions,
                                       dataset_size=dataset_size,
                                       **ssf_param)

        prev_solutions = curr_solutions
        prev_coverages = curr_coverages
        train_hv = compute_hv2d(curr_solutions)
        valid_solutions, _ = pareto_solutions(curr_solutions,
                                              ruleset_coverage=[c[1] for c in curr_coverages],
                                              positive_indices=valid_pos_indices)
        valid_hv = compute_hv2d(valid_solutions)
        if eval_step > 0 and (n-1) % eval_step == 0:
            if test_pos_indices and test_rule_coverage and rule_id_list:
                test_hv = eval_solutions(opt_solutions, rule_id_list, test_rule_coverage, test_pos_indices)
            else:
                test_hv = valid_hv
            evaluations.append([train_hv, valid_hv, test_hv, time.time() - time_s])

        if valid_hv > opt_hv:
            opt_hv = valid_hv
            opt_solutions = curr_solutions
            opt_coverages = curr_coverages
        logger.debug("Round={} train_hv={} valid_hv={}".format(n, train_hv, valid_hv))
    logger.info("Finish! PORS reaches {} rounds ".format(max_round))
    return opt_solutions, np.array(evaluations)
