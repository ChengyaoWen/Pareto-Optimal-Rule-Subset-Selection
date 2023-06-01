import numpy as np
from pyroaring import BitMap
import logging

from utils.moo_utils import gen_next_solutions
from utils.rule_utils import eval_ruleset_coverage


logger = logging.getLogger(__name__)


def select_solutions(solutions, coverages, k, constraint):
    selected_indices = []
    for i, s in enumerate(solutions):
        if s[-2] >= constraint:
            selected_indices.append(i)
    selected_solutions = solutions[selected_indices]
    selected_coverages = [coverages[i] for i in selected_indices]
    if len(selected_indices) > k:
        selected_indices = np.argsort(selected_solutions[:, -1], kind='stable')[-k:]
        selected_solutions = selected_solutions[selected_indices]
        selected_coverages = [coverages[i] for i in selected_indices]
        # print("constraint={}  {}".format(constraint, selected_solutions[0, -2]))
    return selected_solutions, selected_coverages


def find_opt_solution(solutions: np.array,
                      coverages,
                      valid_pos_indices):
    assert solutions.shape[0] == len(coverages)
    if solutions.shape[0] == 0:
        return [], 0
    valid_metrics = [eval_ruleset_coverage(c[1], valid_pos_indices) for c in coverages]
    recalls = [m[1] for m in valid_metrics]
    opt_index = np.argmax(recalls)
    opt_recall = recalls[opt_index]
    opt_solution = solutions[opt_index]
    return opt_solution, opt_recall


def greedy_opt(solutions: np.array,
               rule_coverage_list: list,
               train_pos_indices: BitMap,
               valid_pos_indices: BitMap,
               beam_width: int=10,
               constraint: float=0.3,
               max_round=50,
               n_process=4):
    input_size = solutions.shape[0]
    assert input_size == len(rule_coverage_list)
    logger.info("Start CRSL Greedy Opt. Input size:{}  constraint: {}".format(input_size, constraint))
    coverages = np.copy(rule_coverage_list)
    n = 1
    opt_recall = 0
    opt_solution = []

    while n < max_round:
        solutions, coverages = select_solutions(solutions, coverages, beam_width, constraint)
        if solutions.shape[0] == 0:
            logger.info("Finish! CRSL Converges")
            break
        curr_solution, curr_recall = find_opt_solution(solutions, coverages, valid_pos_indices)
        logger.debug("Round={} opt _recall={}".format(n, curr_recall))
        if curr_recall > opt_recall:
            opt_solution = curr_solution
            opt_recall = curr_recall
        solutions, coverages = gen_next_solutions(solutions, rule_coverage_list,
                                                  coverages, train_pos_indices, n_process)
        n += 1

    logger.info("Finish! CRSL reaches {} rounds".format(max_round))
    return opt_solution
