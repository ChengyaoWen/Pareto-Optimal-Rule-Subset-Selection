import numpy as np
from pyroaring import BitMap
import logging
import time

from utils.moo_utils import gen_next_solutions
from utils.rule_utils import eval_ruleset_coverage


logger = logging.getLogger(__name__)


def compute_fscore(p, r, fbeta):
    assert fbeta > 0
    if p <= 0 or r <= 0:
        return 0
    fbeta2 = np.square(fbeta)
    fscore = (1 + fbeta2) * p * r / (fbeta2 * p + r)
    return fscore


def select_solutions(solutions, coverages, k, fbeta):
    n = solutions.shape[0]
    selected_solutions = []
    selected_coverages = []
    if n > k:
        f_scores = [compute_fscore(s[-2], s[-1], fbeta) for s in solutions]
        selected_indices = np.argsort(f_scores, kind='stable')[-k:]
        selected_solutions = solutions[selected_indices]
        selected_coverages = [coverages[i] for i in selected_indices]
    return selected_solutions, selected_coverages


def find_opt_solution(solutions: np.array,
                      coverages,
                      valid_pos_indices,
                      fbeta):
    valid_metrics = [eval_ruleset_coverage(c[1], valid_pos_indices) for c in coverages]
    f_scores = [compute_fscore(m[0], m[1], fbeta) for m in valid_metrics]
    opt_index = np.argmax(f_scores)
    opt_fscore = f_scores[opt_index]
    opt_solution = solutions[opt_index]
    return opt_solution, opt_fscore


def beam_search(solutions: np.array,
                rule_coverage_list: list,
                train_pos_indices: BitMap,
                valid_pos_indices: BitMap,
                beam_width: int=10,
                fbeta: float=0.1,
                max_round=50,
                n_process=4,
                **kwargs):

    eval_step = int(kwargs.get("eval_step", 0))
    input_size = solutions.shape[0]
    assert input_size == len(rule_coverage_list)
    logger.info("Start Greedy BeamSearch. Input size:{}  fbeta: {}".format(input_size, fbeta))

    coverages = np.copy(rule_coverage_list)
    n = 1
    opt_fscore = 0
    opt_solution = []
    evaluations = []
    time_s = time.time()
    while n <= max_round:
        solutions, coverages = select_solutions(solutions, coverages, beam_width, fbeta)
        curr_solution, curr_fscore = find_opt_solution(solutions, coverages, valid_pos_indices, fbeta)
        logger.debug("Round={} opt fscore={}".format(n, curr_fscore))
        if curr_fscore > opt_fscore:
            opt_solution = curr_solution
            opt_fscore = curr_fscore
        solutions, coverages = gen_next_solutions(solutions, rule_coverage_list,
                                                  coverages, train_pos_indices, n_process)
        if eval_step > 0 and (n - 1) % eval_step == 0:
            evaluations.append([curr_fscore, time.time() - time_s])
        n += 1
        if solutions.shape[0] == 0:
            logger.info("Finish! Greedy BeamSearch Converges")
            break

    logger.info("Finish! Greedy BeamSearch reaches {} rounds".format(max_round))
    return opt_solution, np.array(evaluations)
