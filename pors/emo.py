import random
import logging
import time

from utils.moo_utils import *
from utils.rule_utils import eval_ruleset


logger = logging.getLogger(__name__)


def breeding(population: np.array,
             rank,
             crowding_distance,
             objectives=2):
    offspring = np.copy(population)
    for i in range(0, offspring.shape[0]):
        i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
        if crowded_operator(rank, crowding_distance, individual_1=i1, individual_2=i2):
            parent_1 = i1
        elif crowded_operator(rank, crowding_distance, individual_1=i2, individual_2=i1):
            parent_1 = i2
        else:
            rand = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            if rand > 0.5:
                parent_1 = i1
            else:
                parent_1 = i2
        if crowded_operator(rank, crowding_distance, individual_1=i3, individual_2=i4):
            parent_2 = i3
        elif crowded_operator(rank, crowding_distance, individual_1=i4, individual_2=i3):
            parent_2 = i4
        else:
            rand = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            if rand > 0.5:
                parent_2 = i3
            else:
                parent_2 = i4

        for j in range(0, offspring.shape[1] - objectives):
            rand = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            if rand >= 0.5:
                offspring[i, j] = population[parent_1, j]
            else:
                offspring[i, j] = population[parent_2, j]
    return offspring


def mutation(population: np.array,
             rule_id_list,
             rule_coverage: dict,
             positive_indices: BitMap,
             mutation_rate=0.02,
             objectives=2):

    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[1] - objectives):
            rate = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            if rate < mutation_rate:
                population[i, j] = 0 if population[i, j] == 1 else 1
        ruleset = [rule_id_list[idx] for idx in range(len(population[i][:-objectives])) if population[i][idx] == 1]
        prec, recall = eval_ruleset(ruleset, rule_coverage, positive_indices)
        population[i, -2] = prec
        population[i, -1] = recall
    return population


def gen_offspring_helper(population,
                         args):
    (rule_id_list, rule_coverage, positive_indices, mutation_rate, objectives) = args
    offspring = mutation(population, rule_id_list, rule_coverage,
                         positive_indices, mutation_rate, objectives)
    return offspring


def gen_offspring(population,
                  rule_id_list,
                  rule_coverage,
                  positive_indices,
                  mutation_rate=0.02,
                  objectives=2,
                  n_process=4):

    sub_populations = [[] for _ in range(n_process)]
    for i in range(population.shape[0]):
        sub_populations[i % n_process].append(i)
    args = (rule_id_list, rule_coverage, positive_indices, mutation_rate, objectives)
    sub_offsprings = Parallel(n_jobs=n_process, backend='multiprocessing')(delayed(gen_offspring_helper)(
        population[indices], args) for indices in sub_populations)
    offspring = reduce(lambda p1, p2: np.vstack([p1, p2]), sub_offsprings)
    return offspring


def nsga2(solutions: np.array,
          rule_id_list: list,
          train_rule_coverage: dict,
          valid_rule_coverage: dict,
          train_pos_indices: BitMap,
          valid_pos_indices: BitMap,
          generations=1000,
          mutation_rate=0.02,
          n_process=4,
          **kwargs):

    eval_step = int(kwargs.get("eval_step", 0))
    test_rule_coverage = kwargs.get("test_rule_coverage")
    test_pos_indices = kwargs.get("test_pos_indices")
    evaluations = []

    n = solutions.shape[0]
    logger.info("Start NSGA-II. Input size:{}".format(n))
    populations = copy.deepcopy(solutions)
    offspring = np.copy(solutions)
    archive_solutions = np.empty(shape=(0, offspring.shape[1]))
    count = 0
    time_s = time.time()

    opt_hv = 0
    opt_solutions = []

    while count < generations:
        populations = np.vstack([populations, offspring])
        archive_solutions = np.vstack([archive_solutions, populations])
        archive_solutions, _ = pareto_solutions(archive_solutions)
        train_hv = compute_hv2d(archive_solutions)
        count += 1
        populations, rank = fast_non_dominated_sorting(populations, n)
        crowding_distance = crowding_distance_function(populations)
        offspring = breeding(populations, rank, crowding_distance)
        offspring = gen_offspring(offspring, rule_id_list, train_rule_coverage, train_pos_indices,
                                  mutation_rate, n_process=n_process)
        valid_hv = eval_solutions(archive_solutions, rule_id_list, valid_rule_coverage, valid_pos_indices)
        if valid_hv > opt_hv:
            opt_solutions = archive_solutions
            opt_hv = valid_hv

        if eval_step > 0 and (count - 1) % eval_step == 0:
            if test_pos_indices and test_rule_coverage and rule_id_list:
                test_hv = eval_solutions(archive_solutions, rule_id_list, test_rule_coverage, test_pos_indices)
            else:
                test_hv = valid_hv
            evaluations.append([train_hv, valid_hv, test_hv, time.time() - time_s])

        if count == 1 or count % 100 == 0:
            logger.debug("Round={} train_hv={}".format(count, train_hv))
    logger.info("Finish! NSGA-II reaches {} rounds ".format(generations))
    return opt_solutions, np.array(evaluations)
