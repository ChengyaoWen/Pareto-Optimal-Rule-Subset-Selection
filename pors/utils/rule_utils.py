from pyroaring import BitMap
from functools import reduce


def eval_ruleset_coverage(ruleset_coverage: BitMap,
                          positive_indices: BitMap):
    """
    compute ruleset's precision / recall based on its coverage
    """
    covered_pos_cnt = ruleset_coverage.intersection_cardinality(positive_indices)
    total_pos_cnt = len(positive_indices)
    covered_cnt = len(ruleset_coverage)
    precision = 0 if covered_cnt == 0 else covered_pos_cnt / covered_cnt
    if total_pos_cnt == 0:
        recall = 0
    else:
        recall = covered_pos_cnt / total_pos_cnt
    return [precision, recall]


def eval_ruleset(ruleset: list,
                 rule_coverage_map: dict,
                 positive_indices: BitMap):
    """
    compute ruleset's precision / recall
    """
    precision = 0
    recall = 0
    single_rule_coverage = []
    for r in ruleset:
        if rule_coverage_map.get(r):
            single_rule_coverage.append(rule_coverage_map[r])
        else:
            single_rule_coverage.append(BitMap())
    ruleset_coverage = reduce(lambda bm1, bm2: bm1 | bm2, single_rule_coverage, BitMap())
    if len(ruleset_coverage) == 0 or len(positive_indices) == 0:
        return [precision, recall]

    covered_positives = ruleset_coverage.intersection_cardinality(positive_indices)
    precision = covered_positives / len(ruleset_coverage)
    recall = covered_positives / len(positive_indices)
    return [precision, recall]
