import sys
import pytest
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

sys.path.insert(0, '../../')
import pors.utils.moo_utils as mu
from pors.utils.data_utils import rule_predict, data_preprocess
from pors.pors import init_solutions

RULE_DF = pd.read_csv("../test_data/test_rules.csv")
DATA_DF = pd.read_csv("../test_data/test_data.csv")
COVERAGE_DF = pd.read_csv("../test_data/test_coverage.csv")

assertions = unittest.TestCase('__init__')


class TestDataUtils():
    def test_rule_predict(self):
        test_coverage = rule_predict(RULE_DF, DATA_DF)
        assert_frame_equal(test_coverage, COVERAGE_DF)


class TestMooUtils():
    def setup_class(self):
        self._mock_solutions = np.array([
            [1.0, 0.2],
            [0.9, 0.1],
            [0.2, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 1.0]
        ])
        self._mock_paretos = np.array([
            [1.0, 0.2],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 1.0]
        ])
        self._rule_id_list, self._rule_coverage_map, self._positive_indices, _ \
            = data_preprocess(DATA_DF, COVERAGE_DF)
        self._test_solutions, _ = init_solutions(self._rule_id_list, self._rule_coverage_map,
                                                 self._positive_indices, self._rule_coverage_map)

    def test_pareto_solutions(self):
        test_pareto_solutions, _ = mu.pareto_solutions(self._mock_solutions)
        assert np.array_equal(test_pareto_solutions, self._mock_paretos)
        pareto_solutions, _ = mu.pareto_solutions(self._test_solutions)
        ruleset_coverage = [v for k, v in self._rule_coverage_map.items()]
        test_pareto_solutions, _ = mu.pareto_solutions(self._test_solutions,
                                                       ruleset_coverage=ruleset_coverage,
                                                       positive_indices=self._positive_indices)
        assert np.array_equal(pareto_solutions, test_pareto_solutions)

    def test_compute_hv2d(self):
        test_hv = mu.compute_hv2d(self._mock_paretos)
        assertions.assertAlmostEqual(test_hv, 0.47)
        test_pareto_solutions, _ = mu.pareto_solutions(self._test_solutions)
        test_hv = mu.compute_hv2d(test_pareto_solutions)
        assertions.assertAlmostEqual(test_hv, 0.3095075775294055)

    def test_compute_hvc2d(self):
        mock_new_solutions = np.array([[1.0, 0.4], [0.8, 1.0], [0.5, 0.8], [0.4, 0.6]])
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[0], self._mock_paretos), 0.08)
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[1], self._mock_paretos), 0.37)
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[2], self._mock_paretos), 0.09)
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[3], self._mock_paretos), 0.02)
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[0:2], self._mock_paretos), 0.41)
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[1:3], self._mock_paretos), 0.37)
        assertions.assertAlmostEqual(mu.compute_hvc2d(mock_new_solutions[0:3], self._mock_paretos), 0.41)
