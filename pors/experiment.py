import os
import logging
import pandas as pd
import numpy as np

from emo import nsga2
from greedy import beam_search, compute_fscore
from pors import init_solutions, pors
from crsl import greedy_opt
from utils.data_utils import data_preprocess, rule_predict
from utils.moo_utils import eval_solutions, eval_single_solution
from utils.rule_utils import eval_ruleset_coverage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MY_PATH = os.path.dirname(os.path.abspath(__file__))


METHODS = ['hvc-ss', 'hv-ss', 'igd-ss', 'igd+-ss', 'k-medoids-pr', 'k-medoids-jaccard',
           'equi-jaccard', 'equi-spaced', 'equi-dist', 'nsga2']


def metric2str(metric_list, digit=4):
    mean = format(np.mean(metric_list), '.{}f'.format(digit))
    std = format(np.std(metric_list), '.{}f'.format(digit))
    return "{}$\pm${}".format(mean, std)


def ssf_comparison(stage1: str="SpectralRules",
                 max_input_size=500,
                 output_csv="ssf_comparison.csv"):
    """
    run ssf_comparison to reproduce results of table 3&4 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/ssf_comparison")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    if stage1 == 'SpectralRules':
        rule_csv_name = "../datasets/{}/rules/sr_on_train_{}.csv"
    elif stage1 == 'TreeEns':
        rule_csv_name = "../datasets/{}/rules/te_on_train_{}.csv"
    else:
        raise ValueError("Unknown Stage1 method %s. "
                         "Valid methods are %s " % (stage1, ["SpectralRules", "TreeEns"]))

    ssf_params = {
        'igd-ss': {'metric': 'igd'},
        'igd+-ss': {'metric': 'igd+'},
        'k-medoids-pr': {'metric': 'euclidean'},
        'k-medoids-jaccard': {'metric': 'jaccard'}
    }

    k = 10
    max_round = 50
    n_process = 4

    datasets = ['bank', 'credit', 'default', 'fraud']
    ret_df = []
    for dataset in datasets:
        logger.info("Run ssf_comparison on {} dataset".format(dataset))

        exp_result = dict()
        for method in METHODS:
            exp_result[method] = list()

        for i in [1, 2, 3, 4, 5]:
            train_csv = data_csv_name.format(dataset, "train", i)
            valid_csv = data_csv_name.format(dataset, "valid", i)
            test_csv = data_csv_name.format(dataset, "test", i)
            rule_csv = rule_csv_name.format(dataset, i)

            train_df = pd.read_csv(train_csv)
            valid_df = pd.read_csv(valid_csv)
            test_df = pd.read_csv(test_csv)
            rule_df = pd.read_csv(rule_csv)
            logger.info("Begin Trial {}".format(i))

            train_pred_df = rule_predict(rule_df, train_df)
            valid_pred_df = rule_predict(rule_df, valid_df)
            test_pred_df = rule_predict(rule_df, test_df)
            rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df, train_pred_df)
            _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
            _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)

            if stage1 == "SpectralRules":
                rule_id_list = [rule_id for rule_id in rule_id_list if
                                int(rule_id.split("_")[1]) < max_input_size // 10]
            else:
                rule_id_list = rule_id_list[:max_input_size]
            solutions, rule_coverage_list = init_solutions(rule_id_list, train_rule_coverage,
                                                           train_pos_indices, valid_rule_coverage)

            for method in METHODS:
                if method != 'nsga2':
                    ssf_param = ssf_params.get(method, dict())
                    opt_solutions, _ = pors(solutions,
                                            rule_coverage_list,
                                            train_pos_indices,
                                            valid_pos_indices,
                                            dataset_size=trainset_size,
                                            ssf_method=method,
                                            ssf_param=ssf_param,
                                            k=k,
                                            max_round=max_round,
                                            n_process=n_process)
                else:
                    opt_solutions, _ = nsga2(solutions, rule_id_list, train_rule_coverage, valid_rule_coverage,
                                          train_pos_indices, valid_pos_indices, n_process=n_process)
                opt_hv = eval_solutions(opt_solutions, rule_id_list, test_rule_coverage, test_pos_indices)
                logger.info("test_hv of {}: {}".format(method,  opt_hv))
                exp_result[method].append(opt_hv)

        ret_row = [dataset]
        logging.info("Finish ssf_comparison on {} dataset".format(dataset))
        for method in METHODS:
            hvs = exp_result[method]
            logger.info("{}: hv_mean={} hv_std={}".format(method, np.round(np.mean(hvs), 4), np.round(np.std(hvs), 4)))
            ret_row.append(metric2str(hvs, 3))
        ret_df.append(ret_row)
    for d in ['A1', 'A2', 'A3']:
        row = [d] + [" 0$\pm$0 " for _ in METHODS]
        ret_df.append(row)
    ret_columns = ['DataSet'] + METHODS
    ret_df = pd.DataFrame(np.array(ret_df), columns=ret_columns)
    ret_df.to_csv(output_csv, index=False)

    return


def stage1_comparison(dataset='bank',
                      output_csv="figure_2.csv"):
    """
    run stage1_comparison to reproduce figure 2 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/stage1_evaluation")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    rule_csv_name = "../datasets/{}/rules/{}_on_train_{}.csv"

    k = 10
    max_round = 50
    n_process = 4

    input_size_list = [100, 200, 500, 1000, 2000]
    ret_df = []

    for stage1 in ['sr', 'te']:
        ret_row = [stage1]
        hv_list = [[] for _ in input_size_list]
        for i in [1, 2, 3, 4, 5]:
            train_csv = data_csv_name.format(dataset, "train", i)
            valid_csv = data_csv_name.format(dataset, "valid", i)
            test_csv = data_csv_name.format(dataset, "test", i)
            rule_csv = rule_csv_name.format(dataset, stage1, i)

            train_df = pd.read_csv(train_csv)
            valid_df = pd.read_csv(valid_csv)
            test_df = pd.read_csv(test_csv)
            rule_df = pd.read_csv(rule_csv)
            logger.info("Begin Trial {}".format(i))

            train_pred_df = rule_predict(rule_df, train_df)
            valid_pred_df = rule_predict(rule_df, valid_df)
            test_pred_df = rule_predict(rule_df, test_df)
            rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df, train_pred_df)
            _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
            _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)

            for input_size_idx, input_size in enumerate(input_size_list):
                if stage1 == "sr":
                    if input_size > 500:
                        hv_list[input_size_idx].append(0)
                        continue
                    sub_rule_id_list = [rule_id for rule_id in rule_id_list if
                                        int(rule_id.split("_")[1]) < input_size // 10]
                else:
                    sub_rule_id_list = rule_id_list[:input_size]
                solutions, rule_coverage_list = init_solutions(sub_rule_id_list, train_rule_coverage,
                                                               train_pos_indices, valid_rule_coverage)

                opt_solutions, _ = pors(solutions,
                                        rule_coverage_list,
                                        train_pos_indices,
                                        valid_pos_indices,
                                        dataset_size=trainset_size,
                                        ssf_method='hvc-ss',
                                        ssf_param=dict(),
                                        k=k,
                                        max_round=max_round,
                                        n_process=n_process)
                opt_hv = eval_solutions(opt_solutions, sub_rule_id_list, test_rule_coverage, test_pos_indices)
                logger.info("test_hv:{}".format(opt_hv))
                hv_list[input_size_idx].append(opt_hv)

        for idx, hvs in enumerate(hv_list):
            logger.info("stage1={}, input_size={}, hv_mean={}, hv_std={}".format(
                stage1, input_size_list[idx], np.mean(hvs), np.std(hvs)))
            ret_row.append(metric2str(hvs))
        ret_df.append(ret_row)

    ret_columns = ['Stage-1'] + [str(v) for v in input_size_list]
    ret_df = pd.DataFrame(np.array(ret_df), columns=ret_columns)
    ret_df.to_csv(output_csv, index=False)

    return


def case_study_fscore(fbetas: list=[0.1, 0.2, 0.5],
                     output_csv="fscore.csv"):
    """
    run case_study_fscore to reproduce results of table 6 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/case_fbeta")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    rule_csv_name = "../datasets/{}/rules/sr_on_train_{}.csv"
    methods = ['greedy', 'hvc-ss', 'equi-spaced', 'k-medoids-jaccard']
    ssf_params = {
        'k-medoids-jaccard': {'metric': 'jaccard'}
    }
    k = 10
    max_round = 50
    n_process = 4
    datasets = ['default', 'credit', 'fraud', 'bank']

    ret_df = []
    for dataset in datasets:
        logger.info("Run case_study_fscore on {} dataset".format(dataset))
        fscores = dict()
        for method in methods:
            fscores[method] = [[] for _ in fbetas]

        for i in [1, 2, 3, 4, 5]:
            train_csv = data_csv_name.format(dataset, "train", i)
            valid_csv = data_csv_name.format(dataset, "valid", i)
            test_csv = data_csv_name.format(dataset, "test", i)
            rule_csv = rule_csv_name.format(dataset, i)

            train_df = pd.read_csv(train_csv)
            valid_df = pd.read_csv(valid_csv)
            test_df = pd.read_csv(test_csv)
            rule_df = pd.read_csv(rule_csv)
            logger.info("Begin Trial {}".format(i))

            train_pred_df = rule_predict(rule_df, train_df)
            valid_pred_df = rule_predict(rule_df, valid_df)
            test_pred_df = rule_predict(rule_df, test_df)
            rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df, train_pred_df)
            _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
            _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)
            solutions, rule_coverage_list = init_solutions(rule_id_list, train_rule_coverage,
                                                           train_pos_indices, valid_rule_coverage)
            for method in methods:
                if method != 'greedy':
                    ssf_param = ssf_params.get(method, dict())
                    opt_solutions, _ = pors(solutions,
                                            rule_coverage_list,
                                            train_pos_indices,
                                            valid_pos_indices,
                                            dataset_size=trainset_size,
                                            ssf_method=method,
                                            ssf_param=ssf_param,
                                            k=k,
                                            max_round=max_round,
                                            n_process=n_process)
                    scores = [[] for _ in fbetas]
                    for s in opt_solutions:
                        prec, recall = eval_single_solution(s, rule_id_list, test_rule_coverage, test_pos_indices)
                        for idx, beta in enumerate(fbetas):
                            score = compute_fscore(prec, recall, beta)
                            scores[idx].append(score)
                    log_info = "test"
                    for idx in range(len(fbetas)):
                        max_score = np.max(scores[idx])
                        fscores[method][idx].append(max_score)
                        log_info += " f{}_score={}".format(fbetas[idx], np.round(max_score, 4))
                    logger.info(log_info)
                else:
                    log_info = "test"
                    for idx, fbeta in enumerate(fbetas):
                        opt_solition, _ = beam_search(solutions, rule_coverage_list, train_pos_indices, valid_pos_indices,
                                                      k, fbeta, max_round, n_process)
                        prec, recall = eval_single_solution(opt_solition, rule_id_list, test_rule_coverage, test_pos_indices)
                        fscore = compute_fscore(prec, recall, fbeta)
                        fscores[method][idx].append(fscore)
                        log_info += " f{}_score={}".format(fbeta, np.round(fscore, 4))
                    logger.info(log_info)

        logging.info("Finish case_study_fscore on {} dataset".format(dataset))

        for idx, fbeta in enumerate(fbetas):
            ret_row = [dataset, str(fbeta)]
            for method in methods:
                scores = fscores[method][idx]
                mean = np.round(np.mean(scores), 4)
                std = np.round(np.std(scores), 4)
                logger.info("{} fbeta={} fscore_means={} fscore_stds={}".format(method, fbeta, mean, std))
                ret_row.append(metric2str(scores))
            ret_df.append(ret_row)

    for d in ['A1', 'A2', 'A3']:
        for fbeta in fbetas:
            row = [d, fbeta] + [" 0$\pm$0 " for _ in methods]
            ret_df.append(row)
    ret_columns = ['DataSet', 'fbeta'] + methods
    ret_df = pd.DataFrame(np.array(ret_df), columns=ret_columns)
    ret_df.to_csv(output_csv, index=False)
    return


def case_study_crsl(prec_thresholds: list=[0.3, 0.5, 0.7],
                    output_csv="table_5.csv"):
    """
    run case_study_crsl to reproduce results of table 5 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/case_crsl")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    methods = ['crsl', 'hvc-ss', 'equi-spaced', 'k-medoids-jaccard']
    ssf_params = {
        'k-medoids-jaccard': {'metric': 'jaccard'}
    }

    k = 10
    max_round = 50
    n_process = 4
    datasets = ['default', 'credit', 'fraud', 'bank']

    ret_df = []
    for dataset in datasets:
        logger.info("Run case_study_crsl on {} dataset".format(dataset))
        recalls = dict()
        for method in methods:
            recalls[method] = [[] for _ in prec_thresholds]

        for i in [1, 2, 3, 4, 5]:
            logger.info("Begin Trial {}".format(i))
            train_csv = data_csv_name.format(dataset, "train", i)
            valid_csv = data_csv_name.format(dataset, "valid", i)
            test_csv = data_csv_name.format(dataset, "test", i)
            train_df = pd.read_csv(train_csv)
            valid_df = pd.read_csv(valid_csv)
            test_df = pd.read_csv(test_csv)

            #SpectralRules
            rule_csv_name = "../datasets/{}/rules/sr_on_train_{}.csv"
            rule_csv = rule_csv_name.format(dataset, i)
            rule_df = pd.read_csv(rule_csv)

            train_pred_df = rule_predict(rule_df, train_df)
            valid_pred_df = rule_predict(rule_df, valid_df)
            test_pred_df = rule_predict(rule_df, test_df)
            rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df, train_pred_df)
            _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
            _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)
            solutions, rule_coverage_list = init_solutions(rule_id_list, train_rule_coverage,
                                                           train_pos_indices, valid_rule_coverage)
            for method in methods:
                if method != 'crsl':
                    ssf_param = ssf_params.get(method, dict())
                    opt_solutions, _ = pors(solutions,
                                            rule_coverage_list,
                                            train_pos_indices,
                                            valid_pos_indices,
                                            dataset_size=trainset_size,
                                            ssf_method=method,
                                            ssf_param=ssf_param,
                                            k=k,
                                            max_round=max_round,
                                            n_process=n_process)
                    recs = [[] for _ in prec_thresholds]
                    for s in opt_solutions:
                        prec, recall = eval_single_solution(s, rule_id_list, valid_rule_coverage, valid_pos_indices)
                        for idx, pt in enumerate(prec_thresholds):
                            recs[idx].append(recall if prec >= pt else -1)
                    log_info = "test"
                    for idx, rec in enumerate(recs):
                        opt_index = np.argmax(rec)
                        if rec[opt_index] < 0:
                            opt_recall = 0
                        else:
                            _, opt_recall = eval_single_solution(opt_solutions[opt_index], rule_id_list, test_rule_coverage, test_pos_indices)
                        recalls[method][idx].append(opt_recall)
                        log_info += " cons{}_recall={}".format(prec_thresholds[idx], np.round(opt_recall, 4))
                    logger.info(log_info)

            #CRSL
            log_info = "test"
            rule_csv_name = "../datasets/{}/rules/crsl_on_train_{}.csv"
            rule_csv = rule_csv_name.format(dataset, i)
            rule_df = pd.read_csv(rule_csv)
            for idx, pt in enumerate(prec_thresholds):
                train_pred_df = rule_predict(rule_df, train_df, tag=str(pt))
                valid_pred_df = rule_predict(rule_df, valid_df, tag=str(pt))
                test_pred_df = rule_predict(rule_df, test_df, tag=str(pt))
                rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df,
                                                                                                      train_pred_df)
                rule_id_list = rule_id_list[:500]

                _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
                _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)
                solutions, rule_coverage_list = init_solutions(rule_id_list, train_rule_coverage,
                                                               train_pos_indices, valid_rule_coverage)
                opt_solution = greedy_opt(solutions, rule_coverage_list, train_pos_indices, valid_pos_indices,
                                          k, pt, max_round, n_process)
                _, opt_recall = eval_single_solution(opt_solution, rule_id_list, test_rule_coverage, test_pos_indices)
                recalls["crsl"][idx].append(opt_recall)
                log_info += " cons{}_recall={}".format(pt, np.round(opt_recall, 4))
            logger.info(log_info)
        logging.info("Finish case_study_crsl on {} dataset".format(dataset))

        for idx, pt in enumerate(prec_thresholds):
            ret_row = [dataset, str(pt)]
            for method in methods:
                recs = recalls[method][idx]
                mean = np.round(np.mean(recs), 4)
                std = np.round(np.std(recs), 4)
                logger.info("{} prec_cons={} recall_means={} recall_stds={}".format(method, pt, mean, std))
                ret_row.append(metric2str(recs))
            ret_df.append(ret_row)

    for d in ['A1', 'A2', 'A3']:
        for pt in prec_thresholds:
            row = [d, pt] + [" 0$\pm$0 " for _ in methods]
            ret_df.append(row)
    ret_columns = ['DataSet', 'Prec'] + methods
    ret_df = pd.DataFrame(np.array(ret_df), columns=ret_columns)
    ret_df.to_csv(output_csv, index=False)
    return


def emo_comparison(dataset='bank',
                     output_csv="emo_comp.csv"):
    """
    run emo_comparison to reproduce figure 3 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/5_3_2")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    rule_csv_name = "../datasets/{}/rules/sr_on_train_{}.csv"
    n_process = 4
    pors_evals = []
    emo_evals = []
    for i in [1, 2, 3, 4, 5]:
        train_csv = data_csv_name.format(dataset, "train", i)
        valid_csv = data_csv_name.format(dataset, "valid", i)
        test_csv = data_csv_name.format(dataset, "test", i)
        rule_csv = rule_csv_name.format(dataset, i)

        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        test_df = pd.read_csv(test_csv)
        rule_df = pd.read_csv(rule_csv)
        logger.info("Begin Trial {}".format(i))

        train_pred_df = rule_predict(rule_df, train_df)
        valid_pred_df = rule_predict(rule_df, valid_df)
        test_pred_df = rule_predict(rule_df, test_df)
        rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df, train_pred_df)
        _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
        _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)
        solutions, rule_coverage_list = init_solutions(rule_id_list, train_rule_coverage,
                                                       train_pos_indices, valid_rule_coverage)
        _, pors_evaluations = pors(solutions,
                                   rule_coverage_list,
                                   train_pos_indices,
                                   valid_pos_indices,
                                   dataset_size=trainset_size,
                                   ssf_method='hvc-ss',
                                   ssf_param=dict(),
                                   k=10,
                                   max_round=80,
                                   n_process=n_process,
                                   eval_step=2,
                                   test_rule_coverage=test_rule_coverage,
                                   test_pos_indices=test_pos_indices,
                                   rule_id_list=rule_id_list)
        pors_evals.append(pors_evaluations)
        _, emo_evaluations = nsga2(solutions, rule_id_list, train_rule_coverage, valid_rule_coverage,
                                   train_pos_indices, valid_pos_indices, n_process=n_process, eval_step=2,
                                   test_rule_coverage=test_rule_coverage, test_pos_indices=test_pos_indices)
        emo_evals.append(emo_evaluations)

    pors_evals = pd.DataFrame(np.mean(np.array(pors_evals), axis=0), columns=['train_hv', 'valid_hv', 'test_hv', 'time'])
    emo_evals = pd.DataFrame(np.mean(np.array(emo_evals), axis=0), columns=['train_hv', 'valid_hv', 'test_hv', 'time'])

    pors_evals.to_csv(output_csv.split(".csv")[0] + "_pors.csv", index=False)
    emo_evals.to_csv(output_csv.split(".csv")[0] + "_emo.csv", index=False)
    return


def experiment_5_4_2_2(dataset='bank',
                       fbetas=[0.1, 0.2, 0.5],
                       output_csv="figure_5.csv"):
    """
    run experiment_5_4_2_2 to reproduce figure 5 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/5_4_2")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    rule_csv_name = "../datasets/{}/rules/sr_on_train_{}.csv"
    n_process = 4
    max_round = 50
    k = 10
    trials = 5
    pors_evals = np.zeros([trials, max_round, 1])
    greedy_evals = np.zeros([trials, max_round, len(fbetas)])
    for i in [1, 2, 3, 4, 5]:
        train_csv = data_csv_name.format(dataset, "train", i)
        valid_csv = data_csv_name.format(dataset, "valid", i)
        test_csv = data_csv_name.format(dataset, "test", i)
        rule_csv = rule_csv_name.format(dataset, i)

        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        test_df = pd.read_csv(test_csv)
        rule_df = pd.read_csv(rule_csv)
        logger.info("Begin Trial {}".format(i))

        train_pred_df = rule_predict(rule_df, train_df)
        valid_pred_df = rule_predict(rule_df, valid_df)
        test_pred_df = rule_predict(rule_df, test_df)
        rule_id_list, train_rule_coverage, train_pos_indices, trainset_size = data_preprocess(train_df, train_pred_df)
        _, valid_rule_coverage, valid_pos_indices, _ = data_preprocess(valid_df, valid_pred_df)
        _, test_rule_coverage, test_pos_indices, _ = data_preprocess(test_df, test_pred_df)
        solutions, rule_coverage_list = init_solutions(rule_id_list, train_rule_coverage,
                                                       train_pos_indices, valid_rule_coverage)
        _, pors_evaluations = pors(solutions,
                                   rule_coverage_list,
                                   train_pos_indices,
                                   valid_pos_indices,
                                   dataset_size=trainset_size,
                                   ssf_method='hvc-ss',
                                   ssf_param=dict(),
                                   k=k,
                                   max_round=max_round,
                                   n_process=n_process,
                                   eval_step=1)
        pors_evals[i-1, :, 0] = pors_evaluations[:, -1]
        for idx, fbeta in enumerate(fbetas):
            _, greedy_evaluations = beam_search(solutions, rule_coverage_list, train_pos_indices, valid_pos_indices,
                                                k, fbeta, max_round, n_process, eval_step=1)
            greedy_evals[i-1, :, idx] = greedy_evaluations[:, -1]

    pors_evals = np.mean(pors_evals, axis=0)
    greedy_evals = np.mean(greedy_evals, axis=0)
    evals = pd.DataFrame(np.hstack([pors_evals, greedy_evals]), columns=['PORS'] + fbetas)
    evals.to_csv(output_csv, index=False)

    return


def stage1_diversity(dataset='bank',
                         max_input_size=500,
                         output_csv='diversity.csv'):
    """
    run stage1_diversity to reproduce 5.3.1 in paper
    """
    save_path = os.path.join(MY_PATH, "../exp_result/stage1_comparison")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_csv = os.path.join(save_path, output_csv)

    data_csv_name = "../datasets/{}/data/{}_{}.csv"
    rule_csv_name = "../datasets/{}/rules/{}_on_train_{}.csv"

    ret_df = []
    for stage1 in ['sr', 'te']:
        ret_row = [stage1]
        train_csv = data_csv_name.format(dataset, "train", 1)
        valid_csv = data_csv_name.format(dataset, "valid", 1)
        test_csv = data_csv_name.format(dataset, "test", 1)
        rule_csv = rule_csv_name.format(dataset, stage1, 1)

        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        test_df = pd.read_csv(test_csv)
        data_df = train_df.append(valid_df).append(test_df)
        rule_df = pd.read_csv(rule_csv)

        pred_df = rule_predict(rule_df, data_df)
        rule_id_list, rule_coverage, pos_indices, dataset_size = data_preprocess(data_df, pred_df)
        if stage1 == "sr":
            sub_rule_id_list = [rule_id for rule_id in rule_id_list if
                                int(rule_id.split("_")[1]) < max_input_size // 10]
        else:
            sub_rule_id_list = rule_id_list[:max_input_size]
        _, rule_coverage_list = init_solutions(sub_rule_id_list, rule_coverage, pos_indices, rule_coverage)
        precs = []
        recs = []
        for c in rule_coverage_list:
            [prec, rec] = eval_ruleset_coverage(c[0], pos_indices)
            precs.append(prec)
            recs.append(rec)
        ret_row += [metric2str(precs, 4), metric2str(recs, 4)]
        ret_df.append(ret_row)

    ret_df = pd.DataFrame(ret_df, columns=['Stage-1', 'Precision',  'Recall'])
    ret_df.to_csv(output_csv, index=False)

    return


def run_all():
    ssf_comparison(stage1="TreeEns", max_input_size=500, output_csv="table_3.csv")
    ssf_comparison(stage1="SpectralRules", max_input_size=500, output_csv="table_4.csv")

    stage1_comparison(dataset='bank', output_csv="sr_vs_tree.csv")
    stage1_diversity(dataset='bank', max_input_size=500, output_csv='diversity.csv')

    emo_comparison(dataset='bank', output_csv="emo_comp.csv")

    case_study_crsl(prec_thresholds=[0.3, 0.5, 0.7], output_csv="table_5.csv")
    case_study_fscore(fbetas=[0.1, 0.2, 0.5], output_csv="fscore.csv")


if __name__ == "__main__":
    run_all()
