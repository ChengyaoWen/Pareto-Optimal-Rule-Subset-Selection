import pandas as pd
from pyroaring import BitMap
from typing import Optional


def data_preprocess(data_df: pd.DataFrame,
                    rule_coverage_df: pd.DataFrame,
                    label_col='class',
                    uid_col='id'):
    rule_coverage_map = dict()
    positive_indices = BitMap()
    dataset_size = data_df.shape[0]
    rule_list = list()
    id_index_map = dict()

    assert label_col in data_df.columns
    assert uid_col in data_df.columns
    index = 0
    for _, row in data_df.iterrows():
        uid = str(int(row[uid_col]))
        if uid not in id_index_map.keys():
            id_index_map[uid] = index
            if int(row[label_col]) == 1:
                positive_indices.add(index)
            index += 1

    for _, row in rule_coverage_df.iterrows():
        rule_id = row['rule_id']
        rule_list.append(rule_id)
        rule_coverage = row['covered_indices']
        if len(rule_coverage) > 0:
            rule_coverage = BitMap(list(map(lambda uid: id_index_map[uid],
                                            row['covered_indices'].split(","))))
        else:
            rule_coverage = BitMap()
        rule_coverage_map[rule_id] = rule_coverage

    return rule_list, rule_coverage_map, positive_indices, dataset_size


def rule_predict(rule_df: pd.DataFrame,
                 data_df: pd.DataFrame,
                 tag: Optional[str]=None):
    """
    generate rule covering dataframe based on rules and dataset
    """
    rule_coverage_df = []
    if tag:
        assert 'tag' in rule_df.columns
        rule_df = rule_df.query("tag == {}".format(tag))
    for _, row in rule_df.iterrows():
        rule_id = row['rule_id']
        rule_content = row['rule_content']
        rule_coverage = data_df.query(rule_content)
        if rule_coverage.shape[0] == 0:
            covered_indices = ""
        else:
            covered_indices = ",".join(list(rule_coverage['id'].map(lambda x: str(int(x)))))
        rule_coverage_df.append([rule_id, covered_indices])
    rule_coverage_df = pd.DataFrame(rule_coverage_df, columns=['rule_id', 'covered_indices'])
    return rule_coverage_df
