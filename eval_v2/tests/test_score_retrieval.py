from __future__ import annotations

import math

from eval_v2.scripts.score_retrieval import (
    aggregate_at_k,
    choose_operating_point,
    f_beta,
    score_query,
)


def _record():
    return {
        "query_id": "q1",
        "relevant": [
            {"evidence_group": "g1", "retrieval_unit_id": "u1", "relevance": 3},
            {"evidence_group": "g2", "retrieval_unit_id": "u2", "relevance": 2},
        ],
        "retrieved": [
            {"retrieval_unit_id": "noise", "score": 0.9},
            {"retrieval_unit_id": "u1", "score": 0.8},
            {"retrieval_unit_id": "u2", "score": 0.7},
        ],
    }


def test_hit_recall_precision_mrr_and_ndcg():
    row = score_query(_record(), 2)
    assert row["hit"] == 1
    assert row["recall"] == 0.5
    assert row["precision"] == 0.5
    assert row["rr"] == 0.5
    expected_dcg = 7 / math.log2(3)
    ideal_dcg = 7 + 3 / math.log2(3)
    assert row["ndcg"] == expected_dcg / ideal_dcg


def test_threshold_changes_precision_and_recall():
    low = score_query(_record(), 3, threshold=0.0)
    high = score_query(_record(), 3, threshold=0.75)
    assert low["recall"] == 1
    assert high["recall"] == 0.5
    assert low["precision"] == 2 / 3
    assert high["precision"] == 1 / 2


def test_no_answer_false_positive_is_reported_separately():
    no_answer = {"query_id": "q0", "relevant": [], "retrieved": [{"retrieval_unit_id": "x", "score": 0.1}]}
    aggregate = aggregate_at_k([_record(), no_answer], 3)
    assert aggregate["no_answer_false_positive_rate"] == 1
    assert aggregate["answerable_count"] == 1
    assert aggregate["no_answer_count"] == 1


def test_operating_point_uses_recall_and_fpr_constraints_then_f2():
    curve = [
        {"recall": 0.96, "precision": 0.60, "no_answer_false_positive_rate": 0.04, "mean_returned": 5, "threshold": 0.2},
        {"recall": 0.94, "precision": 0.90, "no_answer_false_positive_rate": 0.01, "mean_returned": 2, "threshold": 0.7},
        {"recall": 0.97, "precision": 0.50, "no_answer_false_positive_rate": 0.10, "mean_returned": 6, "threshold": 0.1},
    ]
    selected = choose_operating_point(curve, recall_floor=0.95, max_no_answer_fpr=0.05)
    assert selected is not None
    assert selected["threshold"] == 0.2
    assert selected["f_beta"] == f_beta(0.60, 0.96, 2)

