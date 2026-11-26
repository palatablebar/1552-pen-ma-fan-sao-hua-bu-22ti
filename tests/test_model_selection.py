from dataclasses import replace

import pytest

from datecode.coder_lot_print.service import load_samples
from datecode.defect_stumps.selection import (
    MEAN_NAME, MODEL_NAME, CandidateScore, decide_model, select_model,
)
from datecode.defect_stumps.contracts import MODEL_NAME as STUMP_NAME
from datecode.ink_change_audit.service import assign_fold, build_windows, model_inputs
from datecode.ink_change_audit.contracts import WindowRule
from datecode.pipeline import open_fixture
from datecode.print_quality_terms.service import fit_features, transform_windows


def score(name, mae, misses, count=5):
    return CandidateScore(name, count, mae, tuple(misses.items()), sum(misses.values()))


def test_stump_replaces_mean_only_with_lower_error_and_no_zone_regression():
    stump = score(MODEL_NAME, 1.2, {'A': 1, 'B': 0})
    mean = score(MEAN_NAME, 2.0, {'A': 2, 'B': 1})
    assert decide_model(stump, mean, (), ['A', 'B'])[0] == MODEL_NAME


def test_higher_overall_error_keeps_reflection_mean():
    stump = score(MODEL_NAME, 2.4, {'A': 0, 'B': 0})
    mean = score(MEAN_NAME, 2.0, {'A': 0, 'B': 0})
    chosen, reason = decide_model(stump, mean, (), ['A', 'B'])
    assert chosen == MEAN_NAME and '绝对误差' in reason


def test_zone_miss_regression_keeps_mean_despite_lower_error():
    # Aggregate error improves, but zone B would let one more over-threshold
    # batch through; the miss guard must veto the swap.
    stump = score(MODEL_NAME, 1.0, {'A': 0, 'B': 2})
    mean = score(MEAN_NAME, 2.0, {'A': 1, 'B': 1})
    chosen, reason = decide_model(stump, mean, (), ['A', 'B'])
    assert chosen == MEAN_NAME and '漏报增加' in reason


def test_miss_improvement_outside_live_zones_does_not_veto():
    stump = score(MODEL_NAME, 1.0, {'A': 0, 'C': 3})
    mean = score(MEAN_NAME, 2.0, {'A': 1, 'C': 1})
    # C is not a current shift zone, so only A and B are guarded.
    assert decide_model(stump, mean, (), ['A', 'B'])[0] == MODEL_NAME


def test_missing_mature_support_for_live_zone_keeps_mean():
    stump = score(MODEL_NAME, 1.0, {'A': 0})
    mean = score(MEAN_NAME, 2.0, {'A': 2})
    chosen, reason = decide_model(stump, mean, ('B',), ['A', 'B'])
    assert chosen == MEAN_NAME and 'B' in reason and '缺成熟越线实绩' in reason


def test_no_mature_replay_keeps_mean():
    stump = score(MODEL_NAME, None, {}, count=0)
    mean = score(MEAN_NAME, None, {}, count=0)
    assert decide_model(stump, mean, (), ['A'])[0] == MEAN_NAME


def selection_inputs():
    sample = open_fixture()
    intake = load_samples(sample['records'], sample['as_of'])
    windows = build_windows(intake.entries)
    model_inputs(windows)
    state = fit_features(windows.train)
    return transform_windows(windows, state), state.fingerprint


def test_preset_swap_is_recorded_with_shared_replay():
    features, stamp = selection_inputs()
    selection = select_model(features.train, features.replay, features.live, stamp)
    assert selection.chosen == STUMP_NAME
    stump, mean = selection.candidates
    assert stump.count == mean.count > 0
    assert sum(count for _, count in stump.zone_misses) == stump.total_miss


def test_live_zone_without_over_threshold_support_keeps_mean():
    features, stamp = selection_inputs()
    stripped = tuple(row for row in features.replay
                      if not (row.cohort == '袋装B线' and row.outcome >= 12))
    selection = select_model(features.train, stripped, features.live, stamp)
    assert selection.chosen == MEAN_NAME
    assert '袋装B线' in selection.missing_support
    # Predictions now carry the reflection-band mean; no stump contributions.
    live = selection.predict(features.live[:1])[0]
    assert len(live.contributions) == 1
    assert live.score == pytest.approx(dict(selection.mean_profile)[int(live.input_values[4] // 18)])


def test_late_revision_cannot_displace_preclose_valid_ticket(observation):
    # Revision 2 arrives before the cutoff but its result matures after the
    # fitting window closes.  Revision 1 was already valid pre-close and must
    # be the one kept.
    rule = WindowRule.standard()
    original = {**observation, 'day': 30, 'received': 31, 'result_day': 33}
    late = {**original, 'revision': 2, 'received': 39, 'result_day': 45}
    admissible = lambda record: assign_fold(record, rule)[0] is not None
    loaded = load_samples([original, late], 96, admissible)
    assert [entry.revision for entry in loaded.entries] == [1]
    codes = [issue.code for issue in loaded.issues]
    assert codes == ['LATE_REVISION_EXCLUDED']
    # The pre-close ticket still lands in the fitting window.
    assert [row.key for row in build_windows(loaded.entries).train] == [observation['key']]


def test_late_revision_without_adjudication_keeps_legacy_newest(observation):
    # Without an admissibility rule the newest revision still wins.
    original = {**observation, 'day': 30, 'received': 31, 'result_day': 33}
    late = {**original, 'revision': 2, 'received': 39, 'result_day': 45}
    loaded = load_samples([original, late], 96)
    assert [entry.revision for entry in loaded.entries] == [2]
