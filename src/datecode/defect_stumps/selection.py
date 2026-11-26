"""Per-shift defect model selection.

Both candidates are trained on the mature fitting history and compared on the
same mature replay tickets.  The boosted stumps replace the reflection-band
mean only when their overall absolute error is lower *and* no current shift
zone would miss more over-threshold batches.  A shift zone without mature
replay evidence keeps the mean so the miss guard stays testable.
"""

from dataclasses import dataclass
from math import fsum, isfinite
from statistics import mean

from .contracts import MODEL_NAME, Fit, Prediction
from .boost_model import score_kernel
from .service import fit_model

MEAN_NAME = '工位反光等级均值'
RISK_CUTOFF = 12


@dataclass(frozen=True, slots=True)
class CandidateScore:
    name: str
    count: int
    mae: float | None
    zone_misses: tuple[tuple[str, int], ...]
    total_miss: int


@dataclass(frozen=True, slots=True)
class ModelSelection:
    chosen: str
    reason: str
    threshold: int
    candidates: tuple[CandidateScore, ...]
    missing_support: tuple[str, ...]
    kernel: Fit
    mean_profile: tuple[tuple[int, float], ...]
    fallback: float

    def score(self, vector):
        return self.annotate(vector).score

    def annotate(self, vector):
        if self.chosen == MODEL_NAME:
            expected, balance = score_kernel(self.kernel.parameters, vector)
            contributions = tuple(balance)
        else:
            expected = lookup_mean(self.mean_profile, self.fallback, vector)
            contributions = (expected,)
        expected = max(0.0, expected)
        if not isfinite(expected) or not all(map(isfinite, contributions)):
            raise ArithmeticError('班次件数预测无法使用')
        return Prediction(vector.key, vector.cohort, vector.day, expected, contributions,
                          vector.missing, vector.excursions, vector.outcome, vector.observed,
                          vector.input_values)

    def predict(self, observations):
        return tuple(map(self.annotate, observations))

    def weights(self):
        if self.chosen == MODEL_NAME:
            return self.kernel.evidence
        profile = tuple((f'反光等级 {band} 档均值', value) for band, value in self.mean_profile)
        return (*profile, ('全库缺陷均值', self.fallback))


def reflection_band(vector):
    return int(vector.input_values[4] // 18)


def fit_reflection_mean(training):
    buckets = {}
    for row in training:
        buckets.setdefault(reflection_band(row), []).append(row.outcome)
    profile = tuple(sorted((band, fsum(values) / len(values)) for band, values in buckets.items()))
    return profile, mean(row.outcome for row in training)


def lookup_mean(profile, fallback, vector):
    return dict(profile).get(reflection_band(vector), fallback)


def _mature_replay(replay):
    return tuple(row for row in replay if row.outcome is not None and row.observed)


def _mae(rows, scorer):
    if not rows:
        return None
    return fsum(abs(scorer(row) - row.outcome) for row in rows) / len(rows)


def _zone_misses(rows, scorer, threshold):
    counts = {}
    for row in rows:
        counts.setdefault(row.cohort, 0)
        if row.outcome >= threshold and scorer(row) < threshold:
            counts[row.cohort] += 1
    return tuple(sorted(counts.items()))


def _supported_zones(mature, threshold):
    support = {}
    for row in mature:
        support.setdefault(row.cohort, 0)
        if row.outcome >= threshold:
            support[row.cohort] += 1
    return support


def _unsupported_zones(mature, live_zones, threshold):
    support = _supported_zones(mature, threshold)
    return tuple(zone for zone in live_zones if support.get(zone, 0) == 0)


def _worse_zones(stump_misses, mean_misses, live_zones):
    stump = dict(stump_misses)
    base = dict(mean_misses)
    return tuple(zone for zone in live_zones if stump.get(zone, 0) > base.get(zone, 0))


def decide_model(stump_score, mean_score, missing, live_zones):
    """Pick a candidate from the shared mature replay comparison.

    ``missing`` lists current-shift zones without mature over-threshold replay
    evidence; those keep the mean because their miss guard cannot be tested.
    """
    if stump_score.count == 0:
        return MEAN_NAME, '回放窗没有成熟实绩，保留反光均值'
    if missing:
        return MEAN_NAME, '当班分区缺成熟越线实绩：' + '、'.join(missing) + '，沿用反光均值'
    if stump_score.mae is None or mean_score.mae is None or not stump_score.mae < mean_score.mae:
        return MEAN_NAME, '树桩整体绝对误差未低于反光均值，保留反光均值'
    worse = _worse_zones(stump_score.zone_misses, mean_score.zone_misses, live_zones)
    if worse:
        return MEAN_NAME, '树桩在当班分区漏报增加：' + '、'.join(worse) + '，保留反光均值'
    return MODEL_NAME, '树桩整体绝对误差更低且当班各区漏报未增加，换用树桩'


def select_model(train_features, replay_features, live_features, feature_stamp, *, threshold=RISK_CUTOFF):
    kernel = fit_model(train_features, feature_stamp)
    profile, fallback = fit_reflection_mean(train_features)

    def stump(vector):
        return max(0.0, score_kernel(kernel.parameters, vector)[0])

    def reflection(vector):
        return max(0.0, lookup_mean(profile, fallback, vector))

    mature = _mature_replay(replay_features)
    # Zones already registered by fitting; unregistered zones are routed to
    # manual review rather than scored on quota, so they need no miss guard.
    registered = {row.cohort for row in train_features}
    live_zones = sorted({row.cohort for row in live_features} & registered)

    stump_misses = _zone_misses(mature, stump, threshold)
    mean_misses = _zone_misses(mature, reflection, threshold)
    stump_score = CandidateScore(MODEL_NAME, len(mature), _mae(mature, stump), stump_misses,
                                 sum(count for _, count in stump_misses))
    mean_score = CandidateScore(MEAN_NAME, len(mature), _mae(mature, reflection), mean_misses,
                                sum(count for _, count in mean_misses))
    missing = _unsupported_zones(mature, live_zones, threshold)
    chosen, reason = decide_model(stump_score, mean_score, missing, live_zones)
    return ModelSelection(chosen, reason, threshold, (stump_score, mean_score), missing,
                          kernel, profile, fallback)


def selection_evidence(selection):
    rows = []
    support = '缺 ' + '、'.join(selection.missing_support) if selection.missing_support else '齐全'
    for score in selection.candidates:
        detail = '、'.join(f'{zone} {count}' for zone, count in score.zone_misses) or '—'
        rows.append({'候选': score.name, '回放实绩': score.count,
                     '整体绝对误差': None if score.mae is None else round(score.mae, 4),
                     '分区漏报': detail, '漏报合计': score.total_miss,
                     '成熟越线实绩': support,
                     '判定': '选用' if score.name == selection.chosen else '未选用'})
    rows.append({'候选': '换用判定', '回放实绩': selection.candidates[0].count,
                 '整体绝对误差': None,
                 '分区漏报': '整体误差低只利于排额度；当班分区漏报不得增加',
                 '漏报合计': None,
                 '成熟越线实绩': support,
                 '判定': selection.chosen + '：' + selection.reason})
    return rows


def deployment_evidence(selection):
    return [{'项目': '当班选用模型', '取值': selection.chosen},
            {'项目': '换用依据', '取值': selection.reason},
            {'项目': '越线阈值', '取值': selection.threshold},
            {'项目': '成熟回放实绩', '取值': selection.candidates[0].count}]
