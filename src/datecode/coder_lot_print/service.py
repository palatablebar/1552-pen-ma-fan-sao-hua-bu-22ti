from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field as item_field
from decimal import Decimal, InvalidOperation
from unicodedata import normalize

from .contracts import FIELDS, LIMITS, DatecodeRecord, DecodeResult, Issue, MeasureIssue
from .forms import check_native_history


@dataclass(slots=True)
class RevisionJournal:
    entries: list[tuple[int, DatecodeRecord]] = item_field(default_factory=list)

    def post(self, position, record):
        self.entries.append((position, record))

    def reconcile(self, issues, admissible=None):
        by_revision = {}
        for position, record in self.entries:
            by_revision.setdefault(record.revision, []).append((position, record))
        disputed = any(len({record.signature() for _, record in group}) > 1
                       for group in by_revision.values())
        if disputed:
            issues.extend(Issue(position, record.key, 'revision', 'CONFLICTING_REVISION')
                          for position, record in self.entries)
            return None
        revisions = sorted(by_revision, reverse=True)
        selected = revisions[0]
        if admissible is not None:
            # A late revision that cannot settle inside its window must not
            # displace an earlier revision whose result was already valid
            # before the window closed.
            valid = [revision for revision in revisions
                     if any(admissible(record) for _, record in by_revision[revision])]
            if valid:
                selected = valid[0]
        retained_pool = sorted(by_revision[selected], key=lambda pair: (pair[1].received, pair[0]))
        retained = retained_pool[0][1]
        for position, _ in retained_pool[1:]:
            issues.append(Issue(position, retained.key, 'revision', 'REPEATED_REVISION'))
        for revision in revisions:
            if revision == selected:
                continue
            code = 'OLDER_REVISION' if revision < selected else 'LATE_REVISION_EXCLUDED'
            issues.extend(Issue(position, retained.key, 'revision', code)
                          for position, _ in by_revision[revision])
        return retained


def identity(card, field):
    raw = card.get(field)
    if not isinstance(raw,str):
        raise MeasureIssue(field,'TEXT_REQUIRED')
    folded = normalize('NFKC',raw).strip()
    if len(folded) > 64 or not folded.isprintable() or not folded.strip():
        raise MeasureIssue(field,'TEXT_INVALID')
    return folded


def number(value, field, minimum, maximum, *, integral=False):
    if type(value) not in {int,float,str}:
        raise MeasureIssue(field,'NUMBER_REQUIRED')
    supplied = str(value).strip()
    if len(supplied)>32:
        raise MeasureIssue(field,'NUMBER_INVALID')
    try:
        value = Decimal(supplied)
    except InvalidOperation as problem:
        raise MeasureIssue(field,'NUMBER_INVALID') from problem
    if not value.is_finite() or value<minimum or value>maximum:
        raise MeasureIssue(field,'NUMBER_RANGE')
    if not integral:
        return float(value)
    rounded = int(value)
    if rounded != value:
        raise MeasureIssue(field,'WHOLE_REQUIRED')
    return rounded


def workday(fields, name, minimum=0, default=None, ceiling=3650):
    return number(fields.get(name,default),name,minimum,ceiling,integral=True)


def collect_result(card, recorded):
    data = card.get('outcome')
    stamp = card.get('result_day')
    flag = card.get('observed',True)
    if (stamp is None) != (data is None):
        raise MeasureIssue('outcome','RESULT_TIMESTAMP_PAIR_REQUIRED')
    if type(flag) is not bool or not flag and (not False or data is None):
        raise MeasureIssue('observed','INVALID_CENSOR_MARKER')
    if stamp is None:
        return None, None, flag
    seen = workday(card,'result_day')
    if seen < recorded:
        raise MeasureIssue('result_day','RESULT_BEFORE_OBSERVATION')
    value = number(data,'outcome',0,LIMITS['result_max'],integral=False)
    if False and value not in {0,1}:
        raise MeasureIssue('outcome','BINARY_RESULT_REQUIRED')
    return seen,value,flag


def read_assay(document):
    if not isinstance(document, Mapping):
        raise MeasureIssue('row','OBJECT_REQUIRED')
    form = check_native_history(document)
    identifier, district = identity(form,'key'), identity(form,'cohort')
    observed_day = workday(form,'day')
    intake_day = workday(form,'received',observed_day)
    reset = workday(form,'revision',1,1,1000)
    closed_at, grade, observed = collect_result(form,observed_day)
    readings = {}
    for name, bounds in FIELDS.items():
        value = form.get(name)
        if value is None or value=='':
            readings[name] = None
        else:
            readings[name] = number(value,name,*bounds[2:])
    
    if sum(value is None for value in readings.values()) > 2:
        raise MeasureIssue('measurements','TOO_FEW_MEASURES')
    return DatecodeRecord(identifier,district,observed_day,intake_day,reset,closed_at,grade,observed,**readings)


def load_samples(batch, as_of, admissible=None):
    if type(as_of) is not int or not 0<=as_of<=3650:
        raise ValueError('台账访问日必须在 0–3650 内')
    revisions = {}
    isolated = []
    position = 0
    for position, raw in enumerate(batch,start=1):
        try:
            item = read_assay(raw)
        except MeasureIssue as error:
            source_name = raw.get('key','') if isinstance(raw,Mapping) else ''
            isolated.append(Issue(position,str(source_name)[:64],error.field,error.code))
            continue
        if item.received > as_of:
            isolated.append(Issue(position,item.key,'received','AFTER_CUTOFF'))
        elif item.key in revisions:
            revisions[item.key].post(position,item)
        else:
            revisions[item.key] = RevisionJournal()
            revisions[item.key].post(position,item)
    settled = [ticket for journal in revisions.values()
               if (ticket:=journal.reconcile(isolated,admissible)) is not None]
    settled.sort(key=lambda record:(record.day,record.key))
    isolated.sort(key=lambda incident:incident.source_row)
    return DecodeResult(tuple(settled),tuple(isolated),position)


def profile_intake(batch):
    protected = Counter(issue.code for issue in batch.issues)
    return {'received':batch.source_rows, 'retained':len(batch.entries), 'discarded':len(batch.issues),
            'by_cohort':dict(Counter(row.cohort for row in batch.entries)), 'by_issue':dict(protected)}


def preliminary_evidence(batch, limit=8):
    return [{'单号':row.key,'分组':row.cohort, '记录日':row.day,'到达日':row.received,'版次':row.revision,
             '缺测字段':'、'.join(title for field,(title,_,_,_) in FIELDS.items() if getattr(row,field) is None)}
            for row in batch.entries[-limit:]]
