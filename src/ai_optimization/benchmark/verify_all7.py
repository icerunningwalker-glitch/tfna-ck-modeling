"""Independent standard-library arithmetic; no training or sklearn imports."""
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product
from statistics import median
import argparse
import csv
import hashlib
import json
import math

from layout import PROJECT_ROOT, locations
OUT = PROJECT_ROOT
LOCATIONS = None


def read(name):
    relative = Path(name)
    loc = LOCATIONS or locations(OUT)
    if relative.parts[0] in ('data', 'models', 'results'):
        path = loc[relative.parts[0]].joinpath(*relative.parts[1:])
    else:
        path = loc['logs'] / name
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def arithmetic(rows):
    positive = [float(r['predicted_score']) for r in rows if int(r['observed_count']) >= 1]
    negative = [float(r['predicted_score']) for r in rows if int(r['observed_count']) == 0]
    auc = sum(1. if p > n else .5 if p == n else 0. for p in positive for n in negative)
    auc /= len(positive) * len(negative)
    error = [float(r['predicted_score']) - int(r['observed_count']) for r in rows]
    mse = sum(e * e for e in error) / len(rows)
    result = dict(n=len(rows), AUC=auc, MAE=sum(abs(e) for e in error) / len(rows), MSE=mse, RMSE=math.sqrt(mse))
    for actual, predicted, field in ((0, 0, 'TN'), (0, 1, 'FP'), (1, 0, 'FN'), (1, 1, 'TP')):
        result[field] = sum((int(r['observed_count']) >= 1) == bool(actual)
                            and (float(r['predicted_score']) >= .5) == bool(predicted) for r in rows)
    integer_errors = [abs(math.floor(float(r['predicted_score']) + .5) - int(r['observed_count'])) for r in rows]
    result['integer_MAE'] = sum(integer_errors) / len(rows)
    result['exact_count_fraction'] = sum(e == 0 for e in integer_errors) / len(rows)
    return result


def main():
    global OUT, LOCATIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT,
                        help='Unified project root; defaults to the project containing this source file.')
    parser.add_argument('--output', type=Path,
                        help='Optional new JSON destination; without this option only print checks.')
    parser.add_argument('--data-dir', type=Path, help='Override benchmark input directory.')
    parser.add_argument('--results-dir', type=Path, help='Override saved/new-run results directory.')
    parser.add_argument('--log-dir', type=Path, help='Override saved/new-run protocol log directory.')
    args = parser.parse_args()
    OUT = args.root.resolve()
    LOCATIONS = locations(OUT)
    for key in ('data', 'results', 'log'):
        value = getattr(args, key + '_dir')
        if value:
            LOCATIONS['logs' if key == 'log' else key] = value.resolve()
    checks = []
    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        assert condition, name
    source = read('data/uniform_single_seed_labels.csv')
    assignments = read('data/outer_assignments.csv')
    inner = read('data/training_group_CV_assignments.csv')
    metrics = read('results/metrics.csv')
    folded = read('results/fold_metrics.csv')
    predictions = read('results/predictions.csv')
    oof = read('results/outer_test_predictions.csv')
    train = read('results/training_fit_predictions.csv')
    candidates = read('training_CV_candidate_losses.csv')
    parameters = json.loads((LOCATIONS['logs'] / 'training_only_parameters.json').read_text())
    plan = json.loads((LOCATIONS['logs'] / 'analysis_plan.json').read_text())
    smap = {r['sequence_id']: r for r in source}
    amap = {r['sequence_id']: int(r['outer_fold']) for r in assignments}
    expected_models = set(plan['models'] + plan['baselines'])
    check('Recorded seed, count threshold and PCA dimension match the fixed protocol',
          plan['seed'] == 20261003 and plan['classification_threshold'] == .5 and plan['pca_components'] == 8)
    check('123 unique source sequences with docking seed 104729', len(source) == len(smap) == 123
          and all(r['docking_seed'] == '104729' for r in source))
    canonical = lambda sequence: min(sequence, sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1])
    check('123 valid canonical-distinct 20nt sequences with family IDs',
          all(len(r['sequence_20nt']) == 20 and set(r['sequence_20nt']) <= set('ACGT') and r['family_id'] for r in source)
          and len({canonical(r['sequence_20nt']) for r in source}) == 123)
    check('Count labels are 31/90/2', Counter(int(r['observed_count']) for r in source) == {0:31, 1:90, 2:2})
    check('Every sequence assigned once to one of five outer test folds', len(assignments) == len(amap) == 123
          and set(amap) == set(smap) and Counter(amap.values()) == {1:25, 2:25, 3:25, 4:24, 5:24})
    check('Fold metadata match label sequences, families and counts',
          all(all(r[key] == smap[r['sequence_id']][key] for key in ('sequence_20nt', 'family_id', 'observed_count'))
              for r in assignments))
    check('Two count-2 sequences occupy distinct test folds',
          len({amap[s] for s, r in smap.items() if int(r['observed_count']) == 2}) == 2)
    model_count = len(expected_models)
    check('All configured algorithms/baselines have unique OOF predictions',
          len(oof) == model_count*123 and {r['model'] for r in oof} == expected_models
          and len({(r['model'], r['sequence_id']) for r in oof}) == model_count*123)
    check('Four distinct in-sample fits per configured model/sequence', len(train) == model_count*492
          and len({(r['model'], r['sequence_id'], r['outer_fold']) for r in train}) == model_count*492
          and {r['model'] for r in train} == expected_models
          and {r['sequence_id'] for r in train} == set(smap)
          and {int(r['outer_fold']) for r in train} == set(range(1, 6))
          and set(Counter((r['model'], r['sequence_id']) for r in train).values()) == {4})
    check('No test sequence enters its own outer training predictions',
          all(amap[r['sequence_id']] == int(r['outer_fold']) for r in oof)
          and all(amap[r['sequence_id']] != int(r['outer_fold']) for r in train))
    check('All output labels match source', all(int(r['observed_count']) == int(smap[r['sequence_id']]['observed_count'])
          for r in oof + train + predictions))
    check('All predictions are finite nonnegative count scores',
          all(math.isfinite(float(r['predicted_score'])) and float(r['predicted_score']) >= 0 for r in oof + train + predictions))
    check('Unique combined diagnostic/OOF rows cover every model and sequence', len(predictions) == model_count*246
          and {(r['model'], r['split'], r['sequence_id']) for r in predictions} ==
          {(model, split, sequence_id) for model in expected_models for split in ('train', 'test') for sequence_id in smap})
    check('Stored binary outcomes match count and score thresholds',
          all(int(r['observed_positive']) == int(int(r['observed_count']) >= 1)
              and int(r['predicted_positive']) == int(float(r['predicted_score']) >= .5) for r in oof + predictions))
    omap = {(r['model'], r['sequence_id']): float(r['predicted_score']) for r in oof}
    tmap = defaultdict(list)
    for r in train:
        tmap[(r['model'], r['sequence_id'])].append(float(r['predicted_score']))
    check('Combined test is exact OOF and train is exact four-fit arithmetic mean',
          all(math.isclose(float(r['predicted_score']), omap[(r['model'], r['sequence_id'])] if r['split']=='test'
              else sum(tmap[(r['model'], r['sequence_id'])])/4, rel_tol=1e-10, abs_tol=1e-12) for r in predictions))
    check('Summary and fold metrics cover each configured split exactly once',
          len(metrics) == 2*model_count and len(folded) == 10*model_count
          and {(r['model'], r['split']) for r in metrics} == {(m, s) for m in expected_models for s in ('train', 'test')}
          and {(r['model'], r['split'], int(r['outer_fold'])) for r in folded} ==
          {(m, s, f) for m in expected_models for s in ('train', 'test') for f in range(1, 6)})
    for m in metrics + folded:
        if 'outer_fold' in m:
            pool = oof if m['split'] == 'test' else train
            rows = [r for r in pool if r['model'] == m['model'] and r['outer_fold'] == m['outer_fold']]
        else:
            rows = [r for r in predictions if r['model'] == m['model'] and r['split'] == m['split']]
        expected = arithmetic(rows)
        # Per-fold TabPFN losses were calculated with float32; raw CSV values allow float64 recomputation.
        tolerance = 1e-7 if m['model'] == 'TabPFN' and 'outer_fold' in m else 1e-10
        check('/'.join([m['model'], m['split'], m.get('outer_fold','pooled')]) + ': AUC/errors/confusion verified',
              all(math.isclose(float(m[k]), v, rel_tol=tolerance, abs_tol=tolerance) for k,v in expected.items()))
    check('492 inner validation assignment records, one per eligible sequence/outer fold',
          len(inner) == 492 and len({(r['outer_fold'], r['sequence_id']) for r in inner}) == 492)
    check('Inner assignments use valid folds and source family IDs',
          all(int(r['outer_fold']) in range(1, 6) and int(r['training_CV_fold']) in range(1, 4)
              and r['family_id'] == smap[r['sequence_id']]['family_id'] for r in inner))
    for outer in range(1,6):
        eligible = {s for s,f in amap.items() if f != outer}
        subset = [r for r in inner if int(r['outer_fold']) == outer]
        check(f'Outer {outer}: inner assignments use only outer training sequences',
              {r['sequence_id'] for r in subset} == eligible)
        for fold in range(1,4):
            val = {r['sequence_id'] for r in subset if int(r['training_CV_fold']) == fold}
            check(f'Outer {outer}/inner {fold}: nonempty fit and validation sets', bool(val) and bool(eligible-val))
            val_family = {smap[s]['family_id'] for s in val}
            fit_family = {smap[s]['family_id'] for s in eligible-val}
            check(f'Outer {outer}/inner {fold}: no family crosses fit/validation', not val_family.intersection(fit_family))
    pmap = {(r['model'], int(r['outer_fold'])): r for r in parameters}
    expected_parameter_keys = {(model, fold) for model in expected_models for fold in range(1, 6)}
    check('Each configured model has one recorded selection per outer fold',
          len(parameters) == len(pmap) and set(pmap) == expected_parameter_keys)
    tuned = expected_models - {'Training-median baseline'}
    check('Candidate logs contain only tuned algorithms and valid outer folds',
          {r['model'] for r in candidates} == tuned
          and all(int(r['outer_fold']) in range(1, 6) for r in candidates))
    for model in expected_models:
        for fold in range(1, 6):
            selected_parameter = pmap[(model, fold)]
            if model == 'Training-median baseline':
                expected_median = median(int(smap[s]['observed_count']) for s, f in amap.items() if f != fold)
                check(f'{model}/{fold}: median uses only outer training targets',
                      selected_parameter['parameters']['training_median'] == expected_median)
                continue
            rows = [r for r in candidates if r['model'] == model and int(r['outer_fold']) == fold]
            grid = plan['grids'][model]
            keys = list(grid)
            expected_grid = {json.dumps({'regressor__model__' + key: value for key, value in zip(keys, values)}, sort_keys=True)
                             for values in product(*(grid[key] for key in keys))}
            actual_grid = {json.dumps(json.loads(r['parameters']), sort_keys=True) for r in rows}
            check(f'{model}/{fold}: all predefined grid candidates recorded exactly once',
                  len(rows) == len(expected_grid) and actual_grid == expected_grid
                  and {int(r['candidate']) for r in rows} == set(range(len(rows))))
            check(f'{model}/{fold}: each selection loss is the unweighted three-fold mean',
                  all(math.isclose(float(r['mean_CV_MAE']), sum(float(r[f'fold_MAE_{k}']) for k in range(1, 4))/3,
                                   rel_tol=1e-10, abs_tol=1e-12) for r in rows))
            chosen = [r for r in rows if r['selected'] == 'True']
            check(f'{model}/{fold}: recorded parameters are the minimum inner-CV MAE selection',
                  len(chosen) == 1 and json.loads(chosen[0]['parameters']) == selected_parameter['parameters']
                  and math.isclose(float(chosen[0]['mean_CV_MAE']), min(float(r['mean_CV_MAE']) for r in rows), rel_tol=1e-10, abs_tol=1e-12)
                  and math.isclose(float(chosen[0]['mean_CV_MAE']), selected_parameter['mean_training_CV_MAE'], rel_tol=1e-10, abs_tol=1e-12))
    hash_path = LOCATIONS['data'] / 'package_input_hashes.json'
    if hash_path.is_file():
        hashes = json.loads(hash_path.read_text())
        for name, digest in hashes.items():
            relative = Path(name)
            prefix = Path('data/ai_optimization/benchmark')
            path = LOCATIONS['data'] / relative.relative_to(prefix)
            check('Packaged input SHA256: ' + name, hashlib.sha256(path.read_bytes()).hexdigest() == digest)
    result = dict(status='passed', checks=checks, algorithms=len(plan['models']), baselines=len(plan['baselines']),
                  unique_sequences=123, OOF_records=len(oof), repeated_training_records=len(train))
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            parser.error('Refusing to overwrite an existing validation file.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
