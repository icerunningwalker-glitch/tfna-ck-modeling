"""Reproduce saved non-TabPFN OOF predictions without fitting any model.

Only load the trusted model snapshots distributed with this package. A joblib
file is executable pickle data; never use an untrusted model file.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import warnings
import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn import __version__ as sklearn_version

from layout import PROJECT_ROOT, locations
ROOT = PROJECT_ROOT
MODELS = ['MLP', 'SVR', 'Ridge', 'Random forest', 'ExtraTrees', 'GPR', 'One-hot Ridge baseline']


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def onehot(sequence):
    reverse = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    return [float(base == letter) for base in min(sequence, reverse) for letter in 'ACGT']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Optional new JSON file; existing output is never overwritten.')
    parser.add_argument('--data-dir', type=Path, help='Override benchmark input directory.')
    parser.add_argument('--results-dir', type=Path, help='Override saved/new-run results directory.')
    parser.add_argument('--model-dir', type=Path, help='Override fitted snapshot directory.')
    args = parser.parse_args()
    root = args.root.resolve()
    loc = locations(root)
    for key in ('data', 'results', 'model'):
        value = getattr(args, key + '_dir')
        if value:
            loc['models' if key == 'model' else key] = value.resolve()
    assignments = read(loc['data'] / 'outer_assignments.csv')
    ids = [r['sequence_id'] for r in assignments]
    folds = np.array([int(r['outer_fold']) for r in assignments])
    assert len(ids) == len(set(ids)) == 123 and set(folds) == set(range(1, 6)), 'Invalid or duplicated assignments.'
    with np.load(loc['data'] / 'hyena_features.npz', allow_pickle=False) as cache:
        X = cache['X'].astype(np.float64)
        assert list(cache['sequence_id']) == ids
        assert X.shape == (123, 128) and np.isfinite(X).all(), 'Invalid features.'
    H = np.array([onehot(r['sequence_20nt']) for r in assignments])
    expected = {(r['model'], r['sequence_id']):float(r['predicted_score'])
                for r in read(loc['results'] / 'outer_test_predictions.csv')}
    checks, version_warnings = [], []
    for model in MODELS:
        features = H if model == 'One-hot Ridge baseline' else X
        stem = model.lower().replace(' ', '_').replace('-', '_')
        for fold in range(1, 6):
            path = loc['models'] / f'{stem}_fold{fold}.joblib'
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', InconsistentVersionWarning)
                snapshot = joblib.load(path)
            counts = Counter((w.category.__name__, str(w.message)) for w in caught)
            version_warnings.extend(dict(model=model, fold=fold, category=category, message=message, count=count)
                                    for (category, message), count in counts.items())
            assert snapshot['model'] == model and snapshot['outer_fold'] == fold
            indexes = np.flatnonzero(folds == fold)
            expected_train = {ids[i] for i in np.flatnonzero(folds != fold)}
            assert len(snapshot['train_sequence_ids']) == len(expected_train) and set(snapshot['train_sequence_ids']) == expected_train, 'Snapshot training IDs differ from the fold complement.'
            actual = np.maximum(snapshot['estimator'].predict(features[indexes]), 0)
            reference = np.array([expected[(model, ids[i])] for i in indexes])
            assert np.allclose(actual, reference, rtol=1e-10, atol=1e-12), (model, fold)
            checks.append(dict(model=model, fold=fold, n=len(indexes),
                               maximum_absolute_difference=float(np.max(np.abs(actual-reference)))))
    result = dict(status='passed', models=len(MODELS), model_snapshots=len(checks),
                  checked_OOF_predictions=sum(r['n'] for r in checks), no_training=True,
                  TabPFN='Saved prediction arithmetic is checked by verify_all7.py; checkpoint inference not executed.',
                  runtime_sklearn=sklearn_version, deserialization_warnings=version_warnings,
                  deserialization_warning_count=sum(r['count'] for r in version_warnings),
                  checks=checks)
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            parser.error('Refusing to overwrite an existing output.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks', 'deserialization_warnings')}))


if __name__ == '__main__':
    main()
