"""Seven regressors, five outer count-stratified folds and inner group CV.

User-authorized rerun; original files preserved. All inputs have previously been
analysed, so this is within-library development evaluation, not new confirmation.
Frozen encoder and local pretrained TabPFN; no docking or MD or model downloads.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import warnings

from layout import PROJECT_ROOT, DATA as DEFAULT_DATA, MODELS as DEFAULT_MODELS, RESULTS as DEFAULT_RESULTS, LOGS as DEFAULT_LOGS
PACKAGE_ROOT = PROJECT_ROOT
OUT = DEFAULT_RESULTS / 'runs' / 'retrained_5fold'
DATA = DEFAULT_DATA
MODEL_OUT = DEFAULT_MODELS / 'runs' / 'retrained_5fold'
LOG_OUT = DEFAULT_LOGS / 'runs' / 'retrained_5fold'
CHECKPOINT_FILE = None
RUN_STARTED = None
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

SEED = 20261003
MODELS = ['MLP', 'SVR', 'Ridge', 'Random forest', 'ExtraTrees', 'GPR', 'TabPFN']
BASELINES = ['Training-median baseline', 'One-hot Ridge baseline']
PREFIX = 'regressor__model__'


def log(value):
    event = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic() - RUN_STARTED, **value)
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False)
    print(payload, flush=True)
    with (LOG_OUT / 'training_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(payload + '\n')


def runtime_info():
    dependencies = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'torch', 'tabpfn', 'threadpoolctl'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return dict(python=platform.python_version(), system=platform.system(),
                machine=platform.machine(), cpu_description=platform.processor() or 'not reported',
                logical_cpus=os.cpu_count(), execution_device='CPU',
                threadpool_limit=1, TabPFN_torch_threads=2, dependencies=dependencies)


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    assert not path.exists(), f'Preserve existing output: {path}'
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def table(path, rows):
    assert rows and not path.exists(), f'Preserve existing output: {path}'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, p):
    p = np.maximum(p, 0)
    b = (y >= 1).astype(int)
    cm = confusion_matrix(b, p >= .5, labels=[0, 1])
    integer = np.floor(p + .5).astype(int)
    mse = float(mean_squared_error(y, p))
    return dict(n=len(y), AUC=float(roc_auc_score(b, p)), MAE=float(mean_absolute_error(y, p)),
                MSE=mse, RMSE=float(np.sqrt(mse)), TN=int(cm[0, 0]), FP=int(cm[0, 1]),
                FN=int(cm[1, 0]), TP=int(cm[1, 1]), integer_MAE=float(mean_absolute_error(y, integer)),
                exact_count_fraction=float(np.mean(y == integer)))


def score(estimator, features, actual):
    return -mean_absolute_error(actual, np.maximum(estimator.predict(features), 0))


def tabpfn():
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=str(CHECKPOINT_FILE),
        device='cpu', n_estimators=4, n_preprocessing_jobs=1, random_state=SEED)


def onehot(sequence):
    rc = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    return [float(base == letter) for base in min(sequence, rc) for letter in 'ACGT']


def main():
    global OUT, DATA, MODEL_OUT, LOG_OUT, CHECKPOINT_FILE, MODELS, RUN_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA,
                        help='Directory containing labels, features and fold assignments.')
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='New run directory; existing runs are never overwritten.')
    parser.add_argument('--model-dir', type=Path, help='New model directory; defaults to models/ai_optimization/benchmark/runs/<output-name>.')
    parser.add_argument('--log-dir', type=Path, help='New log directory; defaults to logs/ai_optimization/benchmark/runs/<output-name>.')
    parser.add_argument('--tabpfn-checkpoint', type=Path,
                        help='User-provided TabPFN V2 regressor checkpoint; not redistributed here.')
    parser.add_argument('--skip-tabpfn', action='store_true',
                        help='Explicit six-model run when no licensed checkpoint is available.')
    parser.add_argument('--check-inputs', action='store_true',
                        help='Validate packaged labels/features/folds and print the protocol; do not fit.')
    args = parser.parse_args()
    RUN_STARTED = time.monotonic()
    DATA, OUT = args.data_dir.resolve(), args.output_dir.resolve()
    MODEL_OUT = args.model_dir.resolve() if args.model_dir else DEFAULT_MODELS / 'runs' / OUT.name
    LOG_OUT = args.log_dir.resolve() if args.log_dir else DEFAULT_LOGS / 'runs' / OUT.name
    CHECKPOINT_FILE = args.tabpfn_checkpoint.resolve() if args.tabpfn_checkpoint else None
    if args.skip_tabpfn:
        MODELS = [name for name in MODELS if name != 'TabPFN']
    if not args.check_inputs and 'TabPFN' in MODELS and not (CHECKPOINT_FILE and CHECKPOINT_FILE.is_file()):
        parser.error('Supply --tabpfn-checkpoint to reproduce all seven models; or explicitly use --skip-tabpfn.')
    if not args.check_inputs and any(path.exists() for path in (OUT, MODEL_OUT, LOG_OUT)):
        parser.error('Output directory already exists; choose a new directory.')
    if not args.check_inputs and len({OUT, MODEL_OUT, LOG_OUT}) != 3:
        parser.error('Results, model and log directories must be distinct.')
    if CHECKPOINT_FILE:
        os.environ['TABPFN_MODEL_CACHE_DIR'] = str(CHECKPOINT_FILE.parent)
    paths = [DATA / 'uniform_single_seed_labels.csv', DATA / 'hyena_features.npz',
             DATA / 'outer_assignments.csv']
    hashes = {path.name: sha(path) for path in paths}
    if CHECKPOINT_FILE:
        hashes['tabpfn_checkpoint'] = sha(CHECKPOINT_FILE)
    labels, originals = read(paths[0]), read(paths[2])
    index = {row['sequence_id']: row for row in labels}
    ordered = [index[row['sequence_id']] for row in originals]
    assert len(labels) == len(index) == len(originals) == 123, 'Duplicate or missing label IDs.'
    assert len({row['sequence_id'] for row in originals}) == 123, 'Duplicate fold assignment IDs.'
    assert all(len(row['sequence_20nt']) == 20 and set(row['sequence_20nt']) <= set('ACGT')
               and row['family_id'] and row['docking_seed'] == '104729' for row in labels), 'Invalid sequence, family or label seed.'
    assert all(all(a[key] == b[key] for key in ('sequence_20nt', 'family_id', 'observed_count'))
               for a, b in zip(ordered, originals)), 'Labels and fold metadata disagree.'
    canonical = [min(row['sequence_20nt'], row['sequence_20nt'].translate(str.maketrans('ACGT', 'TGCA'))[::-1])
                 for row in ordered]
    assert len(set(canonical)) == 123
    with np.load(paths[1], allow_pickle=False) as cached:
        X = cached['X'].astype(np.float64)
        assert X.shape == (123, 128) and np.isfinite(X).all()
        assert np.array_equal(cached['y'], [int(r['observed_count']) for r in ordered])
        assert list(cached['sequence_id']) == [r['sequence_id'] for r in ordered]
    y = np.array([int(row['observed_count']) for row in ordered])
    assert Counter(map(int, y)) == {0: 31, 1: 90, 2: 2}, 'Labels differ from the frozen benchmark.'
    family = np.array([row['family_id'] for row in ordered])
    saved_inner = read(DATA / 'training_group_CV_assignments.csv')
    saved_keys = {(int(r['outer_fold']), r['sequence_id']) for r in saved_inner}
    assert len(saved_inner) == len(saved_keys) == 492, 'Duplicate or missing inner assignments.'
    assert all(int(r['outer_fold']) in range(1, 6) and int(r['training_CV_fold']) in range(1, 4)
               and r['sequence_id'] in index and r['family_id'] == index[r['sequence_id']]['family_id']
               for r in saved_inner), 'Invalid inner assignment metadata.'
    with warnings.catch_warnings(record=True) as split_warnings:
        warnings.simplefilter('always')
        outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    outer_ids = np.zeros(len(y), dtype=int)
    inner, summary, CV_assignments = {}, [], []
    for f, (train, test) in enumerate(outer, 1):
        outer_ids[test] = f
        assert not set(train).intersection(test)
        assert {0, 1}.issubset(y[train]) and {0, 1}.issubset(y[test])
        # Reuse actual group-fold memberships. GroupKFold tie ordering can change
        # with anonymized family names or dependency versions.
        assigned = {r['sequence_id']: int(r['training_CV_fold']) for r in saved_inner if int(r['outer_fold']) == f}
        assert set(assigned) == {ordered[i]['sequence_id'] for i in train}, 'Inner assignments do not cover the outer training set.'
        validation_fold = np.array([assigned[ordered[i]['sequence_id']] for i in train])
        folds = [(np.flatnonzero(validation_fold != k), np.flatnonzero(validation_fold == k)) for k in range(1, 4)]
        inner[f] = folds
        group_summary = []
        for k, (a, b) in enumerate(folds, 1):
            assert not set(family[train[a]]).intersection(family[train[b]]) and len(a) > 8 and len(b) > 0
            group_summary.append(dict(inner_fold=k, training_n=len(a), validation_n=len(b), family_overlap=0,
                training_labels=dict(Counter(map(int, y[train[a]]))),
                validation_labels=dict(Counter(map(int, y[train[b]])))))
            CV_assignments.extend(dict(sequence_id=ordered[train[i]]['sequence_id'], outer_fold=f,
                training_CV_fold=k, family_id=str(family[train[i]])) for i in b)
        summary.append(dict(outer_fold=f, training_n=len(train), test_n=len(test),
            training_labels=dict(Counter(map(int, y[train]))), test_labels=dict(Counter(map(int, y[test]))),
            outer_family_overlap=len(set(family[train]).intersection(family[test])), inner_folds=group_summary))
    assert all(outer_ids > 0) and len(set(outer_ids[y == 2])) == 2
    assert np.array_equal(outer_ids, [int(r['outer_fold']) for r in originals]), 'Saved outer folds differ from fixed protocol.'
    inner_key = lambda row: (row['sequence_id'], int(row['outer_fold']),
                              int(row['training_CV_fold']), row['family_id'])
    assert len(saved_inner) == len(CV_assignments) and set(map(inner_key, saved_inner)) == set(map(inner_key, CV_assignments)), 'Saved inner folds differ from fixed protocol.'
    configs = {
        'MLP': (MLPRegressor(solver='lbfgs', max_iter=2000, tol=1e-5, random_state=SEED),
                dict(hidden_layer_sizes=[(2,), (4,), (8,)], alpha=[1, 10, 100])),
        'SVR': (SVR(kernel='rbf'), dict(C=[.5, 2, 10], epsilon=[.05, .15])),
        'Ridge': (Ridge(), dict(alpha=[1, 10, 100, 1000])),
        'Random forest': (RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=SEED),
                          dict(max_depth=[3, 6, None], min_samples_leaf=[3, 6], max_features=[.5, 1.])),
        'ExtraTrees': (ExtraTreesRegressor(n_estimators=200, n_jobs=1, random_state=SEED),
                       dict(max_depth=[3, 6, None], min_samples_leaf=[3, 6], max_features=[.5, 1.])),
        'GPR': (GaussianProcessRegressor(kernel=ConstantKernel(1., (1e-2, 1e2)) * RBF(10., (.1, 1e3))
                  + WhiteKernel(.1, (.01, 1.)), n_restarts_optimizer=0, random_state=SEED),
                dict(alpha=[.01, .1, 1.])),
        'TabPFN': (None, {}),
        'One-hot Ridge baseline': (Ridge(), dict(alpha=[1, 10, 100, 1000]))}
    plan = dict(models=MODELS, baselines=BASELINES, seed=SEED,
        split='One fixed 5-fold shuffled outer CV, stratified on exact count labels 0, 1, 2',
        full_count_labels=dict(Counter(map(int, y))),
        scope='Within the existing sequence library; previously analysed data, development evaluation',
        target='Uniform single-seed local sampled tight docking-site count, Vina <= -8.0 kcal/mol',
        preprocessing='Variance filter, StandardScaler and fixed PCA8 fitted within each training fold; target scaler likewise',
        tuning='3 saved family-grouped inner folds inside each outer training set; minimum mean continuous nonnegative count MAE',
        outer_and_inner_folds=summary, pca_components=8, classification_threshold=.5, frozen_HyenaDNA_encoder=True,
        grids={name: grid for name, (_, grid) in configs.items()},
        TabPFN='Offline V2, pretrained weights frozen; fit uses training context, no gradient fine tuning',
        limitations=['Related families can cross the outer split; no unseen-family generalization claim',
                     'Only two count-2 examples; they occur in two distinct outer test folds, not all five',
                     'Three unequal inner folds; their unweighted mean selects parameters',
                     'Outer folds are newly assigned but all 123 labels have already been analysed',
                     'Training-fit diagnostic averages the four in-sample outer-model predictions per sequence',
                     'Inner group CV estimates a different, harder transfer task than the within-library outer split',
                     'A local docking-site count is not docking energy or measured tFNA loading capacity'],
        no_new_docking=True, no_new_MD=True)
    if args.check_inputs:
        print(json.dumps(dict(stage='inputs_validated_no_training', X_shape=list(X.shape), **plan), indent=2))
        return
    # Complete input checks before creating any run directories.
    OUT.mkdir(parents=True, exist_ok=False)
    MODEL_OUT.mkdir(parents=True, exist_ok=False)
    LOG_OUT.mkdir(parents=True, exist_ok=False)
    dump(LOG_OUT / 'analysis_plan.json', plan)
    dump(LOG_OUT / 'source_hashes.json', hashes)
    assignment = [dict(sequence_id=row['sequence_id'], sequence_20nt=row['sequence_20nt'],
        family_id=row['family_id'], observed_count=int(y[i]), outer_fold=int(outer_ids[i]))
        for i, row in enumerate(ordered)]
    table(LOG_OUT / 'outer_assignments.csv', assignment)
    table(LOG_OUT / 'training_group_CV_assignments.csv', CV_assignments)
    log(dict(stage='plan_frozen', **plan))
    dump(LOG_OUT / 'runtime_environment.json', runtime_info())
    H = np.array([onehot(row['sequence_20nt']) for row in ordered])
    results, predictions, fold_results, outer_predictions, train_predictions, parameters, candidates = [], [], [], [], [], [], []
    warning_log = [dict(model='outer_split', category=w.category.__name__, message=str(w.message)) for w in split_warnings]
    with threadpool_limits(limits=1):
        for name in MODELS + BASELINES:
            log(dict(stage='model_start', model=name))
            features = H if name == 'One-hot Ridge baseline' else X
            predicted = np.full(len(y), np.nan)
            training_sum, training_n = np.zeros(len(y)), np.zeros(len(y), dtype=int)
            for f, (train, test) in enumerate(outer, 1):
                folds = inner[f]
                if name == 'Training-median baseline':
                    fitted_prediction = np.full(len(y), np.median(y[train]))
                    parameters.append(dict(model=name, outer_fold=f,
                        parameters={'training_median': float(np.median(y[train]))}, selection='Training data only; no grid'))
                else:
                    estimator, grid = configs[name]
                    if name == 'TabPFN':
                        estimator = tabpfn()
                    stages = [('variance', VarianceThreshold()), ('scale', StandardScaler())]
                    if name != 'One-hot Ridge baseline':
                        stages.append(('pca', PCA(n_components=8, svd_solver='full')))
                    pipeline = Pipeline(stages + [('model', clone(estimator))])
                    target = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
                    search = GridSearchCV(target, {PREFIX + key: value for key, value in grid.items()},
                        scoring=score, cv=folds, n_jobs=1, error_score='raise')
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        search.fit(features[train], y[train])
                        fitted_prediction = np.maximum(search.predict(features), 0)
                    warning_log.extend(dict(model=name, outer_fold=f, category=w.category.__name__, message=str(w.message)) for w in caught)
                    parameter = dict(model=name, outer_fold=f, parameters=search.best_params_,
                        mean_training_CV_MAE=-float(search.best_score_), selection='Minimum mean continuous inner CV MAE')
                    if name != 'One-hot Ridge baseline':
                        pca = search.best_estimator_.regressor_.named_steps['pca']
                        parameter['PCA_training_explained_variance_sum'] = float(pca.explained_variance_ratio_.sum())
                    parameters.append(parameter)
                    for c, values in enumerate(search.cv_results_['params']):
                        candidates.append(dict(model=name, outer_fold=f, candidate=c, parameters=json.dumps(values),
                            mean_CV_MAE=-float(search.cv_results_['mean_test_score'][c]),
                            fold_MAE_1=-float(search.cv_results_['split0_test_score'][c]),
                            fold_MAE_2=-float(search.cv_results_['split1_test_score'][c]),
                            fold_MAE_3=-float(search.cv_results_['split2_test_score'][c]), selected=c == search.best_index_))
                    stem = name.lower().replace(' ', '_').replace('-', '_')
                    if name != 'TabPFN':
                        joblib.dump(dict(estimator=search.best_estimator_, model=name, outer_fold=f, target=plan['target'],
                            feature_representation='one-hot canonical 20nt' if name in BASELINES else 'frozen HyenaDNA then training PCA8',
                            train_sequence_ids=[ordered[i]['sequence_id'] for i in train]),
                            MODEL_OUT / (stem + f'_fold{f}.joblib'))
                    else:
                        np.savez_compressed(MODEL_OUT / f'tabpfn_training_context_fold{f}.npz', X=X[train], y=y[train])
                predicted[test] = fitted_prediction[test]
                training_sum[train] += fitted_prediction[train]
                training_n[train] += 1
                outer_predictions.extend(dict(model=name, outer_fold=f, sequence_id=ordered[i]['sequence_id'],
                    observed_count=int(y[i]), observed_positive=int(y[i] >= 1), predicted_score=float(fitted_prediction[i]),
                    predicted_positive=int(fitted_prediction[i] >= .5)) for i in test)
                train_predictions.extend(dict(model=name, outer_fold=f, sequence_id=ordered[i]['sequence_id'],
                    observed_count=int(y[i]), predicted_score=float(fitted_prediction[i])) for i in train)
                for split, ids in (('train', train), ('test', test)):
                    fold_results.append(dict(model=name, outer_fold=f, split=split,
                                             **metrics(y[ids], fitted_prediction[ids])))
                log(dict(stage='outer_fold_done', model=name, outer_fold=f,
                         **metrics(y[test], fitted_prediction[test])))
            assert np.isfinite(predicted).all() and np.all(training_n == 4)
            training_mean = training_sum / training_n
            for split, values in (('train', training_mean), ('test', predicted)):
                result = dict(model=name, split=split, **metrics(y, values))
                results.append(result)
                predictions.extend(dict(model=name, split=split, sequence_id=ordered[i]['sequence_id'],
                    observed_count=int(y[i]), observed_positive=int(y[i] >= 1), predicted_score=float(value),
                    predicted_positive=int(value >= .5)) for i, value in enumerate(values))
                log(dict(stage='model_done', **result))
            # Unique partial output names allow diagnosis without overwriting a finished result.
            table(OUT / (name.lower().replace(' ', '_').replace('-', '_') + '_completed.csv'),
                  [r for r in results if r['model'] == name])
    table(OUT / 'metrics.csv', results)
    table(OUT / 'predictions.csv', predictions)
    table(OUT / 'fold_metrics.csv', fold_results)
    table(OUT / 'outer_test_predictions.csv', outer_predictions)
    table(OUT / 'training_fit_predictions.csv', train_predictions)
    table(LOG_OUT / 'training_CV_candidate_losses.csv', candidates)
    dump(LOG_OUT / 'training_only_parameters.json', parameters)
    dump(LOG_OUT / 'warnings.json', warning_log)
    unchanged = all(sha(path) == hashes[path.name] for path in paths)
    if CHECKPOINT_FILE:
        unchanged = unchanged and sha(CHECKPOINT_FILE) == hashes['tabpfn_checkpoint']
    assert unchanged
    dump(LOG_OUT / 'report.json', dict(plan=plan, metrics=results,
         source_hashes_unchanged=unchanged, sklearn=sklearn_version, prediction_records=len(predictions),
         selection_boundary='Hyperparameters selected by training CV only; holdout metrics not used for refit',
         TabPFN_attribution='Built with PriorLabs-TabPFN',
         runtime=runtime_info(), elapsed_seconds=time.monotonic() - RUN_STARTED))
    log(dict(stage='complete', metrics=results, source_hashes_unchanged=unchanged))


if __name__ == '__main__':
    main()
