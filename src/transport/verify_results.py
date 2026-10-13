"""Compare a new transport run with the preserved reference numerical outputs."""
import argparse
import csv
import json
import math
from pathlib import Path

REL_TOL = 1e-9
ABS_TOL = 1e-10


def compare_value(old, new, location):
    if isinstance(old, dict):
        if not isinstance(new, dict) or old.keys() != new.keys():
            raise AssertionError(f'{location}: keys differ')
        for key in old:
            compare_value(old[key], new[key], f'{location}.{key}')
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            raise AssertionError(f'{location}: list length differs')
        for index, (left, right) in enumerate(zip(old, new)):
            compare_value(left, right, f'{location}[{index}]')
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        if not isinstance(new, (int, float)) or isinstance(new, bool):
            raise AssertionError(f'{location}: numeric type differs')
        if not math.isfinite(old) or not math.isfinite(new):
            raise AssertionError(f'{location}: non-finite number')
        if not math.isclose(old, new, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            raise AssertionError(f'{location}: number differs')
    elif old != new:
        raise AssertionError(f'{location}: value differs')


def main():
    root = Path(__file__).resolve().parents[2]
    reference = root / 'results' / 'transport'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dir', type=Path,
                        default=reference / 'generated')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    generated = args.generated_dir.resolve()
    if generated == reference.resolve():
        raise ValueError('Generated directory must differ from the packaged reference directory.')
    files = ['timeseries.csv', 'layer_retention.csv', 'depth_profiles.csv',
             'depth_time_heatmap.csv', 'uptake_sensitivity.csv']
    if args.report:
        report_path = args.report.resolve()
        protected = [root / name for name in ('data', 'models', 'logs', 'src')]
        protected.append(reference / 'images')
        run_outputs = files + ['summary.json', 'reference_inputs.json']
        if (report_path.parent == reference.resolve()
                or any(report_path.is_relative_to(path.resolve()) for path in protected)
                or report_path in {(generated / name).resolve() for name in run_outputs}):
            raise ValueError('Report path must preserve packaged files and generated numerical outputs.')
    records = []
    for filename in files:
        with (reference / filename).open(encoding='utf-8-sig', newline='') as a:
            reader = csv.DictReader(a)
            left_columns, left = reader.fieldnames, list(reader)
        with (generated / filename).open(encoding='utf-8-sig', newline='') as b:
            reader = csv.DictReader(b)
            right_columns, right = reader.fieldnames, list(reader)
        if not left or not right:
            raise AssertionError(f'{filename}: empty data table')
        if left_columns != right_columns:
            raise AssertionError(f'{filename}: column names or order differ')
        if len(left) != len(right):
            raise AssertionError(f'{filename}: row count differs')
        max_absolute_error = 0.0
        for row_number, (lrow, rrow) in enumerate(zip(left, right), start=2):
            if lrow.keys() != rrow.keys():
                raise AssertionError(f'{filename}: column names differ')
            for key in lrow:
                try:
                    old, new = float(lrow[key]), float(rrow[key])
                except (TypeError, ValueError):
                    if lrow[key] != rrow[key]:
                        raise AssertionError(f'{filename}: text field {key} differs')
                    continue
                compare_value(old, new, f'{filename}:{row_number}:{key}')
                max_absolute_error = max(max_absolute_error, abs(old-new))
        records.append({'file': filename, 'rows': len(left),
                        'maximum_absolute_error': max_absolute_error})
    summary = json.loads((generated/'summary.json').read_text(encoding='utf-8'))
    original = json.loads((reference/'summary.json').read_text(encoding='utf-8'))
    # Residual magnitudes can vary with solver builds while remaining acceptable.
    compare_value({k: v for k, v in original.items() if k != 'checks'},
                  {k: v for k, v in summary.items() if k != 'checks'}, 'summary.json')
    compare_value(json.loads((root/'data'/'transport'/'reference_inputs.json').read_text(encoding='utf-8')),
                  json.loads((generated/'reference_inputs.json').read_text(encoding='utf-8')),
                  'reference_inputs.json')
    checks = summary['checks']
    if set(checks) != {'mass_relative', 'min_state', 'input_after_removal'}:
        raise AssertionError('summary.json: physical check keys differ')
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in checks.values()):
        raise AssertionError('summary.json: physical checks must be finite numbers')
    if not (0 <= checks['mass_relative'] < 1e-7
            and checks['min_state'] > -1e-7
            and 0 <= checks['input_after_removal'] < 1e-8):
        raise AssertionError('summary.json: physical checks failed')
    report = {'status': 'passed', 'reference_directory': 'results/transport',
              'generated_directory': str(generated),
              'tolerances': {'relative': REL_TOL, 'absolute': ABS_TOL},
              'numerical_comparison': records, 'summary_comparison': 'passed',
              'input_comparison': 'passed', 'physical_checks': checks}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
