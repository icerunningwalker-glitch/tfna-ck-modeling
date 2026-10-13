"""Freeze ligand eligibility while preserving preparation exclusions."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from run_screen import atomic_json


def freeze_eligible_manifest(source, output_dir, expected_ligands=None):
    source=Path(source).resolve();output_dir=Path(output_dir).resolve()
    with source.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    if fields is None or not {'identifier','name','pdbqt','status'}.issubset(fields):
        raise ValueError('Manifest requires identifier, name, pdbqt and status columns')
    if expected_ligands is None:expected_ligands=len(rows)
    if expected_ligands<1 or len(rows)!=expected_ligands or len({row['identifier'].casefold() for row in rows})!=expected_ligands:
        raise ValueError('Preparation manifest must retain all unique source catalog IDs')
    if any(row['status'] not in ('prepared','reused','failed','not_prepared') for row in rows):
        raise ValueError('Unexpected preparation status; refusing a stale manifest')
    eligible=[dict(row) for row in rows if row['status'] in ('prepared','reused')]
    excluded=[dict(row) for row in rows if row['status'] not in ('prepared','reused')]
    if not eligible:raise ValueError('No prepared ligand can enter the screen')
    # Rebase relative paths when the manifest is copied into a new directory.
    for row in eligible:
        path=Path(row['pdbqt'])
        if not path.is_absolute():path=source.parent/path
        try:row['pdbqt']=os.path.relpath(path.resolve(),output_dir)
        except ValueError:row['pdbqt']=str(path.resolve())
    def freeze_csv(path, selected):
        buffer=io.StringIO(newline='')
        writer=csv.DictWriter(buffer,fieldnames=fields)
        writer.writeheader();writer.writerows(selected)
        payload=buffer.getvalue().encode('utf-8-sig')
        if path.exists() and path.read_bytes()!=payload:
            raise ValueError('Frozen screening manifest changed; use a new output directory')
        if not path.exists():path.write_bytes(payload)
    output_dir.mkdir(parents=True,exist_ok=True)
    prepared=output_dir/'prepared_ligand_manifest.csv'
    freeze_csv(prepared,eligible)
    freeze_csv(output_dir/'preparation_exclusions.csv',excluded)
    evidence=dict(requested_ligands=expected_ligands,prepared_ligands=len(eligible),
                  preparation_exclusions=len(excluded),requested_jobs=expected_ligands*8,
                  docking_jobs=len(eligible)*8,source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  excluded_ids=[row['identifier'] for row in excluded],
                  interpretation='Preparation failures remain explicit and unranked, never counted as negative hits.')
    atomic_json(output_dir/'preparation_coverage.json',evidence)
    return prepared,evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-ligands',type=int)
    args=parser.parse_args(argv)
    _,result=freeze_eligible_manifest(args.source_manifest,args.output_dir,args.expected_ligands)
    print(json.dumps(result))


if __name__=='__main__':main()
