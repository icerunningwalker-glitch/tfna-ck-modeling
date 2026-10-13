"""One-command prepared-PDBQT screening, resumable under a frozen protocol."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from bootstrap_screen import build_boxes
from execute_pipeline import freeze_eligible_manifest
from finalize_report import finalize
from run_screen import ProcessLock, main as run_screen, atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--receptor',type=Path,required=True)
    parser.add_argument('--vina',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--cpu-per-job',type=int,default=4)
    parser.add_argument('--exhaustiveness',type=int,default=8)
    parser.add_argument('--seed',type=int,default=104729)
    parser.add_argument('--margin',type=float,default=8.0)
    parser.add_argument('--expected-ligands',type=int)
    parser.add_argument('--reference-id')
    parser.add_argument('--quality-annotations',type=Path)
    parser.add_argument('--sequence-config',type=Path)
    parser.add_argument('--track',default='unspecified')
    parser.add_argument('--limit-jobs',type=int)
    parser.add_argument('--execute',action='store_true',help='Required to create outputs and start docking; otherwise check input files only')
    args=parser.parse_args(argv)
    for key in ('manifest','receptor','vina'):
        path=getattr(args,key).resolve()
        if not path.is_file():parser.error('Missing --'+key+' input')
        setattr(args,key,path)
    if args.workers<1 or args.cpu_per_job<1 or args.exhaustiveness<1:parser.error('Positive resource parameters required')
    with args.manifest.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
    if not rows:parser.error('Empty ligand manifest')
    if not args.execute:
        print(json.dumps(dict(mode='input_check_only',source_records=len(rows),boxes=8,
                              note='No files changed and no Vina jobs launched. Add --execute to run.')))
        return 0
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    with ProcessLock(root/'pipeline.lock'):
        build_boxes(args.receptor,root,args.margin,args.sequence_config)
        manifest,coverage=freeze_eligible_manifest(args.manifest,root,args.expected_ligands)
        if args.quality_annotations:
            source=args.quality_annotations.resolve();target=root/'ligand_quality_annotations.csv'
            if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Quality annotations changed; choose a new run directory')
            if not target.exists():shutil.copyfile(source,target)
        command=['--ligand-manifest',str(manifest),'--site-jobs',str(root/'site_jobs.csv'),
                 '--vina',str(args.vina),'--output-dir',str(root/'screen'),
                 '--workers',str(args.workers),'--cpu-per-job',str(args.cpu_per_job),
                 '--exhaustiveness',str(args.exhaustiveness),'--seed',str(args.seed),
                 '--expected-ligands',str(coverage['prepared_ligands'])]
        if args.limit_jobs:command+=['--limit-jobs',str(args.limit_jobs)]
        code=run_screen(command)
        state=json.loads((root/'screen'/'progress.json').read_text(encoding='utf-8'))
        if state['status'] in ('completed','completed_with_failures'):
            finalize(root,args.reference_id,args.track)
        else:print('Partial/preflight run only; no final candidate list has been claimed.')
        return code


if __name__=='__main__':raise SystemExit(main())
