"""Export full-coverage rankings, quality annotations and candidate structures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from run_screen import atomic_csv, atomic_json, utc_now


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def join_rankings(rows,catalog,quality):
    joined=[]
    for row in rows:
        if int(row['successful_boxes'])!=8 or str(row['complete']).lower()!='true':
            raise ValueError('A final ranking may contain only eight-box-complete ligands')
        ident=row['ligand_id']
        if ident not in catalog or ident not in quality:
            raise ValueError('Missing source or conformer annotation for '+ident)
        metadata=catalog[ident]
        annotations={key:value for key,value in quality[ident].items() if key not in ('identifier','name')}
        joined.append(dict(row,cas=metadata.get('cas',''),formula=metadata.get('formula',''),
                           molecular_weight=metadata.get('molecular_weight',''),**annotations))
    return joined


def export_pose_pdb(source, target):
    """Coordinate-only first-mode PDB; original scored PDBQT is kept separately."""
    atoms=[]
    types={'A':'C','C':'C','N':'N','NA':'N','NS':'N','OA':'O','OS':'O','SA':'S','S':'S',
           'P':'P','F':'F','Cl':'CL','Br':'BR','I':'I','HD':'H','HS':'H','H':'H'}
    for line in Path(source).read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('ENDMDL'):break
        if line.startswith(('ATOM  ','HETATM')):
            element=types.get(line[77:].strip(),'')
            atoms.append(line[:66].ljust(76)+f'{element:>2}')
    if not atoms:raise ValueError('No first-mode coordinates to export')
    target.write_text('\n'.join(atoms)+'\nEND\n',encoding='ascii')


def finalize(root, reference_id=None, track='unspecified'):
    root=Path(root).resolve();screen=root/'screen'
    synthetic = (root/'SYNTHETIC_SOFTWARE_TEST.json').is_file() or track=='synthetic-demo'
    progress=json.loads((screen/'progress.json').read_text(encoding='utf-8'))
    if progress['status'] not in ('completed','completed_with_failures'):
        raise ValueError('Screen is not terminal; refusing to label a partial ranking final')
    protocol=json.loads((screen/'protocol.json').read_text(encoding='utf-8'))
    if progress['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('Protocol mismatch')
    source_rows=read_rows(root/'prepared_ligand_manifest.csv')
    catalog={row['identifier']:row for row in source_rows}
    quality_path=root/'ligand_quality_annotations.csv'
    if quality_path.exists():
        quality_rows=read_rows(quality_path)
        quality={row['identifier']:row for row in quality_rows}
        if len(quality)!=len(quality_rows):raise ValueError('Duplicate quality annotation IDs')
    else:
        quality={ident:dict(identifier=ident,quality_annotation='not_supplied') for ident in catalog}
    if len(catalog)!=protocol['identity']['ligand_count'] or set(catalog)!=set(quality):
        raise ValueError('Catalog and quality annotations must exactly cover screened ligands')
    primary=join_rankings(read_rows(screen/'ranking_primary_complete.csv'),catalog,quality)
    best=join_rankings(read_rows(screen/'ranking_best_score_complete.csv'),catalog,quality)
    for filename,rows in (('ranking_strong_box_fraction_annotated.csv',primary),('ranking_best_score_annotated.csv',best)):
        fields=list(dict.fromkeys(key for row in rows for key in row)) or ['rank','ligand_id','name']
        atomic_csv(root/filename,fields,rows)
    eligibility=json.loads((root/'preparation_coverage.json').read_text(encoding='utf-8'))
    result=dict(status=progress['status'],generated_at=utc_now(),requested_ligands=eligibility['requested_ligands'],
                preparation_exclusions=eligibility['preparation_exclusions'],screened_ligands=len(catalog),
                ranked_full_coverage_ligands=len(primary),completed_jobs=progress['terminal_jobs'],
                successful_jobs=progress['success'],failed_jobs=progress['failed'],total_jobs=progress['total_jobs'],
                protocol_sha256=protocol['protocol_sha256'],reference_id=reference_id,
                reference_primary=next((row for row in primary if row['ligand_id']==reference_id),None),
                reference_best=next((row for row in best if row['ligand_id']==reference_id),None),
                predicted_hit_threshold_kcal_mol=-8.0,
                interpretation='Predicted hits only. Overlapping boxes are not independent sites or loading capacity.',
                quality_annotations_supplied=quality_path.exists(),
                evidence_type='synthetic_software_test' if synthetic else 'computed_docking',
                scientific_candidate=not synthetic)
    atomic_json(root/'ranking_summary.json',result)
    states={}
    for row in read_rows(screen/'current_runs.csv'):
        if row['status']=='success':states.setdefault(row['ligand_id'],[]).append(row)
    destination=root/'results';structures=destination/'structures'
    structures.mkdir(parents=True,exist_ok=True)
    candidates=[]
    for row in primary:
        ident=row['ligand_id'];state=min(states[ident],key=lambda x:float(x['best_score']))
        original=Path(state['pose_file'])
        if not original.is_absolute():original=screen/original
        snapshot=structures/(ident+'.pdb')
        export_pose_pdb(original,snapshot)
        warning=quality[ident].get('geometrywarnings',quality[ident].get('geometry_warning_flags',''))
        candidates.append(dict(candidate_id=ident,track=track,smiles=catalog[ident].get('smiles',''),
                               structure_file='structures/'+snapshot.name,rank=row['rank'],
                               best_vina_score_kcal_mol=row['best_score'],
                               predicted_hit_box_count_8=row['strong_box_count_le_minus8'],
                               predicted_hit_box_fraction_8=row['strong_box_fraction_le_minus8'],
                               model_version='AutoDock Vina; protocol '+protocol['identity']['vina']['sha256'],
                               run_version='portable-screening-1.0',protocol_sha256=protocol['protocol_sha256'],
                               evidence_type='synthetic_software_test' if synthetic else 'computed_docking',
                               scientific_candidate=not synthetic,
                               notes='Predicted hit, not experimental binding; quality='+str(warning or quality[ident].get('quality_annotation','see annotations'))))
    fields=list(candidates[0]) if candidates else ['candidate_id','track','smiles','structure_file','rank','best_vina_score_kcal_mol','predicted_hit_box_count_8','predicted_hit_box_fraction_8','model_version','run_version','protocol_sha256','notes']
    export_name='synthetic_results.csv' if synthetic else 'results.csv'
    if synthetic and (destination/'results.csv').exists():
        raise ValueError('A synthetic run must never contain a scientific results.csv candidate table')
    atomic_csv(destination/export_name,fields,candidates)
    lines=['Eight-box single-seed virtual screening',
           f"Prepared ligands: {len(catalog)}; excluded during preparation: {result['preparation_exclusions']}",
           f"Terminal jobs: {result['completed_jobs']}/{result['total_jobs']}; successful: {result['successful_jobs']}; failed: {result['failed_jobs']}",
           'Primary: score <= -8 kcal/mol box fraction, then best score, then mean score, then identifier.',
           'Secondary: best score, then mean score, then identifier.',
           'Missing/failed jobs are not negative hits; only complete eight-box ligands are ranked.',
           'Sensitivity counts at -8.5 and -9 are retained in annotated rankings.',
           'No spatial pose deduplication or MD is performed by this module.',
           'PDB snapshots contain coordinates only; full scored PDBQT poses remain in screen/raw.',
           'Quality warnings are preserved; absence of supplied annotations is explicitly marked unknown.',
           'Fill in the event-specific track/template if required; no track is invented.']
    if synthetic:
        lines.insert(0,'SYNTHETIC SOFTWARE TEST ONLY; these rows must be excluded from scientific candidate lists.')
    (root/'RESULTS_README.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference-id')
    parser.add_argument('--track',default='unspecified')
    args=parser.parse_args(argv)
    print(json.dumps(finalize(args.root,args.reference_id,args.track)))


if __name__=='__main__':main()
