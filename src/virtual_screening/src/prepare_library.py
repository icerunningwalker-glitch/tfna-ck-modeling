"""Auditable 2D SDF -> 3D MOL2 -> ADT PDBQT for the supplied catalog."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def parse_sdf(path):
    rows = []
    for block in path.read_text(encoding='utf-8-sig').split('$$$$'):
        if not block.strip():
            continue
        block = block.strip('\r\n')+'\n'
        lines = block.splitlines()
        props = {m[1]:m[2].strip() for m in re.finditer(r'^>\s*<([^>]+)>[^\n]*\n(.*?)(?=\n\s*\n|\Z)', block, re.M|re.S)}
        ident = props.get('ID','')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',ident):
            raise ValueError('Invalid catalog ID: '+ident)
        na, nb = int(lines[3][:3]), int(lines[3][3:6])
        atoms = lines[4:4+na]
        adjacency = [set() for _ in atoms]
        for line in lines[4+na:4+na+nb]:
            a,b = int(line[:3])-1,int(line[3:6])-1
            adjacency[a].add(b); adjacency[b].add(a)
        seen, components = set(), []
        for a in range(na):
            if a in seen: continue
            stack, comp = [a], set()
            while stack:
                b = stack.pop()
                if b in comp: continue
                comp.add(b); stack.extend(adjacency[b]-comp)
            seen.update(comp)
            components.append({'atoms':len(comp), 'heavy':sum(atoms[i][31:34].strip()!='H' for i in comp),
                               'elements':sorted(atoms[i][31:34].strip() for i in comp)})
        # Fragment handling requires explicit review; disconnected records need not be salts.
        largest = max(components, key=lambda x:x['heavy'])
        rows.append(dict(index=len(rows)+1,identifier=ident,name=lines[0],smiles=props.get('SMILES',''),
                         cas=props.get('CAS',''),formula=props.get('Formula',''),
                         molecular_weight=props.get('MolWt',''),heavy_atoms=largest['heavy'],
                         components=components,source_max_abs_z=max(abs(float(a[20:30])) for a in atoms),
                         block=block))
    if len({r['identifier'] for r in rows})!=len(rows): raise ValueError('Duplicate catalog IDs')
    return rows


def run(cmd,cwd=None,timeout=240):
    env = os.environ.copy()
    result = subprocess.run(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,errors='replace',timeout=timeout)
    if result.returncode: raise RuntimeError(f'exit={result.returncode}: {result.stderr[-3000:]} {result.stdout[-1000:]}')
    return result.stdout, result.stderr


def audit_pdbqt(path, expected, allow_planar=False):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if 'ROOT' not in lines: raise ValueError('No ligand ROOT')
    coords,types,charges=[],[],[]
    for line in lines:
        if not line.startswith(('ATOM  ','HETATM')): continue
        typ=line[77:].strip()
        xyz=[float(line[30:38]),float(line[38:46]),float(line[46:54])]
        if not all(math.isfinite(x) for x in xyz): raise ValueError('Nonfinite coordinates')
        charge=float(line[70:76])
        if not math.isfinite(charge):raise ValueError('Nonfinite PDBQT charge')
        charges.append(charge)
        if typ not in ('H','HD','HS'):
            coords.append(xyz); types.append(typ)
    if len(coords)!=expected: raise ValueError(f'Heavy atoms {len(coords)} != source {expected}')
    spans=[max(x[k] for x in coords)-min(x[k] for x in coords) for k in range(3)]
    if not allow_planar and min(spans)<0.1: raise ValueError(f'Degenerate/2D coordinates: {spans}')
    allowed={'C','A','N','NA','NS','OA','OS','S','SA','P','F','Cl','Br','I'}
    if set(types)-allowed: raise ValueError(f'Unreviewed atom types {set(types)-allowed}')
    return dict(heavy_atoms=len(coords),spans_A=spans,charge_sum=round(sum(charges),4),
                atom_types=sorted(set(types)),pdbqt_sha256=sha(path))


def rdkit_3d(source, output, seed):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdCIPLabeler
    mol=Chem.SDMolSupplier(str(source),removeHs=False)[0]
    if mol is None:raise ValueError('RDKit could not parse source mol block')
    mol=max(Chem.GetMolFrags(mol,asMols=True),key=lambda m:m.GetNumHeavyAtoms())
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    source_smiles=Chem.MolToSmiles(Chem.RemoveHs(mol),isomericSmiles=True)
    mol=Chem.AddHs(mol)
    # The modern CIP implementation is stable across implicit/explicit H;
    # the legacy labeler can switch R/S names without any stereo change.
    rdCIPLabeler.AssignCIPLabels(mol)
    defined={a.GetIdx():a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')}
    defined_double={b.GetIdx():str(b.GetStereo()) for b in mol.GetBonds()
                    if b.GetStereo() not in (Chem.BondStereo.STEREONONE,Chem.BondStereo.STEREOANY)}
    params=AllChem.ETKDGv3();params.randomSeed=seed;params.enforceChirality=True
    params.numThreads=1;params.maxIterations=1000;params.clearConfs=True
    if hasattr(params,'timeout'):params.timeout=30
    cids=[cid for cid in AllChem.EmbedMultipleConfs(mol,numConfs=4,params=params) if cid>=0]
    if not cids:
        params.useRandomCoords=True
        cids=[cid for cid in AllChem.EmbedMultipleConfs(mol,numConfs=4,params=params) if cid>=0]
    if not cids:raise ValueError('ETKDG embedding failed')
    # A partially failed embedding can leave gaps in conformer IDs. Normalize
    # bookkeeping before legacy bulk minimizers; coordinates are untouched.
    for index,conformer in enumerate(mol.GetConformers()):conformer.SetId(index)
    cids=list(range(mol.GetNumConformers()))
    if not cids:raise ValueError('ETKDG returned no usable coordinate conformer')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        forcefield='MMFF94s'
        energies=AllChem.MMFFOptimizeMoleculeConfs(mol,numThreads=1,maxIters=1000,mmffVariant='MMFF94s')
    else:
        forcefield='UFF';energies=AllChem.UFFOptimizeMoleculeConfs(mol,numThreads=1,maxIters=1000)
    scored=[(converged!=0,energy,converged,cid) for cid,(converged,energy) in zip(cids,energies)
            if converged>=0 and math.isfinite(energy)]
    if not scored:raise ValueError('No finite conformer energy')
    def valid_conformers(scored_pool):
        accepted=[]
        for candidate in sorted(scored_pool):
            _,energy,converged,cid=candidate
            check=Chem.Mol(mol)
            Chem.AssignStereochemistryFrom3D(check,confId=cid,replaceExistingTags=True)
            Chem.AssignStereochemistry(check,cleanIt=True,force=True)
            rdCIPLabeler.AssignCIPLabels(check)
            centers_ok=all(check.GetAtomWithIdx(index).HasProp('_CIPCode') and
                           check.GetAtomWithIdx(index).GetProp('_CIPCode')==cip for index,cip in defined.items())
            bonds_ok=all(str(check.GetBondWithIdx(index).GetStereo())==stereo for index,stereo in defined_double.items())
            if centers_ok and bonds_ok:accepted.append(candidate)
        return accepted
    valid=valid_conformers(scored)
    attempted=len(cids);seeds=[seed]
    if not valid:
        params.randomSeed=seed+1;params.useRandomCoords=True
        cids=[cid for cid in AllChem.EmbedMultipleConfs(mol,numConfs=8,params=params) if cid>=0]
        for index,conformer in enumerate(mol.GetConformers()):conformer.SetId(index)
        cids=list(range(mol.GetNumConformers()))
        attempted+=len(cids);seeds.append(seed+1)
        if not cids:raise ValueError('ETKDG retry could not embed a source-stereochemistry-preserving conformer')
        if forcefield=='MMFF94s':
            energies=AllChem.MMFFOptimizeMoleculeConfs(mol,numThreads=1,maxIters=1000,mmffVariant='MMFF94s')
        else:
            energies=AllChem.UFFOptimizeMoleculeConfs(mol,numThreads=1,maxIters=1000)
        scored=[(flag!=0,energy,flag,cid) for cid,(flag,energy) in zip(cids,energies) if flag>=0 and math.isfinite(energy)]
        valid=valid_conformers(scored)
    if not valid:raise ValueError('No minimized conformer preserves all source-defined stereochemistry')
    _,energy,converged,cid=min(valid)
    writer=Chem.SDWriter(str(output));writer.write(mol,confId=cid);writer.close()
    return dict(rdkit_version=rdBase.rdkitVersion,embedding='ETKDGv3',random_seed=seed,
                sampled_conformers=attempted,embedding_seeds=seeds,stereochemically_valid_conformers=len(valid),forcefield=forcefield,energy_kcal_mol=energy,
                minimization_converged=converged==0,source_rdkit_smiles=source_smiles,
                specified_stereocenters=len(defined),cip_algorithm='rdCIPLabeler',defined_stereochemistry_preserved=True,
                formal_charge=Chem.GetFormalCharge(mol))


def prepare(row,args):
    folder=args.output_dir/('molecules_rdkit' if args.rdkit_tools else 'molecules')/row['identifier']
    folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/'preparation.json'
    pdbqt=folder/'ligand.pdbqt'
    if status_path.exists():
        old=json.loads(status_path.read_text(encoding='utf-8'))
        cached_pdbqt=Path(old.get('pdbqt',str(pdbqt)))
        if old.get('status')=='prepared' and cached_pdbqt.exists() and sha(cached_pdbqt)==old['audit']['pdbqt_sha256']:
            if args.rdkit_tools:
                from chemistry_geometry_checks import audit_conversion_geometry
                expected_source=(row['block']+'$$$$\n').replace('\r\n','\n')
                actual_source=(folder/'source.sdf').read_text(encoding='utf-8-sig').replace('\r\n','\n')
                if actual_source!=expected_source:
                    raise ValueError('Cached source SDF differs from the current catalog record')
                old['audit'].update(audit_pdbqt(cached_pdbqt,row['heavy_atoms'],allow_planar=True))
                geometry=audit_conversion_geometry(folder/'ligand_3d.sdf',folder/'ligand.mol2',cached_pdbqt)
                if abs(old['audit']['charge_sum']-geometry['source_formal_charge'])>0.15:
                    raise ValueError('Cached PDBQT charge sum differs from source formal charge')
                old['audit']['conversion_geometry']=geometry
                old['audit']['conversion_audit_version']=2
                save(status_path,old)
            return old
    start=time.monotonic()
    status={k:v for k,v in row.items() if k!='block'}
    status.update(pdbqt=str(pdbqt),status='failed',detail='')
    try:
        source=folder/'source.sdf'; source.write_text(row['block']+'$$$$\n',encoding='utf-8')
        mol2=folder/'ligand.mol2'
        conformer_audit={}
        if args.rdkit_tools:
            three_d=folder/'ligand_3d.sdf'
            try:
                conformer_audit=rdkit_3d(source,three_d,20260930)
            except ValueError as error:
                conformer_errors=('ETKDG embedding failed',
                                  'ETKDG returned no usable coordinate conformer',
                                  'ETKDG retry could not embed a source-stereochemistry-preserving conformer',
                                  'No minimized conformer preserves all source-defined stereochemistry')
                if str(error) not in conformer_errors:
                    raise
                from rigid_conformer_fallback import build_rigid_conformer
                conformer_audit=build_rigid_conformer(source,three_d,seed=20260930)
                conformer_audit['primary_embedding_failure']=str(error)
            out,err=run([str(args.obabel),'-isdf',str(three_d),'-omol2','-O',str(mol2)],timeout=args.timeout)
        else:
            out,err=run([str(args.obabel),'-isdf',str(source),'-omol2','-r','--gen3d','-O',str(mol2)],timeout=args.timeout)
        (folder/'gen3d.log').write_text(out+err,encoding='utf-8')
        out,err=run([str(args.adt_python),str(args.prepare_ligand),'-l',mol2.name,'-o',str(pdbqt)],cwd=str(folder),timeout=90)
        (folder/'pdbqt.log').write_text(out+err,encoding='utf-8')
        audit=audit_pdbqt(pdbqt,row['heavy_atoms'],allow_planar=bool(args.rdkit_tools))
        audit.update(conformer_audit)
        if args.rdkit_tools:
            from chemistry_geometry_checks import audit_conversion_geometry
            geometry=audit_conversion_geometry(three_d,mol2,pdbqt)
            audit['charge_method']='ADT Gasteiger'
            if abs(audit['charge_sum']-geometry['source_formal_charge'])>0.15:
                # Legacy ADT mischarges some neutral acids by one charge unit.
                # Keep that failed file; export a separate ligand preserving the
                # source-aware OpenBabel Gasteiger charges already in MOL2.
                corrected=folder/'ligand_charge_preserved.pdbqt'
                out,err=run([str(args.adt_python),str(args.prepare_ligand),'-l',mol2.name,'-o',str(corrected),'-C'],cwd=str(folder),timeout=90)
                (folder/'pdbqt_preserved_charge.log').write_text(out+err,encoding='utf-8')
                corrected_audit=audit_pdbqt(corrected,row['heavy_atoms'],allow_planar=True)
                if abs(corrected_audit['charge_sum']-geometry['source_formal_charge'])>0.15:
                    raise ValueError('Both ADT/default and preserved MOL2 charge sums differ from source formal charge')
                geometry=audit_conversion_geometry(three_d,mol2,corrected)
                audit.update(corrected_audit)
                audit['charge_method']='Source-aware OpenBabel Gasteiger retained with ADT -C'
                audit['failed_default_charge_file']=str(pdbqt)
                status['pdbqt']=str(corrected)
            audit['conversion_geometry']=geometry
            audit['conversion_audit_version']=2
        source_can=run([str(args.obabel),'-isdf',str(source),'-r','-ocan'])[0].splitlines()[0].split()[0]
        try:
            mol2_can=run([str(args.obabel),'-imol2',str(mol2),'-ocan'])[0].splitlines()[0].split()[0]
        except RuntimeError as error:
            if not args.rdkit_tools or 'exit=3221225477' not in str(error):
                raise
            # OpenBabel 2.3.2 crashes perceiving one already exported MOL2.
            # Stereo and every atom/bond/coordinate were independently checked
            # above; this optional canonical-SMILES export is only metadata.
            mol2_can=None
            audit['canonical_smiles_export_error']=str(error)
            audit['canonical_smiles_fallback']='Independent RDKit source stereo and full MOL2/PDBQT graph-coordinate audit'
        audit.update(source_canonical_smiles=source_can,generated_canonical_smiles=mol2_can)
        if source_can!=mol2_can:
            if not args.rdkit_tools:
                raise ValueError('3D connectivity/stereochemistry mismatch: '+source_can+' vs '+mol2_can)
            audit['canonical_stereo_note']='OpenBabel MOL2 perception differs in charge/stereo/implicit H; actual full atom geometry and bond graph preserved, and PDBQT net charge checked against source.'
        status.update(status='prepared',audit=audit)
    except Exception as error:
        status['detail']=str(error)
    status['elapsed_seconds']=round(time.monotonic()-start,3)
    save(status_path,status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--obabel',type=Path,help='OpenBabel executable supplied by the recipient')
    p.add_argument('--adt-python',type=Path,help='AutoDockTools Python interpreter')
    p.add_argument('--prepare-ligand',type=Path,help='AutoDockTools prepare_ligand4.py')
    p.add_argument('--workers',type=int,default=4); p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--ids',default=''); p.add_argument('--audit-only',action='store_true')
    p.add_argument('--rdkit-tools',type=Path)
    p.add_argument('--use-rdkit',action='store_true',help='Use RDKit in the active Python environment')
    p.add_argument('--largest-fragment',action='store_true',help='Approve largest-heavy-fragment selection for reviewed disconnected records')
    args=p.parse_args()
    args.input=args.input.resolve();args.output_dir=args.output_dir.resolve()
    if not args.audit_only:
        for key in ('obabel','adt_python','prepare_ligand'):
            value=getattr(args,key)
            if value is None or not value.is_file():
                p.error('An existing --'+key.replace('_','-')+' file is required for preparation')
            setattr(args,key,value.resolve())
    if args.use_rdkit and args.rdkit_tools:p.error('Choose --use-rdkit or --rdkit-tools, not both')
    if args.use_rdkit:
        import rdkit
        args.rdkit_tools=Path(rdkit.__file__).resolve().parent.parent
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.rdkit_tools:
        sys.path.insert(0,str(args.rdkit_tools))
        import rdkit  # This option requires an already installed, user-authorized isolated dependency.
    copied=args.output_dir/args.input.name
    if copied.exists() and sha(copied)!=sha(args.input): raise ValueError('Source copy differs; refusing replacement')
    if not copied.exists(): shutil.copy2(args.input,copied)
    rows=parse_sdf(copied)
    if not args.audit_only and not args.largest_fragment and any(len(row['components'])>1 for row in rows):
        p.error('Review disconnected SDF records first; --largest-fragment approves fragment removal')
    duplicates={}
    for r in rows: duplicates.setdefault(r['smiles'],[]).append(r['identifier'])
    audit=dict(source=str(args.input),copy=str(copied),sha256=sha(copied),records=len(rows),
               unique_ids=len({r['identifier'] for r in rows}),all_2D=all(r['source_max_abs_z']==0 for r in rows),
               salts=[dict(identifier=r['identifier'],name=r['name'],components=r['components']) for r in rows if len(r['components'])>1],
               exact_smiles_duplicates=[x for x in duplicates.values() if len(x)>1],
               ck=[dict(identifier=r['identifier'],name=r['name'],cas=r['cas']) for r in rows if r['cas']=='39262-14-1'],
               structure_authority='Supplied SDF mol blocks; vendor SMILES retained as metadata.',
               preparation=('RDKit ETKDGv3 + MMFF94s/UFF from SDF' if args.rdkit_tools else 'OpenBabel3D from SDF')+'; largest fragment for counterion salts, ADT Gasteiger charges; no pH enumeration.')
    save(args.output_dir/'library_audit.json',audit)
    fields=['index','identifier','name','cas','formula','molecular_weight','smiles','heavy_atoms','source_max_abs_z']
    with (args.output_dir/'library_catalog.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps(audit),flush=True)
    if args.audit_only:return
    selected=set(args.ids.split(',')) if args.ids else {r['identifier'] for r in rows}
    if selected-{r['identifier'] for r in rows}:raise ValueError('Unknown requested ID')
    results=[];start=time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare,row,args) for row in rows if row['identifier'] in selected]
        for fut in cf.as_completed(futures):
            result=fut.result();results.append(result)
            success=sum(r['status']=='prepared' for r in results)
            save(args.output_dir/'preparation_progress.json',dict(stage='ligand_preparation',completed=len(results),total=len(futures),success=success,failed=len(results)-success,elapsed_seconds=round(time.monotonic()-start,1)))
            print(f"{len(results)}/{len(futures)} {result['identifier']} {result['status']} {result['detail'][:240]}",flush=True)
    manifest=[]
    for row in rows:
        path=args.output_dir/('molecules_rdkit' if args.rdkit_tools else 'molecules')/row['identifier']/'preparation.json'
        result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else row|dict(status='not_prepared',pdbqt='',detail='')
        fields=['index','identifier','name','cas','formula','molecular_weight','smiles','pdbqt','status','detail']
        manifest.append({k:result.get(k,'') for k in fields})
    with (args.output_dir/'ligand_preparation_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest)
    if any(r['status']=='failed' for r in results):raise SystemExit(2)


if __name__=='__main__':main()
