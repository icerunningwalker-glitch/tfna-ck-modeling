"""Small synthetic workflow example; mock scores are NOT scientific results.

Default mode uses no external software. --vina uses real Vina on a toy receptor
only to test installation and file handling, never to infer ligand binding.
"""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch
import sys
sys.dont_write_bytecode = True
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT / 'src'))
import screen
import run_screen


def atom(serial, kind, xyz, residue='LIG'):
    x,y,z=xyz;name=kind[0]+str(serial)
    return f'ATOM  {serial:5d} {name:<4} {residue:>3} A   1    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}    {0.0:6.3f} {kind}'


def fixtures(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    receptor=folder/'toy_receptor.pdbqt'
    receptor.write_text('\n'.join(atom(i,'C',xyz,'DUM') for i,xyz in enumerate(
        ((-4,-4,-4),(4,-4,-4),(-4,4,-4),(4,4,4)),1))+'\n',encoding='ascii')
    rows=[]
    for ident,kind,smiles in (('SYN001','OA','CO'),('SYN002','NA','CN')):
        ligand=folder/(ident+'.pdbqt')
        ligand.write_text('ROOT\n'+atom(1,'C',(0,0,0))+'\n'+atom(2,kind,(1.43,0,0))+
                          '\nENDROOT\nTORSDOF 0\n',encoding='ascii')
        rows.append(dict(identifier=ident,name='SYNTHETIC fixture '+ident,pdbqt=ligand.name,
                         status='prepared',cas='',formula='',molecular_weight='',smiles=smiles,detail='Toy pipeline fixture; no biological inference'))
    manifest=folder/'ligands.csv'
    with manifest.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return manifest,receptor


def mock_vina(command,**kwargs):
    """Deterministic artificial scores, explicitly unrelated to affinity."""
    pose=Path(command[command.index('--out')+1]);log=Path(command[command.index('--log')+1])
    ligand=Path(command[command.index('--ligand')+1])
    score=-8.6 if ligand.stem=='SYN001' else -7.2
    content=ligand.read_text(encoding='utf-8')
    pose.write_text(f'MODEL 1\nREMARK VINA RESULT: {score:.3f} 0.0 0.0\n'+content+'ENDMDL\n',encoding='utf-8')
    log.write_text(f'mode | affinity | dist\n    1    {score:.1f}    0.0    0.0\n',encoding='utf-8')
    return argparse.Namespace(returncode=0,stdout='SYNTHETIC mock Vina output\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT / 'results' / 'virtual_screening' / 'synthetic_demo')
    parser.add_argument('--vina',type=Path,help='Optional real executable for toy installation smoke test')
    args=parser.parse_args(argv);root=args.output_dir.resolve()
    # Avoid changing an existing run, protocol or fixtures.
    if root.exists():parser.error('Demo output already exists; choose a new directory')
    manifest,receptor=fixtures(root/'inputs')
    # Persist this marker before the first job, so export-only paths also know
    # that every candidate is synthetic, even if the caller changes --track.
    (root/'run').mkdir(parents=True,exist_ok=True)
    (root/'run'/'SYNTHETIC_SOFTWARE_TEST.json').write_text(json.dumps(dict(
        evidence_type='synthetic_software_test',scientific_candidate=False)),encoding='utf-8')
    vina=args.vina
    if vina is None:
        vina=root/'mock_vina.txt';vina.write_text('Synthetic test sentinel, not executable\n',encoding='ascii')
    command=['--manifest',str(manifest),'--receptor',str(receptor),'--vina',str(vina),
             '--output-dir',str(root/'run'),'--workers','2','--cpu-per-job','1',
             '--reference-id','SYN001','--track','synthetic-demo','--execute']
    if args.vina is None:
        with patch.object(run_screen.subprocess,'run',side_effect=mock_vina):code=screen.main(command)
    else:
        command+=['--exhaustiveness','1']
        code=screen.main(command)
    (root/'SYNTHETIC_EXAMPLE.txt').write_text(
        'SYNTHETIC SOFTWARE TEST ONLY. Toy receptor, approximate fixture geometry.\n'
        'Default scores are invented deterministic mock values, not biological predictions.\n'
        'Real Vina mode is an installation smoke test, not receptor validation.\n',encoding='utf-8')
    print(json.dumps(dict(example='synthetic-software-test',real_vina=bool(args.vina),exit_code=code)))
    return code


if __name__=='__main__':raise SystemExit(main())
