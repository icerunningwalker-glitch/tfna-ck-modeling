"""Create eight equal-sized overlapping spatial tiles for one PDBQT receptor.

Reuses the original 2 x 2 x 2 geometry. Tiles are not biological domains.
This script does not repair structures or prepare receptor chemistry.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import shutil
from pathlib import Path
from run_screen import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_csv(path, rows):
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    payload=buffer.getvalue().encode('utf-8-sig')
    if path.exists() and path.read_bytes()!=payload:
        raise ValueError('Frozen boxes changed; choose a new output directory')
    if not path.exists():path.write_bytes(payload)


def build_boxes(source, output_dir, margin=8.0, sequence_config=None):
    source=Path(source).resolve();output_dir=Path(output_dir).resolve()
    if not source.is_file():raise ValueError('Receptor file does not exist')
    if not math.isfinite(margin) or margin<0:raise ValueError('margin must be finite and nonnegative')
    atoms=[];residue_maps={}
    for line in source.read_text(encoding='utf-8-sig').splitlines():
        if not line.startswith(('ATOM  ','HETATM')):continue
        if line[77:].strip() not in ('H','HD','HS'):
            xyz=tuple(float(line[a:b]) for a,b in ((30,38),(38,46),(46,54)))
            if not all(math.isfinite(value) for value in xyz):raise ValueError('Nonfinite receptor coordinates')
            atoms.append(xyz)
        residue_maps.setdefault(line[21],{})[int(line[22:26])]=line[17:20].strip()
    if not atoms:raise ValueError('Receptor has no readable heavy atoms')
    if sequence_config:
        config=json.loads(Path(sequence_config).read_text(encoding='utf-8-sig'))
        dna={base+end:base[-1] for base in ('DA','DC','DG','DT') for end in ('','3','5')}
        actual={chain:''.join(dna.get(residue,'?') for _,residue in sorted(residues.items()))
                for chain,residues in residue_maps.items()}
        expected={chain:config['strands'][strand] for chain,strand in zip('ABCD',('1c','2c','3c','4c'))}
        if actual!=expected:raise ValueError('Receptor sequences differ from supplied configuration')
    lo=[min(xyz[k] for xyz in atoms) for k in range(3)]
    hi=[max(xyz[k] for xyz in atoms) for k in range(3)]
    centers=[[a+(b-a)/4,a+3*(b-a)/4] for a,b in zip(lo,hi)]
    sizes=[math.ceil(((b-a)/2+2*margin+.002)*1000)/1000 for a,b in zip(lo,hi)]
    output_dir.mkdir(parents=True,exist_ok=True)
    target=output_dir/'receptor.pdbqt'
    if target.exists() and sha(target)!=sha(source):raise ValueError('Existing receptor differs; refusing replacement')
    if not target.exists():shutil.copyfile(source,target)
    boxes=[]
    for indices in itertools.product(range(2),repeat=3):
        box=dict(candidate_id='WT',site_id='T'+''.join(map(str,indices)),receptor=target.name)
        for k,axis in enumerate('xyz'):box['center_'+axis]=round(centers[k][indices[k]],4)
        for k,axis in enumerate('xyz'):box['size_'+axis]=sizes[k]
        boxes.append(box)
    def covered(xyz):
        return any(all(abs(xyz[k]-box['center_'+axis])<=box['size_'+axis]/2
                       for k,axis in enumerate('xyz')) for box in boxes)
    if not all(covered(xyz) for xyz in atoms):raise ValueError('Uncovered receptor heavy atoms')
    freeze_csv(output_dir/'site_jobs.csv',boxes)
    report=dict(status='PASS',receptor_file=target.name,receptor_sha256=sha(target),
                heavy_atoms=len(atoms),covered_heavy_atoms=len(atoms),bbox_min=lo,bbox_max=hi,
                margin_A=margin,box_count=8,sequence_checked=bool(sequence_config),
                note='Overlapping spatial tiles are not independent binding sites or physical domains.')
    atomic_json(output_dir/'box_coverage_audit.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receptor',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--margin',type=float,default=8.0)
    parser.add_argument('--sequence-config',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(build_boxes(args.receptor,args.output_dir,args.margin,args.sequence_config)))


if __name__=='__main__':main()
