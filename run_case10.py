"""Reproduce this exact Case10 comparison using locally supplied source masks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-dir', required=True, type=Path)
    parser.add_argument('--after-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    if not args.check_inputs and args.output is None:
        parser.error('--output is required unless --check-inputs is used')
    baseline = json.loads((REPO/'results/preparation.json').read_text(encoding='utf-8'))
    for name, record in baseline['structures'].items():
        for path, key in [(args.before_dir/(name+'.nii.gz'), 'before_source_sha256'),
                          (args.after_dir/(name+'.seg.nrrd'), 'after_source_sha256')]:
            if not path.is_file():
                parser.error('Missing source mask: '+str(path))
            if digest(path) != record[key]:
                parser.error('Source mask does not match this Case10 snapshot: '+str(path))
    manifest = json.loads((REPO/'tool_manifest.json').read_text(encoding='utf-8'))
    archive = REPO/'source_archive/VesMamba_HiPaS_AirRC_metrics.zip'
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('Metric archive checksum mismatch')
    for item in manifest['files']:
        if digest(REPO/'tool'/item['file']) != item['sha256']:
            raise ValueError('Metric source checksum mismatch: '+item['file'])
    print('All 8 source masks and the original metric package match the saved Case10 snapshot.', flush=True)
    if args.check_inputs:
        return
    packages = ['numpy','scipy','scikit-image','nibabel','surface-distance','openpyxl']
    versions = {name:importlib.metadata.version(name) for name in packages}
    work = args.output.resolve()
    if work.exists():
        parser.error('Output directory already exists; choose a new directory')
    work.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO/'tool', work/'tool', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (work/'source_archive').mkdir()
    shutil.copy2(archive, work/'source_archive'/archive.name)
    scripts = ['prepare_comparison.py','run_metric_sets.py','lineage_audit.py','overlap_audit.py','build_report.py','compute_lung_reference_metrics.py','build_gt_report.py','rerun_metrics.ps1']
    for name in scripts:
        shutil.copy2(REPO/name, work/name)
    manifest['archive'] = str(work/'source_archive'/archive.name)
    (work/'tool_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (work/'results').mkdir()
    (work/'results/environment_versions.json').write_text(json.dumps({'python':sys.version,'packages':versions},indent=2),encoding='utf-8')

    def run(script, *arguments):
        subprocess.run([sys.executable,'-B',str(work/script),*[str(arg) for arg in arguments]],cwd=work,check=True)

    with (work/'results/unit_tests.txt').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(work/'tool/pulmonary_metrics'),'-p','test*.py'],cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('prepare_comparison.py',work,args.before_dir.resolve(),args.after_dir.resolve())
    for structure in ['pulmonary_artery','pulmonary_vein','airway','lung']:
        run('run_metric_sets.py',work,structure)
    run('lineage_audit.py')
    run('overlap_audit.py')
    run('build_report.py',work)
    run('compute_lung_reference_metrics.py',work)
    run('build_gt_report.py',work)
    print('Completed: '+str(work),flush=True)


if __name__ == '__main__':
    main()
