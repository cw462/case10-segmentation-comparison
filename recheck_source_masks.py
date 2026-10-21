"""Recheck original source masks directly, without relying on converted copies."""
import ast
import gzip
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import nibabel as nib
import numpy as np

work=Path(__file__).resolve().parent
before=Path(sys.argv[1])
after=Path(sys.argv[2])
source=ast.parse((work/'prepare_comparison.py').read_text(encoding='utf-8'))
function=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='load_nrrd')
namespace={'gzip':gzip,'np':np,'re':re}
exec(compile(ast.Module(body=[function],type_ignores=[]),'prepare_comparison.py:load_nrrd','exec'),namespace)
load_nrrd=namespace['load_nrrd']
prior=json.loads((work/'results/preparation.json').read_text(encoding='utf-8'))
rows=[]
for name in ['airway','lung','pulmonary_artery','pulmonary_vein']:
    old=before/(name+'.nii.gz');new=after/(name+'.seg.nrrd')
    img=nib.load(old);p=np.asanyarray(img.dataobj)
    g,aff=load_nrrd(new)
    shape_match=p.shape==g.shape
    affine_match=bool(np.allclose(img.affine,aff,atol=1e-5,rtol=0))
    assert shape_match and affine_match
    assert np.isin(p,[0,1]).all() and np.isin(g,[0,1]).all()
    differing=int(np.count_nonzero(p!=g))
    added=int(np.count_nonzero((p==0)&(g==1)))
    removed=int(np.count_nonzero((p==1)&(g==0)))
    hashes={}
    for label,path,key in [('before',old,'before_source_sha256'),('after',new,'after_source_sha256')]:
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==prior['structures'][name][key]
        hashes[label]=digest
    rows.append({'structure':name,'before_filename':old.name,'after_filename':new.name,
                 'same_shape':shape_match,'same_physical_grid':affine_match,
                 'foreground_voxels_before':int(np.count_nonzero(p)),
                 'foreground_voxels_after':int(np.count_nonzero(g)),
                 'identical_voxelwise':differing==0,'different_voxels':differing,
                 'added_voxels':added,'removed_voxels':removed,
                 'source_hashes_match_previous_run':True,'sha256':hashes})
    del img,p,g
report={'created':datetime.now().isoformat(timespec='seconds'),'scope':'Original NIfTI masks vs original NRRD masks; not directory or bytewise file equality.',
        'structures':rows,'all_original_source_hashes_match_previous_run':True}
(work/'results/source_masks_recheck.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
