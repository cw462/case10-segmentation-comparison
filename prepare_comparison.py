"""Prepare unchanged-grid NIfTI copies and quantify actual annotation edits."""
import csv, gc, gzip, hashlib, itertools, json, re, shutil, sys
from datetime import datetime
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

work=Path(sys.argv[1]).resolve()
before_root=Path(sys.argv[2]).resolve()
after_root=Path(sys.argv[3]).resolve()
names=['airway','lung','pulmonary_artery','pulmonary_vein']
for sub in ['inputs/before','inputs/after','changes','results']:
    (work/sub).mkdir(parents=True,exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_nrrd(p):
    fields={}
    with p.open('rb') as f:
        assert f.readline().startswith(b'NRRD')
        while True:
            line=f.readline()
            if line in (b'\n',b'\r\n'): break
            if not line: raise ValueError('Bad NRRD header')
            s=line.decode('utf-8').strip()
            if s.startswith('#'): continue
            sep=':=' if ':=' in s else ':'
            k,v=s.split(sep,1);fields[k]=v.strip()
        assert fields['type']=='unsigned char' and fields['dimension']=='3' and fields['encoding']=='gzip'
        assert fields['space']=='left-posterior-superior'
        with gzip.GzipFile(fileobj=f) as z: raw=z.read()
    size=tuple(map(int,fields['sizes'].split()))
    assert len(raw)==np.prod(size)
    data=np.frombuffer(raw,dtype=np.uint8).reshape(size[::-1]).transpose(2,1,0)
    assert np.all((data==0)|(data==1))
    directions=np.asarray([list(map(float,x.split(','))) for x in re.findall(r'\(([^)]+)\)',fields['space directions'])]).T
    origin=np.asarray(list(map(float,fields['space origin'].strip('()').split(','))))
    affine=np.eye(4);affine[:3,:3]=directions;affine[:3,3]=origin
    affine=np.diag([-1,-1,1,1])@affine
    return data,affine

def save_nii(data,affine,path):
    img=nib.Nifti1Image(data.astype(np.uint8,copy=False),affine)
    img.header.set_xyzt_units('mm')
    img.set_qform(affine,code=1);img.set_sform(affine,code=1)
    nib.save(img,path)

def cc(mask):
    if not mask.any(): return {'6':{'components':0},'26':{'components':0}}
    bounds=[]
    for axis in range(3):
        ix=np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append(slice(int(ix[0]),int(ix[-1])+1))
    cropped=mask[tuple(bounds)]
    result={}
    for conn in [6,26]:
        lab,n=ndi.label(cropped,structure=ndi.generate_binary_structure(3,1 if conn==6 else 3))
        counts=np.zeros(n+1,dtype=np.int64)
        for plane in lab:
            counts+=np.bincount(plane.ravel(),minlength=n+1)
        sizes=np.sort(counts[1:])[::-1]
        result[str(conn)]={'components':int(n),'largest_voxels':int(sizes[0]),'largest_fraction':float(sizes[0]/sizes.sum()),
                          'outside_largest_voxels':int(sizes.sum()-sizes[0]),'components_ge20':int((sizes>=20).sum()),
                          'components_ge100':int((sizes>=100).sum()),'largest10_sizes':sizes[:10].tolist()}
        del lab
    return result

report={'created':datetime.now().isoformat(),'before_source':str(before_root),'after_source':str(after_root),
        'reference_role':'supplemented_annotation_as_final_version_reference_not_independent_ground_truth',
        'independent_gt_available':False,'resampling':'none','mask_fusion':'none; overlapping channels retained separately',
        'structures':{},'overlaps':{},'complete':False}
def save():
    (work/'results/preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

rows=[]
for name in names:
    print('PREPARE',name,flush=True)
    src=before_root/(name+'.nii.gz');dst=work/'inputs/before'/src.name
    if dst.exists(): raise RuntimeError('Prepared input exists: '+str(dst))
    shutil.copy2(src,dst)
    before_img=nib.load(dst)
    before_data=np.asanyarray(before_img.dataobj)
    if not np.all((before_data==0)|(before_data==1)): raise ValueError('Before is not binary')
    if before_img.header.get_xyzt_units()[0]!='mm': raise ValueError('Unexpected NIfTI units')
    nrrd=after_root/(name+'.seg.nrrd')
    after_data,after_affine=load_nrrd(nrrd)
    if before_data.shape!=after_data.shape or not np.allclose(before_img.affine,after_affine,rtol=0,atol=1e-5):
        raise ValueError('Geometry mismatch; do not resample silently')
    out=work/'inputs/after'/(name+'.nii.gz')
    save_nii(after_data,after_affine,out)
    p=before_data.astype(bool);g=after_data.astype(bool)
    del before_data,after_data
    bp=int(np.count_nonzero(p));ag=int(np.count_nonzero(g));both=int(np.count_nonzero(p&g))
    added=ag-both;removed=bp-both
    volume=float(abs(np.linalg.det(after_affine[:3,:3]))/1000)
    print('EDITS',name,'before',bp,'after',ag,'added',added,'removed',removed,flush=True)
    item={'before_file':str(dst),'after_file':str(out),'original_before_file':str(src),'original_after_file':str(nrrd),
          'before_source_sha256':sha(src),'before_copy_sha256':sha(dst),'after_source_sha256':sha(nrrd),'after_nifti_sha256':sha(out),
          'shape_xyz':list(p.shape),'spacing_xyz_mm':list(map(float,before_img.header.get_zooms()[:3])),
          'affine_ras':after_affine.tolist(),'before_voxels':bp,'after_voxels':ag,'shared_voxels':both,
          'added_voxels':added,'removed_voxels':removed,'net_voxel_change':ag-bp,'identical_voxelwise':added==0 and removed==0,
          'before_volume_ml':bp*volume,'after_volume_ml':ag*volume,'net_volume_change_ml':(ag-bp)*volume,
          'version_agreement_Dice':2*both/(bp+ag) if bp+ag else None}
    item['before_components']=cc(p)
    item['after_components']=item['before_components'] if item['identical_voxelwise'] else cc(g)
    if added: save_nii(g&~p,after_affine,work/'changes'/(name+'_added.nii.gz'))
    if removed: save_nii(p&~g,after_affine,work/'changes'/(name+'_removed.nii.gz'))
    row={'structure':name,**{k:item[k] for k in ['before_voxels','after_voxels','added_voxels','removed_voxels','net_voxel_change','before_volume_ml','after_volume_ml','net_volume_change_ml','version_agreement_Dice','identical_voxelwise']}}
    for stage in ['before','after']:
        c=item[stage+'_components']['26']
        for k in ['components','largest_fraction','outside_largest_voxels','components_ge20','components_ge100']:
            row[stage+'_26_'+k]=c[k]
    report['structures'][name]=item;rows.append(row);save()
    del p,g,before_img;gc.collect()

for stage in ['before','after']:
    report['overlaps'][stage]={}
    for a,b in itertools.combinations(['airway','pulmonary_artery','pulmonary_vein'],2):
        x=np.asanyarray(nib.load(work/'inputs'/stage/(a+'.nii.gz')).dataobj).astype(bool)
        y=np.asanyarray(nib.load(work/'inputs'/stage/(b+'.nii.gz')).dataobj).astype(bool)
        overlap=x&y;count=int(np.count_nonzero(overlap))
        comps=cc(overlap)['26']['components'] if count else 0
        report['overlaps'][stage][a+'__'+b]={'voxels':count,'components_26':comps,'volume_ml':count*volume}
        del x,y,overlap;gc.collect()
    save()
report['complete']=True;save()
with (work/'results/structural_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PREPARATION_COMPLETE',json.dumps(rows,ensure_ascii=False),flush=True)
