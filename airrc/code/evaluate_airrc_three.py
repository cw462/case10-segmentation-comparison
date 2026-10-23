from pathlib import Path
import sys, json, gzip, shutil, hashlib, csv, gc, re, time
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
BASE=Path('data/airrc')
OUT=BASE/'分割效果评估_20261022'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vesmamba_metrics import evaluate_binary
import vesmamba_metrics as vm
original_validate_binary=vm.validate_binary
def validate_binary_low_memory(mask,name):
    if isinstance(mask,np.ndarray) and mask.dtype==np.bool_ and mask.ndim==3 and mask.size:
        return mask
    return original_validate_binary(mask,name)
vm.validate_binary=validate_binary_low_memory
from surface_distance import metrics as sd, lookup_tables
from scipy.ndimage import correlate
from scipy.spatial import cKDTree
def sparse_surfaces(gt,pred,spacing):
    table=lookup_tables.create_table_neighbour_code_to_surface_area(spacing)
    points=[];areas=[]
    lo,hi=sd._compute_bounding_box(gt|pred)
    for mask in (gt,pred):
        crop=sd._crop_to_bounding_box(mask,lo,hi)
        codes=correlate(crop.astype(np.uint8),lookup_tables.ENCODE_NEIGHBOURHOOD_3D_KERNEL,mode='constant',cval=0)
        border=(codes!=0)&(codes!=255)
        points.append(np.argwhere(border)*np.asarray(spacing))
        areas.append(table[codes[border]])
        del crop,codes,border
    dg=cKDTree(points[1]).query(points[0],workers=1)[0]
    dp=cKDTree(points[0]).query(points[1],workers=1)[0]
    dg,ag=sd._sort_distances_surfels(dg,areas[0]);dp,ap=sd._sort_distances_surfels(dp,areas[1])
    return dict(distances_gt_to_pred=dg,distances_pred_to_gt=dp,surfel_areas_gt=ag,surfel_areas_pred=ap)
rng=np.random.default_rng(42)
for spacing in [(1,1,1),(.7,.8,2.5)]:
    a=rng.random((12,13,14))>.8;b=rng.random((12,13,14))>.85
    old=sd.compute_surface_distances(a,b,spacing);new=sparse_surfaces(a,b,spacing)
    for k in ['distances_gt_to_pred','distances_pred_to_gt']:assert np.allclose(old[k],new[k],atol=1e-12)
    assert np.isclose(sd.compute_robust_hausdorff(old,95),sd.compute_robust_hausdorff(new,95),atol=1e-12)
    for tol in [0.,.5,1.,2.]:assert np.isclose(sd.compute_surface_dice_at_tolerance(old,tol),sd.compute_surface_dice_at_tolerance(new,tol),atol=1e-12)
sd.compute_surface_distances=sparse_surfaces
print('Sparse surface equivalence checks passed',flush=True)
NAMES={1:'气道腔',2:'气道壁',3:'肺动脉',4:'肺静脉'}
ENAMES={1:'Airway lumen',2:'Airway wall',3:'Pulmonary arteries',4:'Pulmonary veins'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def header(p):
    h={}
    with p.open('rb') as f:
        while True:
            l=f.readline()
            if l in (b'\n',b'\r\n',b''):break
            s=l.decode().strip()
            if ':' in s and not s.startswith('#'):
                k,v=s.split(':',1);h[k]=v.lstrip('= ').strip()
        offset=f.tell()
    return h,offset
rows=[]; audits=[]
for case in ['case01','case02','case03']:
    print(case,'loading',flush=True)
    pred=BASE/'原始CT与totalsegmental'/case/'segmentation.seg.nrrd'
    gtpath=BASE/'原始CT与标准mask'/case/'分割mask'/f'{case}.nii.gz'
    ctpath=next((BASE/'原始CT与totalsegmental'/case).glob('*.nii*'))
    h,offset=header(pred);sizes=tuple(map(int,h['sizes'].split()))
    assert h['space']=='left-posterior-superior' and h['encoding']=='gzip' and sizes[0]==5
    dirs=re.findall(r'\(([^)]+)\)',h['space directions'])
    aff=np.eye(4);aff[:3,:3]=np.array([[float(v) for v in d.split(',')] for d in dirs]).T
    aff[:3,3]=[float(v) for v in h['space origin'].strip('()').split(',')]
    aff=np.diag([-1,-1,1,1])@aff
    ct=nib.load(ctpath)
    assert ct.shape==sizes[1:] and np.allclose(ct.affine,aff,atol=1e-4)
    raw=OUT/f'working_{case}.raw'; gtnii=OUT/f'working_{case}.nii'
    with pred.open('rb') as f,raw.open('wb') as o:
        f.seek(offset)
        with gzip.GzipFile(fileobj=f) as z:shutil.copyfileobj(z,o,1024*1024)
    with gzip.open(gtpath,'rb') as f,gtnii.open('wb') as o:shutil.copyfileobj(f,o,1024*1024)
    data=np.memmap(raw,mode='r',dtype=np.uint8,shape=sizes,order='F')
    gi=nib.load(gtnii);g=np.asanyarray(gi.dataobj)
    assert set(np.unique(g))=={0,1,2,3,4}
    assert np.allclose(nib.affines.voxel_sizes(gi.affine),[1,1,1])
    transform=np.linalg.inv(aff)@gi.affine
    audit={'case':case,'prediction_sha256':sha(pred),'gt_sha256':sha(gtpath),'ct_sha256':sha(ctpath),'gt_shape':list(gi.shape),'prediction_shape':list(sizes),'gt_affine':gi.affine.tolist(),'prediction_affine':aff.tolist(),'output_to_input_transform':transform.tolist(),'resampling':'prediction to unchanged GT, nearest-neighbor, constant zero outside prediction grid','prediction_segments':{k:v for k,v in h.items() if k.startswith('Segment')}}
    audits.append(audit);dump(OUT/'input_audit.json',audits)
    for label,name in NAMES.items():
        saved=OUT/f'{case}_label{label}.json'
        if saved.exists():rows.append(json.loads(saved.read_text(encoding='utf-8')));continue
        assert h[f'Segment{label}_Name']==ENAMES[label]
        layer=int(h[f'Segment{label}_Layer']);val=int(h[f'Segment{label}_LabelValue'])
        assert val==1
        aligned=affine_transform(data[layer],transform[:3,:3],offset=transform[:3,3],output_shape=gi.shape,output=np.uint8,order=0,mode='constant',cval=0,prefilter=False)
        p=aligned==val;del aligned
        ref=g==label
        npred=int(p.sum());ngt=int(ref.sum());tp=int(np.count_nonzero(p & ref))
        print(case,name,'scoring',npred,ngt,flush=True)
        score=evaluate_binary(p,ref,(1.,1.,1.),1.)
        row={'case':case,'structure':name,'label':label,**score,'Precision':tp/npred if npred else None,'Recall':tp/ngt if ngt else None,'IoU':tp/(npred+ngt-tp),'TP':tp,'FP':npred-tp,'FN':ngt-tp,'pred_volume_ml':npred/1000,'gt_volume_ml':ngt/1000}
        assert abs(row['Dice']-2*tp/(npred+ngt))<1e-12
        rows.append(row);dump(saved,row)
        print(case,name,json.dumps(score),flush=True)
        del p,ref;gc.collect()
    del data,g,gi;gc.collect()
    raw.unlink();gtnii.unlink()
dump(OUT/'metrics_per_case.json',rows)
with (OUT/'metrics_per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=[k for k in rows[0] if k!='status'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('COMPLETE',str(OUT),flush=True)
