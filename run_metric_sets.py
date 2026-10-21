"""Apply the supplied, unmodified metric package to separate binary channels.

The supplemented version is a comparison reference, NOT independent GT.
No after-versus-itself performance gain is reported.
"""
import gc, importlib.metadata, json, sys, time
from pathlib import Path

work=Path(sys.argv[1]).resolve()
structure=sys.argv[2]
sys.path.insert(0,str(work/'tool/pulmonary_metrics'))
import numpy as np
from io_utils import load_volume, check_grids
from vesmamba_metrics import evaluate_vesmamba
from hipas_metrics import evaluate_hipas
from airrc_metrics import evaluate_airrc

start=time.time()
loaded={stage:load_volume(work/'inputs'/stage/(structure+'.nii.gz')) for stage in ['before','after']}
spacing=check_grids(loaded)
p,g=loaded['before']['data'],loaded['after']['data']
report={'structure':structure,'reference_role':'after_supplemented_final_annotation_for_version_agreement_only',
        'independent_gt_available':False,'interpretation':'Before is compared against the supplemented version. Scores describe agreement between versions; no independent accuracy improvement is estimable.',
        'grid_shape':list(p.shape),'spacing_mm_array_axes':spacing,'evaluation_extent':'entire identical original grid',
        'input_channels':'separate binary masks; no cross-class overlap overwritten or discarded',
        'nsd_tolerance_mm':1.0,'nsd_tolerance_note':'Fixed exploratory tolerance chosen for this comparison; not a verified paper threshold.',
        'versions':{x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-image','nibabel','surface-distance']},
        'metric_sets':{},'not_computable':{'independent_GT_accuracy_change':'No independent reference annotation supplied.',
            'AUC':'No probability maps supplied.','airway_BDR_DLR':'No frozen reviewed reference branch-ID centerline map.',
            'airway_wall':'No airway-wall mask; lung is not wall.','MCS':'Supplemented arterial/venous masks are nonexclusive; standard single-label MCS is not computed.'}}

def save():
    report['elapsed_seconds']=time.time()-start
    (work/'results'/(structure+'_metrics.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

print('METRICS_START',structure,flush=True)
if structure!='lung':
    print('VESMAMBA',structure,flush=True)
    report['metric_sets']['vesmamba']=evaluate_vesmamba(p,g,spacing,1.0)
    save();gc.collect()
else:
    report['metric_sets']['vesmamba']={'status':'not_applied_to_lung_parenchyma','reason':'Lung is an auxiliary ROI structure, not a tubular vascular/airway endpoint. Its voxelwise equivalence and standard overlap scores are provided separately.'}

print('AIRRC',structure,flush=True)
air=evaluate_airrc(p,g,spacing,labels={structure:1},airway_label=1 if structure=='airway' else None)
if structure=='lung':
    air['scope_note']='Supplementary binary overlap on lung; not an AirRC airway-wall endpoint.'
report['metric_sets']['airrc']=air
save();gc.collect()

if structure in ('pulmonary_artery','pulmonary_vein'):
    prefix='artery' if structure=='pulmonary_artery' else 'vein'
    print('HIPAS',structure,flush=True)
    # Exactly one genuine binary channel is evaluated per call. The dummy absent
    # label enables the supplied API without inventing a priority for overlaps.
    # Discard the dummy class and MCS; only the genuine per-class results survive.
    artery_label,vein_label=(1,2) if prefix=='artery' else (2,1)
    raw=evaluate_hipas(p,g,spacing,artery_label=artery_label,vein_label=vein_label)
    report['metric_sets']['hipas']={
        'metrics':{k:v for k,v in raw['metrics'].items() if k.startswith(prefix+'_')},
        'status':{k:v for k,v in raw['status'].items() if k.startswith(prefix+'_')},
        'protocol':raw['protocol'],'notes':raw['notes'],
        'adapter_note':'Supplied evaluate_hipas called on one genuine binary channel (label 1); results for absent dummy label 2 and MCS are discarded. Other binary channels and overlaps are preserved in their own evaluations.',
        'MCS':None,'MCS_status':'not_computed_nonexclusive_av_masks'}
    metrics=report['metric_sets']['hipas']['metrics']
    report['actual_proxy_changes']={}
    for measure in ['BC_proxy','SL_voxels_proxy']:
        before=metrics[prefix+'_'+measure+'_pred'];after=metrics[prefix+'_'+measure+'_gt']
        report['actual_proxy_changes'][measure]={'before':before,'after':after,'delta_after_minus_before':after-before,
            'relative_change_percent':100*(after-before)/before if before else None,
            'interpretation':'Native-grid count proxy; increase is not evidence of improved accuracy or branch recall.'}
    # Meaningful consistency checks across independently implemented formulas.
    if abs(metrics[prefix+'_DSC']-air['metrics'][structure+'.DSC'])>1e-12: raise AssertionError('DSC mismatch across metric sets')
    if abs(metrics[prefix+'_SEN']-air['metrics'][structure+'.Recall'])>1e-12: raise AssertionError('Recall mismatch across metric sets')
    ves=report['metric_sets']['vesmamba']['metrics']
    if abs(ves['binary.Dice']-metrics[prefix+'_DSC'])>1e-12: raise AssertionError('Dice mismatch')
    if abs(ves['binary.HD95_mm']-metrics[prefix+'_HD95_mm'])>1e-6: raise AssertionError('HD95 mismatch')

report['complete']=True
save()
print('METRICS_COMPLETE',structure,json.dumps({k:v.get('metrics',{}) for k,v in report['metric_sets'].items()}),flush=True)
