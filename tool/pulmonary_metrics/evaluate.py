"""Evaluate VesMamba, HiPaS or AirRC metric sets on hard label masks."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from io_utils import load_volume, check_grids
from vesmamba_metrics import evaluate_vesmamba


def parse_classes(items):
    result = {}
    for item in items:
        try:
            name, number = item.split('=', 1)
            label = int(number)
        except ValueError as e:
            raise ValueError('classes require name=positive_integer, e.g. artery=3') from e
        if not name or name in result or label <= 0 or label in result.values():
            raise ValueError('class names and positive IDs must be unique')
        result[name] = label
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper', choices=['vesmamba', 'hipas', 'airrc'], required=True)
    ap.add_argument('--pred', help='one prediction mask')
    ap.add_argument('--before', help='initial mask; use together with --after')
    ap.add_argument('--after', help='corrected mask; use together with --before')
    ap.add_argument('--gt', required=True, help='fixed reference, in the same physical grid')
    ap.add_argument('--spacing', type=float, nargs=3, metavar=('S0', 'S1', 'S2'))
    ap.add_argument('--assume-mm', action='store_true', help='explicitly interpret unknown NIfTI spatial units as mm')
    ap.add_argument('--nsd-mm', type=float, help='required for VesMamba; no paper threshold is assumed')
    ap.add_argument('--labels', type=int, nargs='+', help='VesMamba labels, independently evaluated; omit only for binary masks')
    ap.add_argument('--artery-label', type=int, help='HiPaS artery ID; required')
    ap.add_argument('--vein-label', type=int, help='HiPaS vein ID; required')
    ap.add_argument('--artery-prob', help='HiPaS artery probability NIfTI/NumPy, only with --pred; enables ROC AUC')
    ap.add_argument('--vein-prob', help='HiPaS vein probability NIfTI/NumPy, only with --pred; enables ROC AUC')
    ap.add_argument('--classes', nargs='+', help='AirRC name=ID; default airway=1 wall=2 artery=3 vein=4')
    ap.add_argument('--airway-label', type=int, help='AirRC lumen ID; default 1 with the default class map')
    ap.add_argument('--branch-labels', help='fixed reference airway skeleton labelled by branch ID, same grid')
    ap.add_argument('--output', required=True, help='JSON filename; companion CSV is also saved')
    args = ap.parse_args(argv)
    try:
        if args.pred:
            if args.before or args.after:
                raise ValueError('use --pred OR --before with --after')
            paths = {'prediction': args.pred, 'gt': args.gt}
        else:
            if not args.before or not args.after:
                raise ValueError('supply --pred, or both --before and --after')
            paths = {'before': args.before, 'after': args.after, 'gt': args.gt}
        output = Path(args.output)
        if output.suffix.lower() != '.json':
            raise ValueError('--output must end in .json')
        if args.paper == 'vesmamba' and args.nsd_mm is None:
            raise ValueError('VesMamba requires an explicit --nsd-mm; the paper main text does not specify it')
        if args.paper == 'hipas' and (args.artery_label is None or args.vein_label is None):
            raise ValueError('HiPaS requires --artery-label and --vein-label')
        if args.branch_labels:
            if args.paper != 'airrc':
                raise ValueError('--branch-labels is only used by the AirRC airway branch evaluator')
            paths['branch_labels'] = args.branch_labels
        if args.artery_prob or args.vein_prob:
            if args.paper != 'hipas' or not args.pred:
                raise ValueError('probability inputs are only supported for a single --pred with --paper hipas')
            if args.artery_prob:
                paths['artery_probability'] = args.artery_prob
            if args.vein_prob:
                paths['vein_probability'] = args.vein_prob
        loaded = {name: load_volume(path, args.assume_mm, 'probability' if name.endswith('_probability') else 'labels') for name, path in paths.items()}
        spacing = check_grids(loaded, args.spacing)
        gt = loaded['gt']['data']
        notes = [n for item in loaded.values() for n in item['notes']]
        classes = parse_classes(args.classes) if args.classes else {'airway': 1, 'wall': 2, 'artery': 3, 'vein': 4}
        airway_label = args.airway_label if args.airway_label is not None else (None if args.classes else 1)
        if args.paper == 'airrc' and args.branch_labels and airway_label is None:
            raise ValueError('with custom --classes and branch labels, specify --airway-label')
        if args.paper != 'vesmamba' and (args.labels or args.nsd_mm is not None):
            raise ValueError('--labels/--nsd-mm apply only to VesMamba')
        if args.paper != 'hipas' and (args.artery_label is not None or args.vein_label is not None):
            raise ValueError('--artery-label/--vein-label apply only to HiPaS')
        if args.paper != 'airrc' and (args.classes or args.airway_label is not None):
            raise ValueError('--classes/--airway-label apply only to AirRC')
        results = {}
        for name, item in loaded.items():
            if name in ('gt', 'branch_labels', 'artery_probability', 'vein_probability'):
                continue
            pred = item['data']
            if args.paper == 'vesmamba':
                result = evaluate_vesmamba(pred, gt, spacing, args.nsd_mm, args.labels)
            elif args.paper == 'hipas':
                from hipas_metrics import evaluate_hipas
                result = evaluate_hipas(pred, gt, spacing, args.artery_label, args.vein_label,
                    artery_probability=loaded.get('artery_probability', {}).get('data'),
                    vein_probability=loaded.get('vein_probability', {}).get('data'))
            else:
                from airrc_metrics import evaluate_airrc
                branches = loaded['branch_labels']['data'] if 'branch_labels' in loaded else None
                result = evaluate_airrc(pred, gt, spacing, classes, airway_label, branches)
            results[name] = result
            if np.array_equal(pred, gt):
                notes.append(f'{name} equals GT voxel-for-voxel; an after-vs-itself score is not independent evidence of improvement.')
        changes = {}
        if 'before' in results:
            for metric, before in results['before']['metrics'].items():
                after = results['after']['metrics'].get(metric)
                changes[metric] = {'before': before, 'after': after,
                    'delta_after_minus_before': None if before is None or after is None else float(after - before)}
        versions = {}
        for package in ['numpy', 'scipy', 'scikit-image', 'nibabel', 'surface-distance']:
            versions[package] = importlib.metadata.version(package)
        report = {'paper_metric_set': args.paper,
            'implementation_notice': 'Independent implementation with explicit protocols; not an exact reproduction of undisclosed author evaluation code.',
            'inputs': paths, 'spacing_mm_array_axes': spacing, 'package_versions': versions,
            'results': results, 'changes': changes, 'notes': notes,
            'comparison_rule': 'Deltas are after minus before. Counts and count ratios do not by themselves establish segmentation accuracy.'}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        with output.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['stage', 'metric', 'value', 'status', 'delta_after_minus_before'])
            writer.writeheader()
            for stage, result in results.items():
                for metric, value in result['metrics'].items():
                    writer.writerow({'stage': stage, 'metric': metric, 'value': value,
                        'status': result.get('status', {}).get(metric, ''),
                        'delta_after_minus_before': changes.get(metric, {}).get('delta_after_minus_before') if stage == 'after' else ''})
        print(output)
        print(output.with_suffix('.csv'))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        ap.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
