"""Create tiny SYNTHETIC examples only; not patient CT or paper results."""
from pathlib import Path
import numpy as np
import nibabel as nib


def main():
    output = Path(__file__).parent / 'demo'
    output.mkdir(exist_ok=True)
    gt = np.zeros((24, 24, 24), np.uint8)
    gt[3:20, 5:8, 5:8] = 2
    gt[3:20, 6, 6] = 1
    gt[3:20, 12, 12] = 3
    gt[3:20, 18, 18] = 4
    before, after = gt.copy(), gt.copy()
    before[15:20, 6, 6] = 0
    before[10:14, 12, 12] = 0
    before[10:12, 18, 18] = 3
    after[19, 6, 6] = 0
    after[11, 12, 12] = 0
    after[10, 18, 18] = 3
    branches = np.zeros_like(gt)
    branches[3:20, 6, 6] = 1
    items = {'gt': gt, 'initial': before, 'corrected': after, 'gt_branch_ids': branches,
        'artery_prob': np.where(after == 3, .9, .05).astype(np.float32),
        'vein_prob': np.where(after == 4, .9, .05).astype(np.float32)}
    for name, array in items.items():
        image = nib.Nifti1Image(array, np.diag([.8, .8, 1., 1.]))
        image.header.set_xyzt_units('mm')
        nib.save(image, output / (name + '.nii.gz'))
    print(f'Synthetic example masks saved to {output}')


if __name__ == '__main__':
    main()
