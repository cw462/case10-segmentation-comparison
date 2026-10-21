"""Evaluate initial masks against the user's designated supplemented ground truth."""
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

root = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
results = root/'results'
names = {'pulmonary_artery':'肺动脉','pulmonary_vein':'肺静脉','airway':'气道','lung':'肺'}
records=[]
for code,label in names.items():
    r=json.loads((results/(code+'_metrics.json')).read_text(encoding='utf-8'))
    if code=='lung':
        lung=json.loads((results/'lung_reference_metrics.json').read_text(encoding='utf-8'))
        # Accept the explicit raw-result wrapper, preserving source provenance.
        v=lung.get('package_result',lung.get('vesmamba',lung.get('result',lung)))
        v=v['metrics']
    else:
        v=r['metric_sets']['vesmamba']['metrics']
    a=r['metric_sets']['airrc']['metrics']
    records.append({'structure':code,'structure_zh':label,
                    'Dice':v['binary.Dice'],'clDice':v['binary.clDice'],
                    'NSD_1mm':v['binary.NSD'],'HD95_mm':v['binary.HD95_mm'],
                    'Precision':a[code+'.Precision'],'Recall':a[code+'.Recall'],'FPR':a[code+'.FPR']})

report={'created':datetime.now().isoformat(timespec='seconds'),
        'prediction':'before_initial_annotation','ground_truth':'after_supplemented_annotation',
        'reference_role':'user_designated_supplemented_ground_truth',
        'ground_truth_definition_zh':'按用户指定，将补标后版本视为本次评测的绝对真值（GT），评估补标前结果。',
        'nsd_tolerance_mm':1.0,'structures':records,
        'notes_zh':[
            '动脉、静脉、气道沿用已完成的同方向计算，肺的Dice/clDice/HD95/NSD由相同原始函数补算。',
            'GT为用户指定的补标版本；这些指标衡量补标前结果与该真值的吻合程度。',
            '气道、肺两版逐体素完全相同，所以对该真值的Dice、clDice、NSD、Precision、Recall为100%。',
            '肺的clDice按同一骨架公式补算，仅供统一表格比较；解释管状连通性时主要关注气道、血管。',
            'HD95按表面元面积加权的双向95%分位数取最大值；少量局部修改时可为0。',
            '动脉、静脉、气道和肺分别按独立二值通道计算，未覆盖标签重叠。',
            'NSD使用1.0 mm阈值；全病例同一原生网格，不重采样。']}
(results/'user_designated_gt_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
headers=['结构（补标前预测）','Dice','clDice','NSD（1 mm）','HD95（mm）','Precision','Recall','FPR']
rows=[[r['structure_zh'],r['Dice'],r['clDice'],r['NSD_1mm'],r['HD95_mm'],r['Precision'],r['Recall'],r['FPR']] for r in records]
with (results/'以补标后为GT_补标前指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
book=Workbook();sheet=book.active;sheet.title='补标前相对补标后GT'
sheet.append(headers)
for row in rows: sheet.append(row)
sheet.freeze_panes='B2';sheet.auto_filter.ref=sheet.dimensions
sheet.sheet_view.showGridLines=False
for col in 'ABCDEFGH': sheet.column_dimensions[col].width=23 if col=='A' else 18
sheet.row_dimensions[1].height=32
for cell in sheet[1]:
    cell.font=Font(name='微软雅黑',bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='17365D');cell.alignment=Alignment(wrap_text=True,vertical='center')
for row in sheet.iter_rows(min_row=2):
    sheet.row_dimensions[row[0].row].height=26
    for cell in row:
        cell.font=Font(name='微软雅黑',size=11)
        if cell.column>1: cell.number_format='0.000' if cell.column==5 else ('0.000000%' if cell.column==8 else '0.00%')
notes=book.create_sheet('真值定义与说明');notes.column_dimensions['A'].width=125
for text in [report['ground_truth_definition_zh']]+report['notes_zh']:
    notes.append([text]);cell=notes.cell(notes.max_row,1);cell.alignment=Alignment(wrap_text=True,vertical='center');cell.font=Font(name='微软雅黑',size=11);notes.row_dimensions[notes.max_row].height=42
dest=root/'Case10_以补标后为GT_补标前指标.xlsx';book.save(dest)
check=load_workbook(dest,read_only=True,data_only=True)
assert check['补标前相对补标后GT'].max_row==5
for i,r in enumerate(records,start=2):
    assert abs(check['补标前相对补标后GT'].cell(i,2).value-r['Dice'])<1e-12
check.close()
print(json.dumps(report,ensure_ascii=False,indent=2))
