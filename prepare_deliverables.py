"""Generate supplementary figures, notebook and numerical validation."""
from pathlib import Path
import json, platform, inspect
import cv2 as cv
import numpy as np
import scipy, matplotlib, reportlab
import matplotlib.pyplot as plt
import assignment as a
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
m=json.loads((OUT/'metrics.json').read_text())

data=np.load(OUT/'q2_data.npz'); x=data['points']
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,title,fit,mask in [(axes[0],'Line first, then circle',m['q2']['circle_refined'],None),
                         (axes[1],'Circle first',m['q2']['circle_first_model'],a.ransac(x,'circle',20,min_consensus=25)[2])]:
    ax.scatter(*x[:50].T,c='tab:orange',s=15,label='True circle points')
    ax.scatter(*x[50:].T,c='tab:blue',s=15,label='True line points')
    ax.add_patch(plt.Circle((2,3),10,fill=False,ls=':',color='green',label='True circle'))
    ax.add_patch(plt.Circle(fit[:2],fit[2],fill=False,color='red',label='Fitted circle'))
    if mask is not None:
        ids=np.flatnonzero(mask & (np.arange(100)>=50))
        ax.scatter(*x[ids].T,facecolors='none',edgecolors='black',s=75,label='Line points removed')
    ax.set(xlim=(-15,17),ylim=(-14,18),aspect='equal',title=title,xlabel='x',ylabel='y')
    ax.legend(fontsize=7)
a.savefig('q2_order_comparison.png')

names=[Path(r['image']).stem for r in m['q4']]
fig,axes=plt.subplots(5,2,figsize=(11,13))
for row,name in enumerate(names):
    for col,kind in enumerate(['matches','aligned']):
        im=cv.imread(str(OUT/f'q4_{name}_{kind}.jpg'))
        axes[row,col].imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
        axes[row,col].set_title(f'{name}: '+('RANSAC inlier matches' if col==0 else 'Aligned to AECC1'),fontsize=10)
        axes[row,col].axis('off')
a.savefig('q4_registration.png')

fig,axes=plt.subplots(2,5,figsize=(13,6.8))
for col,name in enumerate(names):
    raw=cv.imread(str(OUT/f'q4_{name}_difference.png'))
    for row,gain in enumerate([1,4]):
        axes[row,col].imshow(cv.cvtColor(np.clip(raw.astype(float)*gain,0,255).astype(np.uint8),cv.COLOR_BGR2RGB))
        axes[row,col].set_title(name+(' raw' if row==0 else ' (4× gain)'),fontsize=10)
        axes[row,col].axis('off')
a.savefig('q4_differences.png')

fig=plt.figure(figsize=(12,5.5))
grid=fig.add_gridspec(2,5,height_ratios=[1.1,1])
ax=fig.add_subplot(grid[0,:])
im=cv.imread(str(OUT/'q4_AECC2_matches.jpg')); ax.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB));ax.axis('off')
ax.set_title('AECC2 → AECC1: up to 60 inlier correspondences',fontsize=10)
for col,name in enumerate(names):
    ax=fig.add_subplot(grid[1,col]);raw=cv.imread(str(OUT/f'q4_{name}_difference.png'))
    ax.imshow(cv.cvtColor(np.clip(raw.astype(float)*4,0,255).astype(np.uint8),cv.COLOR_BGR2RGB))
    ax.set_title(name+' (4×)',fontsize=10);ax.axis('off')
a.savefig('q4_compact.png')

checks={}
unit=np.linalg.norm(m['q2']['line_refined'][:2])
checks['unit_line_normal']=bool(abs(unit-1)<1e-12)
fit=np.array(m['q2']['circle_refined'])
checks['circle_center_within_one_unit']=bool(np.linalg.norm(fit[:2]-[2,3])<1)
checks['circle_radius_within_one_unit']=bool(abs(fit[2]-10)<1)
checks['exact_line_sample']=bool(np.max(a.residual(np.array([[0.,2.],[2.,0.]]),a.line_model(np.array([[0.,2.],[2.,0.]])),'line'))<1e-12)
checks['exact_circle_sample']=bool(np.allclose(a.circle_model(np.array([[12.,3.],[2.,13.],[-8.,3.]])),[2,3,10]))
checks['collinear_circle_rejected']=a.circle_model(np.array([[0.,0.],[1.,1.],[2.,2.]])) is None
corner_errors=[]
for i,p in enumerate(m['q3']['pairs']):
    im=cv.imread(str(ROOT/'my_images'/p['overlay']))
    h,w=im.shape[:2]
    if i==1:h,w=900,1480
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    projected=cv.perspectiveTransform(src.reshape(-1,1,2),np.array(p['homography'])).reshape(-1,2)
    corner_errors.append(float(np.max(np.linalg.norm(projected-np.array(p['destination_points']),axis=1))))
checks['homography_corner_error_below_1e_3_px']=max(corner_errors)<1e-3
checks['five_non_template_pcbs']=len(m['q4'])==5
checks['all_pcb_median_errors_below_3px']=all(r['median_reprojection_px']<3 for r in m['q4'])
checks['all_pcb_homographies_finite_invertible']=all(np.isfinite(r['homography']).all() and abs(np.linalg.det(r['homography']))>1e-8 for r in m['q4'])
template=cv.imread(str(a.DATA/'pcbs/AECC1.jpg'))
checks['zero_template_self_difference']=bool(not cv.imread(str(OUT/'q4_AECC1_difference.png')).any())
checks['lossless_maps_and_support']=True
for name in names:
    diff=cv.imread(str(OUT/f'q4_{name}_difference.png'))
    support=cv.imread(str(OUT/f'q4_{name}_support.png'),cv.IMREAD_GRAYSCALE)>0
    checks['lossless_maps_and_support'] &= diff.shape==template.shape and not diff[~support].any()
validation={'checks':checks,'all_passed':bool(all(checks.values())),'corner_errors_px':corner_errors,
            'line_normal_norm':float(unit),'versions':{'python':platform.python_version(),'opencv':cv.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'reportlab':reportlab.Version},
            'notes':['AAAA1 has few inliers; global alignment has lower confidence.','Image quality and defect labels are not established by these numerical checks.']}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2))
if not all(checks.values()):raise RuntimeError('A validation check failed')

print(json.dumps(validation,indent=2))
print('Supplementary figures and validation generated. Notebook outputs are preserved.')
