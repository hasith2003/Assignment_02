"""EN3160 A02: reproducible experiments. Run: python assignment.py"""
from pathlib import Path
import json
import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'a2images' / 'a2images'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=170, bbox_inches='tight')
    plt.close()

def blobs():
    im = cv.imread(str(DATA/'the_berry_farms_sunflower_field.jpeg'), cv.IMREAD_REDUCED_COLOR_4)
    rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(float)/255
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype(float)/255
    yellow = np.maximum(0, np.minimum(rgb[:,:,0], rgb[:,:,1])-rgb[:,:,2])
    sigmas = np.geomspace(1.5, 18, 35)
    # Positive LoG extrema detect the dark central disks, rather than petals.
    stack = np.stack([s*s*gaussian_laplace(gray, s) for s in sigmas])
    maxima = (stack == maximum_filter(stack, size=(3,3,3))) & (stack > .10)
    maxima[[0,-1]] = False
    yy,xx = np.indices(gray.shape)
    candidates=[]
    for k,y,x in np.argwhere(maxima):
        r=float(np.sqrt(2)*sigmas[k])
        if x-r<0 or y-r<0 or x+r>=gray.shape[1] or y+r>=gray.shape[0]: continue
        distance=np.hypot(xx-x,yy-y)
        ring=(distance>r)&(distance<1.9*r)
        ring_yellow=float(yellow[ring].mean())
        if ring_yellow<.4: continue
        candidates.append((float(stack[k,y,x]),int(x),int(y),r,float(sigmas[k]),ring_yellow))
    kept=[]
    for c in sorted(candidates, reverse=True):
        if all(np.hypot(c[1]-d[1],c[2]-d[2])>.8*(c[3]+d[3]) for d in kept): kept.append(c)
    largest=sorted(kept,key=lambda c:c[3],reverse=True)[:5]
    fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(rgb); ax.axis('off')
    for c in kept: ax.add_patch(plt.Circle(c[1:3],c[3],fill=False,color='lime',lw=1))
    for i,c in enumerate(largest): ax.text(c[1],c[2],str(i+1),color='red',fontsize=10)
    ax.set_title(f'Scale-normalized LoG: {len(kept)} dark-centre detections')
    savefig('q1_blobs.png')
    fig,ax=plt.subplots(figsize=(8,3.6))
    for i,c in enumerate(largest):
        ax.plot(sigmas,stack[:,c[2],c[1]],label=f'{i+1}: ({c[1]}, {c[2]})')
        ax.scatter([c[4]],[c[0]],s=25)
    ax.axhline(.10,c='gray',ls=':',label='Response threshold')
    ax.set(xlabel='Sigma (reduced-image pixels)',ylabel='Scale-normalized LoG response')
    ax.legend(fontsize=8); savefig('q1_scale_profiles.png')
    return {'sigma_range':[float(sigmas[0]),float(sigmas[-1])], 'scales':len(sigmas),
            'threshold':.10,'yellow_ring_threshold':.4,'ring_outer_radius_factor':1.9,
            'suppression_distance_factor':.8,'target':'dark central disks surrounded by yellow petals',
            'detections':len(kept),'largest':[{'x':c[1],'y':c[2],'radius':c[3],'sigma':c[4],'response':c[0],'yellow_ring_mean':c[5]} for c in largest]}

def line_model(p):
    v=p[1]-p[0]
    if np.linalg.norm(v)<1e-10: return None
    n=np.array([-v[1],v[0]])/np.linalg.norm(v); d=n@p[0]
    if d<0: n,d=-n,-d
    return np.r_[n,d]

def circle_model(p):
    a=2*(p[1:]-p[0]); b=(p[1:]**2).sum(1)-(p[0]**2).sum()
    if abs(np.linalg.det(a))<1e-8: return None
    c=np.linalg.solve(a,b); return np.r_[c,np.linalg.norm(p[0]-c)]

def residual(x,m,kind):
    return abs(x@m[:2]-m[2]) if kind=='line' else abs(np.linalg.norm(x-m[:2],axis=1)-m[2])

def ransac(x,kind,seed,threshold=1.5,iterations=5000,min_consensus=30):
    rng=np.random.default_rng(seed); size=2 if kind=='line' else 3
    best=None; score=(-1,-np.inf)
    for _ in range(iterations):
        ids=rng.choice(len(x),size,replace=False)
        m=(line_model if kind=='line' else circle_model)(x[ids])
        if m is None: continue
        e=residual(x,m,kind); mask=e<threshold
        key=(int(mask.sum()),-float(np.sum(np.minimum(e,threshold)**2)))
        if key[0]>=min_consensus and key>score: best=(m,mask,ids); score=key
    if best is None: raise RuntimeError('No consensus')
    sample,mask,ids=best; pts=x[mask]
    if kind=='line':
        mean=pts.mean(0); _,_,vt=np.linalg.svd(pts-mean); n=vt[-1]; d=n@mean
        if d<0: n,d=-n,-d
        fit=np.r_[n,d]
    else:
        fit=least_squares(lambda m:np.linalg.norm(pts-m[:2],axis=1)-m[2],sample,bounds=([-np.inf,-np.inf,1e-8],[np.inf]*3)).x
    return sample,fit,mask,ids

def fitting():
    # Match Listing 1's legacy RNG exactly, with the suggested seed enabled.
    np.random.seed(0); n=50; t=np.random.uniform(0,2*np.pi,n); noise=.625*np.random.randn(n)
    circ=np.c_[2+(10+noise)*np.cos(t),3+(10+noise)*np.sin(t)]
    xx=np.linspace(-12,12,n); line=np.c_[xx,-xx+2+np.random.randn(n)]
    x=np.vstack([circ,line]); ls,lf,lmask,li=ransac(x,'line',10)
    remaining=x[~lmask]; cs,cf,cmask,ci=ransac(remaining,'circle',20,min_consensus=25)
    first=ransac(x,'circle',20,min_consensus=25)
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(*x.T,s=13,c='.7',label='All points')
    ax.scatter(*x[lmask].T,s=24,c='tab:blue',label='Line consensus')
    ax.scatter(*remaining[cmask].T,s=24,c='tab:orange',label='Circle consensus')
    u=np.array([-15,16])
    for m,style,label in [(ls,'--','Line: winning sample'),(lf,'-','Line: TLS refinement')]: ax.plot(u,(m[2]-m[0]*u)/m[1],style,label=label,lw=1.4)
    for m,style,color,label in [(cs,'--','purple','Circle: winning sample'),(cf,'-','red','Circle: radial refinement'),(np.array([2,3,10]),':','green','Ground truth circle')]: ax.add_patch(plt.Circle(m[:2],m[2],fill=False,ls=style,color=color,label=label))
    ax.plot(u,-u+2,':',color='black',label='Ground truth line')
    ax.scatter(*x[li].T,marker='s',s=100,facecolors='none',edgecolors='blue',label='Line sample (2)')
    ax.scatter(*remaining[ci].T,marker='^',s=130,facecolors='none',edgecolors='purple',label='Circle sample (3)')
    ax.set(xlim=(-15,17),ylim=(-14,18),aspect='equal',xlabel='x',ylabel='y'); ax.legend(fontsize=6,loc='upper left'); savefig('q2_ransac.png')
    np.savez(OUT/'q2_data.npz',points=x,line_mask=lmask,circle_remaining_mask=cmask)
    return {'seed':0,'threshold':1.5,'iterations':5000,'line_min_consensus':30,'circle_min_consensus':25,'line_sample_model':ls.tolist(),'line_refined':lf.tolist(),'line_consensus':int(lmask.sum()),'line_circle_contamination':int(lmask[:50].sum()),'circle_sample_model':cs.tolist(),'circle_refined':cf.tolist(),'circle_consensus':int(cmask.sum()),'circle_first_model':first[1].tolist(),'circle_first_consensus':int(first[2].sum()),'circle_first_line_contamination':int(first[2][50:].sum())}

def composite():
    base=cv.imread(str(DATA/'wadham.jpg')); flag=cv.imread(str(DATA/'flag.png'))
    dst=np.float32([[160,225],[338,262],[337,335],[159,304]])
    h,w=flag.shape[:2]; src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    H=cv.getPerspectiveTransform(src,dst)
    example_H=H.copy()
    warped=cv.warpPerspective(flag,H,(base.shape[1],base.shape[0]))
    mask=cv.warpPerspective(np.ones((h,w),np.float32),H,(base.shape[1],base.shape[0]))[:,:,None]
    result=np.uint8(base*(1-.8*mask)+warped*.8*mask)
    cv.imwrite(str(OUT/'q3_wadham.png'),result)
    records=[]
    pairs=[('pair1_background.jpg','pair1_overlay_mars.jpg',[[928,216],[1351,217],[1350,861],[928,860]],'Mars street display'),
           ('pair2_background.jpg','pair2_overlay_grand_tour.jpg',[[499,381],[1342,380],[1345,892],[499,892]],'Grand Tour wall display')]
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,(bg,fg,corners,title) in enumerate(pairs):
        base=cv.imread(str(ROOT/'my_images'/bg)); art=cv.imread(str(ROOT/'my_images'/fg))
        corner_file=ROOT/'my_images'/'corners.json'
        saved=json.loads(corner_file.read_text()) if corner_file.exists() else {}
        dst=np.float32(saved.get(bg,corners))
        # Landscape panel: preserve the entire portrait poster on a cream canvas.
        if row==1:
            canvas=np.full((900,1480,3),(220,235,246),np.uint8)
            scaled=cv.resize(art,(int(art.shape[1]*900/art.shape[0]),900))
            left=(1480-scaled.shape[1])//2; canvas[:,left:left+scaled.shape[1]]=scaled; art=canvas
        h,w=art.shape[:2]; src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        H=cv.getPerspectiveTransform(src,dst); size=(base.shape[1],base.shape[0])
        warped=cv.warpPerspective(art,H,size)
        mask=cv.warpPerspective(np.ones((h,w),np.float32),H,size)[:,:,None]
        # Slight transparency retains a little local illumination from the photo.
        alpha=.94; result=np.uint8(np.clip(base*(1-alpha*mask)+warped*alpha*mask,0,255))
        cv.imwrite(str(OUT/f'q3_pair{row+1}.jpg'),result)
        marked=base.copy()
        for i,p in enumerate(dst.astype(int)):
            cv.circle(marked,tuple(p),9,(0,0,255),-1); cv.putText(marked,str(i+1),tuple(p+12),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        for ax,img,label in zip(axes[row],[marked,art,result],['Selected corners','Source artwork',title]):
            ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)); ax.set_title(label,fontsize=10); ax.axis('off')
        records.append({'background':bg,'overlay':fg,'destination_points':dst.tolist(),'homography':H.tolist(),'alpha':alpha})
    savefig('q3_pairs.png')
    return {'example_homography':example_H.tolist(),'pairs':records,'corner_selection':'Manually selected from displayed photos; saved coordinates in TL, TR, BR, BL order.'}

def pcb():
    cv.setRNGSeed(0); template=cv.imread(str(DATA/'pcbs'/'AECC1.jpg')); gray=cv.cvtColor(template,cv.COLOR_BGR2GRAY)
    sift=cv.SIFT_create(nfeatures=12000); kt,dt=sift.detectAndCompute(gray,None); records=[]
    fig,axes=plt.subplots(5,3,figsize=(10,14))
    for row,path in enumerate(p for p in sorted((DATA/'pcbs').glob('*.jpg')) if p.name!='AECC1.jpg'):
        im=cv.imread(str(path)); ki,di=sift.detectAndCompute(cv.cvtColor(im,cv.COLOR_BGR2GRAY),None)
        pairs=cv.BFMatcher(cv.NORM_L2).knnMatch(di,dt,k=2); good=[a for a,b in pairs if a.distance<.75*b.distance]
        if len(good)<4: raise RuntimeError(f'Insufficient matches: {path.name}')
        src=np.float32([ki[m.queryIdx].pt for m in good]); dst=np.float32([kt[m.trainIdx].pt for m in good])
        H,inliers=cv.findHomography(src,dst,cv.RANSAC,3.0,maxIters=20000,confidence=.999)
        if H is None or inliers is None: raise RuntimeError(f'No homography: {path.name}')
        valid=inliers.ravel().astype(bool)
        if H is None or valid.sum()<10: raise RuntimeError(f'Unreliable homography: {path.name}')
        size=(template.shape[1],template.shape[0]); aligned=cv.warpPerspective(im,H,size)
        support=cv.warpPerspective(np.ones(im.shape[:2],np.uint8),H,size,flags=cv.INTER_NEAREST)
        support=cv.erode(support,np.ones((7,7),np.uint8)).astype(bool)
        diff=cv.absdiff(aligned,template); diff[~support]=0
        matches=cv.drawMatches(im,ki,template,kt,[good[i] for i in np.flatnonzero(valid)[:60]],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        stem=path.stem
        for name,img in [('aligned',aligned),('matches',matches)]: cv.imwrite(str(OUT/f'q4_{stem}_{name}.jpg'),img)
        cv.imwrite(str(OUT/f'q4_{stem}_difference.png'),diff)
        cv.imwrite(str(OUT/f'q4_{stem}_support.png'),support.astype(np.uint8)*255)
        for ax,img,title in zip(axes[row],[matches,aligned,diff],[f'{stem}: inlier matches','Aligned','Absolute difference']): ax.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB)); ax.set_title(title,fontsize=9); ax.axis('off')
        projected=cv.perspectiveTransform(src.reshape(-1,1,2),H).reshape(-1,2)
        records.append({'image':path.name,'ratio_matches':len(good),'ransac_inliers':int(valid.sum()),'median_reprojection_px':float(np.median(np.linalg.norm(projected[valid]-dst[valid],axis=1))),'overlap_fraction':float(support.mean()),'overlap_mae':float(diff[support].mean()),'homography':H.tolist()})
    savefig('q4_overview.png')
    cv.imwrite(str(OUT/'q4_AECC1_difference.png'),np.zeros_like(template))
    return records

if __name__=='__main__':
    metrics={}
    for name,fn in [('q1',blobs),('q2',fitting),('q3',composite),('q4',pcb)]:
        print(f'Running {name}...',flush=True); metrics[name]=fn()
        (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Results written to',OUT)
