"""Build detailed and four-page PDF reports from measured experiment results."""
from pathlib import Path
import json, html, textwrap
import numpy as np
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
M=json.loads((OUT/'metrics.json').read_text())
FONT=Path(matplotlib.get_data_path())/'fonts'/'ttf'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
W,H=A4
PROFILE='https://github.com/hasith2003'
REPO=PROFILE+'/Assignment_02'

class Report:
    def __init__(self,path,detailed=False):
        self.c=canvas.Canvas(str(path),pagesize=A4)
        self.c.setTitle('EN3160 Assignment 2 — NAHG Nettikumara — 230436X')
        self.c.setAuthor('NAHG Nettikumara')
        self.y=H-42; self.n=0; self.detailed=detailed
    def page(self,title):
        if self.n: self.end()
        self.n+=1; self.y=H-40
        self.c.setFillColor(colors.HexColor('#12364b'))
        self.c.setFont('Bold',15); self.c.drawString(36,self.y,title); self.y-=22
        self.p('NAHG Nettikumara | 230436X | EN3160 — Fitting and Alignment',8.2)
        if self.n==1:
            self.p(f'GitHub: <link href="{PROFILE}" color="#17669c">{PROFILE}</link> · <link href="{REPO}" color="#17669c">Assignment repository</link>',8)
        self.y-=5
    def p(self,text,size=9.2):
        style=ParagraphStyle('p',fontName='Body',fontSize=size,leading=size*1.4,textColor=colors.HexColor('#172d38'),spaceAfter=0)
        p=Paragraph(text,style); _,h=p.wrap(W-72,1000)
        p.drawOn(self.c,36,self.y-h); self.y-=h+7
        self.check()
    def heading(self,text):
        self.p('<font name="Bold">'+text+'</font>',10)
    def image(self,name,height):
        path=OUT/name
        iw,ih=PILImage.open(path).size
        scale=min((W-72)/iw,height/ih)
        dw,dh=iw*scale,ih*scale
        self.c.drawImage(str(path),(W-dw)/2,self.y-dh,width=dw,height=dh,mask='auto')
        self.y-=dh+8; self.check()
    def table(self,rows,widths=None,size=8):
        style=ParagraphStyle('t',fontName='Body',fontSize=size,leading=size*1.3)
        rows=[[Paragraph(html.escape(str(x)),style) for x in row] for row in rows]
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#7e9bae'))]))
        _,h=t.wrap(W-72,1000); t.drawOn(self.c,36,self.y-h); self.y-=h+10; self.check()
    def code(self,text):
        lines=textwrap.dedent(text).strip().splitlines()
        self.c.setFillColor(colors.HexColor('#edf3f6'))
        self.c.rect(36,self.y-len(lines)*9-12,W-72,len(lines)*9+12,fill=1,stroke=0)
        self.c.setFillColor(colors.HexColor('#163347')); self.c.setFont('Mono',6.9)
        for line in lines:
            self.y-=9; self.c.drawString(43,self.y,line)
        self.y-=17; self.check()
    def check(self):
        if self.y<38: raise RuntimeError(f'Page {self.n} overflows: y={self.y}')
    def end(self):
        self.c.setFont('Body',7); self.c.setFillColor(colors.grey)
        self.c.drawString(36,23,'230436X | Reproducible results: assignment.py + results/metrics.json')
        self.c.drawRightString(W-36,23,str(self.n)); self.c.showPage()
    def finish(self):
        self.end(); self.c.save()

def largest_rows():
    return [['Rank','x (px)','y (px)','σ (px)','Radius (px)']]+[
        [i+1,c['x'],c['y'],f"{c['sigma']:.2f}",f"{c['radius']:.2f}"] for i,c in enumerate(M['q1']['largest'])]

def pcb_rows():
    return [['Image','Ratio matches','H inliers','Median error px','Overlap MAE']]+[
        [r['image'],r['ratio_matches'],r['ransac_inliers'],f"{r['median_reprojection_px']:.3f}",f"{r['overlap_mae']:.2f}"] for r in M['q4']]

def q1_method(r):
    q=M['q1']
    r.p(f"The supplied JPEG was loaded with IMREAD_REDUCED_COLOR_4 (360 × 360 pixels). Dark flower centres were detected in grayscale I, scaled to [0,1]. The response L(x,y,σ) = σ²∇²(Gσ * I) was evaluated at {q['scales']} logarithmically spaced scales, σ = 1.5–18 pixels.")
    r.p(f"Positive local maxima were selected in a 3 × 3 × 3 position/scale neighbourhood above {q['threshold']}; boundary scales and circles crossing the image edge were rejected. A yellow-petal prior requires mean max(0,min(R,G)−B) ≥ 0.4 in the annulus r to 1.9r. Greedy suppression rejects candidates within 0.8(r₁+r₂) of a stronger detection. Radius r = √2σ gives {q['detections']} candidate central disks.")

def q1_discussion(r):
    r.p("The red labels identify the five largest retained circles in decreasing radius. Coordinates use the reduced image, with origin at the top left; original-image coordinates and radii are approximately four times larger. Circles describe the dark central disks rather than the outer petal boundary. Leaves surrounded by yellow petals, touching flowers and occlusion can produce false positives or missed centres. These are detections, not a validated flower count.")

def q2_method(r):
    q=M['q2']; a,b,d=q['line_refined']; cx,cy,rad=q['circle_refined']
    r.p("Listing 1 was reproduced with seed 0: 50 noisy circle points and 50 noisy line points. Custom RANSAC samples two points for a line and three for a circle, rejecting degenerate samples. The line is ax + by = d with a²+b² = 1 and d ≥ 0. Line error is |ax+by−d|; circle error is |‖p−c‖−r|.")
    r.p(f"Both thresholds are 1.5 coordinate units; 5,000 hypotheses are tried per model, with minimum consensuses of 30 (line) and 25 (circle). Largest consensus wins; truncated squared error breaks ties. The winning line consensus is removed before circle fitting. Total least squares refines the line; bounded radial least squares refines the circle on its winning consensus.")
    r.table([['Model','Refined parameters','Consensus'],['Line',f'a={a:.4f}, b={b:.4f}, d={d:.4f}',q['line_consensus']],['Circle',f'cx={cx:.4f}, cy={cy:.4f}, r={rad:.4f}',q['circle_consensus']]],widths=[55,355,113])

def q2_discussion(r):
    q=M['q2']; f=q['circle_first_model']
    r.p(f"The line consensus includes {q['line_circle_contamination']} true circle points near the intersections. The reported consensuses are the winning sample's sets used for refinement; membership is not relabelled afterward. If the circle is fit first, this run returns centre ({f[0]:.2f}, {f[1]:.2f}), radius {f[2]:.2f}, with {q['circle_first_consensus']} consensus points including {q['circle_first_line_contamination']} true line points.")
    r.p("A large-radius circle can approximate a noisy line over a short arc, and a flexible circle can also absorb points near line/circle intersections. Thus fitting the circle first can remove useful line points or select a spurious large circle. It need not fail every time: the outcome depends on noise, thresholds and samples.")

def q3_method(r):
    r.p("Four corners were selected manually from each displayed background and stored in top-left, top-right, bottom-right, bottom-left order. A homography H maps artwork corners to these points: λ[x′,y′,1]ᵀ = H[x,y,1]ᵀ. cv.getPerspectiveTransform solves the four correspondences; cv.warpPerspective resamples the artwork and a support mask into the photograph.")
    r.p("Both examples advertise imaginary space tourism: a Mars poster in a street display, and The Grand Tour on a wall display. The portrait Grand Tour artwork is centred on a cream landscape canvas before warping to preserve its proportions. Inside the warped mask, I = 0.94W + 0.06B; outside it, I = B. The supplied Wadham/flag example was also reproduced.")

def q3_discussion(r):
    r.p("The displays are nearly frontal, so these examples show modest projective distortion. A single homography assumes one plane and does not model fabric wrinkles, surface curvature or foreground occluders. Alpha blending retains a little background illumination but does not physically reproduce shadows, glare or folds. Corner selection is a source of edge-placement error.")

def sources(r):
    r.p('Backgrounds: Julia Filirovska, <link href="https://www.pexels.com/photo/4913828/" color="#17669c">Pexels 4913828</link>; Timothy Huliselan, <link href="https://www.pexels.com/photo/27467330/" color="#17669c">Pexels 27467330</link>. Artwork: NASA/JPL, <link href="https://www.jpl.nasa.gov/images/mars-jpl-travel-poster/" color="#17669c">Mars</link> and <link href="https://www.jpl.nasa.gov/images/the-grand-tour-jpl-travel-poster/" color="#17669c">The Grand Tour</link>. Downloaded 19 October 2026; full URLs and credits: my_images/SOURCES.md.',7.5)

def q4_method(r):
    r.p("AECC1.jpg is the template. SIFT (up to 12,000 features) is applied to grayscale images at their original resolution. BFMatcher with L2 distance finds two neighbours; Lowe's ratio test retains d₁ &lt; 0.75d₂. A RANSAC homography maps each input to template coordinates using a 3-pixel reprojection threshold, up to 20,000 iterations and confidence 0.999.")
    r.p("The input is warped to the template size. Raw absolute BGR differences D = |Ialigned − Itemplate| are computed in uint8. A separately warped support mask, eroded by 3 pixels, excludes unsupported borders from the figures and mean absolute error (MAE). MAE is averaged across valid pixels and colour channels; it is an intensity difference in [0,255], not a defect rate.")

def q4_discussion(r):
    r.p("Large differences on the USB Type B port, capacitors and reset button can arise from height above the PCB plane: viewpoint changes cause parallax that one planar homography cannot remove. Specular reflections on metal, shadows, illumination changes and occlusion also contribute. Real component/layout differences are another possibility; the difference map alone cannot separate these causes.")
    r.p("AAAA1 has fewer stable correspondences and a much smaller inlier fraction than the AECC images, so its alignment deserves extra caution. Low inlier reprojection error supports alignment of matched features, but does not prove every component is registered. A global image MAE also includes background and lighting changes. Reliable defect detection would need board-region masking, photometric normalization and local/component-level verification.")

def build_compact():
    r=Report(ROOT/'230436X_a02.pdf')
    r.page('1  Sunflower detection using scale-space LoG')
    q1_method(r); r.image('q1_blobs.png',285)
    r.table(largest_rows(),widths=[65,105,105,120,128],size=7.8)
    r.code("response = sigma**2 * gaussian_laplace(gray, sigma)\npeaks = (stack == maximum_filter(stack, size=(3,3,3))) & (stack > .10)\nradius = np.sqrt(2) * sigma")
    q1_discussion(r)
    r.page('2  Sequential RANSAC: line, then circle')
    q2_method(r); r.image('q2_ransac.png',290)
    r.code("line = ransac(X, 'line', seed=10)\nremaining = X[~line[2]]\ncircle = ransac(remaining, 'circle', seed=20, min_consensus=25)")
    q2_discussion(r)
    r.page('3  Planar image compositing')
    q3_method(r); r.image('q3_pairs.png',300)
    r.code("H = cv.getPerspectiveTransform(source_corners, target_corners)\nW = cv.warpPerspective(artwork, H, background_size)\nM = cv.warpPerspective(np.ones((h,w), np.float32), H, background_size)\nresult = background*(1 - .94*M[...,None]) + .94*W*M[...,None]")
    q3_discussion(r); sources(r)
    r.page('4  PCB feature alignment and differences')
    q4_method(r)
    r.image('q4_compact.png',190)
    r.p('Top: AECC2 inlier matches. Bottom: all five non-template differences, displayed at 4× gain for visibility; raw lossless maps are saved separately.',7)
    r.table(pcb_rows(),widths=[72,100,85,130,136],size=7.2)
    r.code("good = [a for a,b in matcher.knnMatch(desc, template_desc, k=2)\n        if a.distance < .75*b.distance]\nH, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.0)\ndifference = cv.absdiff(cv.warpPerspective(image, H, size), template)")
    q4_discussion(r); r.finish()

def build_detailed():
    r=Report(ROOT/'230436X_a02_detailed.pdf',True)
    r.page('1  Detecting sunflower heads')
    q1_method(r); r.image('q1_blobs.png',360)
    r.table(largest_rows(),widths=[65,105,105,120,128])
    q1_discussion(r)
    r.page('2  LoG implementation and scale selection')
    r.heading('Why normalize by scale?')
    r.p("The Laplacian measures spatial curvature. Smoothing suppresses detail smaller than σ; multiplying the Laplacian by σ² makes responses at different scales comparable. For an ideal dark disk, the normalized response at its centre peaks at σ = r/√2. Real petals and dark centres violate this ideal, so the radius is an effective blob radius.")
    r.code("gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(float) / 255\nsigmas = np.geomspace(1.5, 18, 35)\nstack = np.stack([s*s*gaussian_laplace(gray, s) for s in sigmas])\npeaks = (stack == maximum_filter(stack, size=(3,3,3))) & (stack > .10)\npeaks[[0,-1]] = False\nradius = np.sqrt(2) * sigmas[scale_index]")
    r.image('q1_scale_profiles.png',285)
    r.p("The profiles plot the actual scale responses at the five largest detections. Peaks are selected jointly across position and scale; a profile at a fixed pixel can have other extrema that are not joint maxima. Excluding the first and last scales avoids reporting unsupported scale-boundary extrema.")
    r.heading('Parameter choices and limitations')
    r.p("σ = 1.5–18 corresponds to nominal radii 2.12–25.46 pixels at quarter resolution. Thirty-five geometric samples provide similar relative scale resolution throughout the interval. The positive threshold 0.10 suppresses weak texture responses. The yellow-annulus threshold 0.4 is a colour prior, not a learned classifier. Strong overlap suppression reduces duplicate rings but can merge neighbouring heads. No Hough circle detector is used.")
    r.p("A smaller threshold increases both weak flower detections and texture false positives. Increasing the minimum scale loses distant flowers; increasing the maximum scale can admit dark leaf regions. Detecting dark centres avoids treating each bright petal as a separate flower; the colour prior rejects many dark foliage regions. No annotated ground truth was supplied, so precision and recall are not claimed.")
    r.page('3  RANSAC line and circle estimates')
    q2_method(r); r.image('q2_ransac.png',365); q2_discussion(r)
    r.page('4  Model fitting details and ordering')
    r.heading('Minimal samples, residuals and refinement')
    r.p("A two-point direction v gives a unit normal n = (−vᵧ,vₓ)/‖v‖ and d = nᵀp. Flipping n,d if d is negative removes the sign ambiguity. TLS takes the right singular vector of the centred consensus matrix with smallest singular value as the normal. This minimizes squared orthogonal distances while enforcing the unit-normal constraint.")
    r.code("v = p[1] - p[0]\nn = np.array([-v[1], v[0]]) / np.linalg.norm(v)\nd = n @ p[0]\nline_error = np.abs(X @ n - d)\nA = 2*(p[1:] - p[0])\nb = (p[1:]**2).sum(axis=1) - (p[0]**2).sum()\ncentre = np.linalg.solve(A, b)\ncircle_error = np.abs(np.linalg.norm(X-centre, axis=1) - radius)")
    r.p("Three non-collinear circle points determine a centre by subtracting squared-distance equations. Degenerate 2×2 systems are rejected. Circle refinement minimizes the sum of squared signed radial residuals over the consensus, with a strictly positive radius bound; it is a nonlinear least-squares optimization with three unknowns.")
    r.image('q2_order_comparison.png',250)
    r.p("Threshold 1.5 is about 2.12 standard deviations of the line's orthogonal noise (vertical noise σ = 1, slope −1) and 2.4 standard deviations of the circle's radial noise (σ = 0.625). Minimum consensus sizes below the expected 50 allow noise and overlap losses. These thresholds are design choices, not estimates of ground-truth membership.")
    r.p("For an inlier fraction w and sample size s, confidence after K trials is 1−(1−wˢ)ᴷ. With w = 0.5 and confidence 0.999, about 25 line trials or 52 circle trials suffice for an all-inlier sample; 5,000 is a conservative fixed budget for contaminated consensus scoring. Finding such a sample does not guarantee the correct model wins.")
    r.page('5  Two space tourism composites')
    q3_method(r); r.image('q3_pairs.png',340); q3_discussion(r); sources(r)
    r.page('6  Homography construction and blending')
    r.heading('Correspondences and coordinate convention')
    r.p("Coordinates below refer to the downloaded 1800-pixel-wide photographs. The saved destination coordinates are ordered consistently around each quadrilateral. Source corners use pixel centres (0,0), (w−1,0), (w−1,h−1), (0,h−1). H has eight independent degrees of freedom; four non-collinear correspondences determine it up to scale.")
    for i,pair in enumerate(M['q3']['pairs']):
        r.heading(f"Pair {i+1}")
        r.p('Destination corners: '+html.escape(str(pair['destination_points'])),8.5)
        r.code("\n".join(" ".join(f"{v: .7f}" for v in row) for row in pair['homography']))
    r.code("src = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])\nH = cv.getPerspectiveTransform(src, dst)\nwarped = cv.warpPerspective(art, H, (background_width, background_height))\nmask = cv.warpPerspective(np.ones((h,w), np.float32), H, size)[...,None]\nalpha = .94\nresult = np.uint8(np.clip(base*(1-alpha*mask)+warped*alpha*mask, 0, 255))")
    r.heading('Interpreting the result')
    r.p("The mask confines the blend to the quadrilateral, so pixels outside the display retain their original values. Bilinear resampling handles fractional projected locations. The Grand Tour is not cropped: adding side margins before warping prevents its lettering and planets from being stretched across a wide panel.")
    r.p("The photos were selected for unobstructed display boundaries and a coherent advertising theme. Their nearly frontal views provide weaker perspective variation than the Wadham architecture example. The wall display's fabric is not perfectly planar, which limits the geometric model. All source images remain in my_images/ for inspection.")
    sources(r)
    r.page('7  Registering PCB images with SIFT')
    q4_method(r); r.table(pcb_rows(),widths=[72,100,85,130,136])
    r.image('q4_registration.png',300)
    r.code("sift = cv.SIFT_create(nfeatures=12000)\nkeypoints, descriptors = sift.detectAndCompute(gray, None)\nmatcher = cv.BFMatcher(cv.NORM_L2)\ngood = [a for a,b in matcher.knnMatch(descriptors, template_desc, k=2)\n        if a.distance < .75*b.distance]\nH, mask = cv.findHomography(input_points, template_points, cv.RANSAC, 3.0)")
    r.p("Rows show each non-template input's inlier correspondences (up to 60 displayed) and its warped image. All accepted matches, not just the displayed subset, are used to estimate H. Median reprojection error is computed on the RANSAC inliers in template pixels. Matching and RANSAC seeds are fixed for repeatability.")
    r.page('8  PCB differences and interpretation')
    r.image('q4_differences.png',280)
    r.p("The first row shows raw per-channel absolute differences after alignment to AECC1; the second shows the same differences multiplied by 4 for visibility (clipped to [0,255]). Unsupported border pixels are blacked out. The template compared with itself gives exactly zero and is saved separately; it is omitted from the five non-template comparison panels.")
    q4_discussion(r)
    r.code("aligned = cv.warpPerspective(image, H, template_size)\nsupport = cv.warpPerspective(np.ones(image.shape[:2], np.uint8),\n                             H, template_size, flags=cv.INTER_NEAREST)\nsupport = cv.erode(support, np.ones((7,7), np.uint8)).astype(bool)\ndiff = cv.absdiff(aligned, template)\ndiff[~support] = 0\nmae = diff[support].mean()")
    r.heading('Reproducibility and supplied artifacts')
    r.p("Run assignment.py to regenerate figures, per-image outputs and metrics.json; then run build_report.py to rebuild both reports. The notebook organizes the same functions by question. Python and package versions are recorded in results/validation.json. All measurements in these reports are read from the saved results, rather than transcribed manually.")
    r.p("The detailed report expands the analysis at the student's request. The companion 230436X_a02.pdf is constrained to four pages to match the assignment submission rule. Git history has not been fabricated: the instructor's requirement for regular development commits must be satisfied through genuine work and review over time.")
    r.finish()

if __name__=='__main__':
    build_compact(); build_detailed()
    print('Built 230436X_a02.pdf (4 pages) and 230436X_a02_detailed.pdf (8 pages)')

