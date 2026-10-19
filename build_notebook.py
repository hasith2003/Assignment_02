"""Build the assignment notebook; execution is handled by execute_notebook.py."""
from pathlib import Path
import ast, json
import nbformat as nbf

ROOT=Path(__file__).resolve().parent

def build_notebook():
    source=(ROOT/'assignment.py').read_text()
    tree=ast.parse(source)
    functions={node.name:ast.get_source_segment(source,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('# EN3160 Assignment 2: Fitting and Alignment\n**NAHG Nettikumara · 230436X**\n\n[GitHub profile](https://github.com/hasith2003) | [Repository](https://github.com/hasith2003/Assignment_02)\n\nThis notebook performs all four experiments from the supplied images. Run all cells in order from the assignment folder. Each result figure is generated and embedded in its code-cell output. Fixed seeds make the synthetic experiment repeatable.')
    header=source[:source.index('def savefig(')].replace("ROOT = Path(__file__).resolve().parent","ROOT = Path.cwd()")
    code(header+"\nfrom IPython.display import display, Image, Markdown\nassert DATA.exists(), 'Run this notebook from the assignment folder.'\nprint('OpenCV:', cv.__version__, '| NumPy:', np.__version__)")
    code("""def savefig(name):
    plt.tight_layout()
    path = OUT / name
    plt.savefig(path, dpi=170, bbox_inches='tight')
    plt.close()
    display(Image(filename=str(path), width=950))
""")
    md('## 1. Sunflower blob detection using LoG\nThe image is read at quarter resolution as required. A positive, scale-normalized Laplacian detects dark central disks; a yellow annulus rejects many foliage blobs. Extrema are selected jointly across position and scale, then nearby candidates are suppressed. No Hough circle detector is used.\n\nThe reported radius is the **dark central disk radius**, not the outer petal radius. The search uses 35 geometric scales from σ = 1.5 to 18 reduced-image pixels.')
    code(functions['blobs'])
    code("""q1 = blobs()
print(f"Retained candidate centres: {q1['detections']}")
print(f"Sigma search range: {q1['sigma_range']} (reduced-image pixels)")
print("Five largest detected central disks:")
print("rank      x      y      sigma      radius")
for rank, c in enumerate(q1['largest'], 1):
    print(f"{rank:4d} {c['x']:6d} {c['y']:6d} {c['sigma']:10.3f} {c['radius']:11.3f}")
""")
    md('### Interpretation\nThe largest foreground flower disks are recovered at larger scales than distant centres. Parameters refer to the 360 × 360 image; multiply coordinates and radii approximately by four for the original image. Irregular shape, occlusion and dark leaves surrounded by petals can still cause errors. There is no supplied annotation set, so the candidate count is not a validated count of every flower.')
    md('## 2. Custom RANSAC: line first, then circle\nThe supplied generation code is reproduced with seed 0. A line is represented by **ax + by = d**, with a² + b² = 1 and d ≥ 0. Its error is orthogonal distance. A circle is represented by its centre and positive radius, and uses radial error.\n\nTwo-point line samples and three-point circle samples are drawn by our own RANSAC implementation. Each search tries 5,000 hypotheses. The inlier threshold is 1.5 units; minimum consensuses are 30 for the line and 25 for the remaining circle. The winning sample consensus is kept fixed during refinement.')
    md('### Minimal models and geometric errors')
    for name in ['line_model','circle_model','residual']:code(functions[name])
    md('### RANSAC and consensus refinement\nHypotheses are ranked by consensus size; truncated squared residual breaks ties. TLS refines the line under the unit-normal constraint. Nonlinear least squares minimizes squared radial residuals for the circle. Degenerate samples are rejected.')
    code(functions['ransac'])
    md('### Generate the supplied noisy points and show all requested elements\nThe plot includes all points, both consensuses, the winning line sample, the winning three-point circle sample, the sample models, the refined models and the known ground truth.')
    code(functions['fitting'])
    code("""q2 = fitting()
print('Line refined [a, b, d]:', np.round(q2['line_refined'], 6))
print('Normal length:', np.linalg.norm(q2['line_refined'][:2]))
print('Line consensus:', q2['line_consensus'])
print('Circle refined [cx, cy, r]:', np.round(q2['circle_refined'], 6))
print('Circle consensus after line removal:', q2['circle_consensus'])
print('Ground truth: line y = -x + 2; circle centre (2, 3), radius 10.')
""")
    md('### 2(d). What happens if the circle is fit first?\nA circle fit can absorb line points near the intersections. A sufficiently large circle can also approximate a line along a short arc. Failure is possible, but not inevitable. The next cell actually runs circle-first RANSAC and highlights the true line points it would remove; ground-truth labels are used only for evaluation.')
    code("""points = np.load(OUT/'q2_data.npz')['points']
_, first_circle, first_mask, _ = ransac(points, 'circle', seed=20, min_consensus=25)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, title, fit in [
    (axes[0], 'Line first, then circle', q2['circle_refined']),
    (axes[1], 'Circle first', first_circle),
]:
    ax.scatter(*points[:50].T, s=15, c='tab:orange', label='True circle points')
    ax.scatter(*points[50:].T, s=15, c='tab:blue', label='True line points')
    ax.add_patch(plt.Circle((2, 3), 10, fill=False, color='green', ls=':', label='True circle'))
    ax.add_patch(plt.Circle(fit[:2], fit[2], fill=False, color='red', label='Fitted circle'))
    ax.set(xlim=(-15, 17), ylim=(-14, 18), aspect='equal', title=title, xlabel='x', ylabel='y')
removed_line = first_mask & (np.arange(len(points)) >= 50)
axes[1].scatter(*points[removed_line].T, facecolors='none', edgecolors='black',
                s=80, label='True line points removed')
for ax in axes:
    ax.legend(fontsize=7)
savefig('q2_order_comparison.png')
print('Circle-first [cx, cy, r]:', np.round(first_circle, 6))
print('Circle-first consensus:', int(first_mask.sum()))
print('True line points removed:', int(removed_line.sum()))
""")
    md('### Interpretation and threshold choice\nFor slope −1 and vertical noise σ = 1, orthogonal line noise has standard deviation 1/√2. The threshold 1.5 is about 2.12 such standard deviations, and about 2.4 circle radial-noise standard deviations (σ = 0.625). Removing a line consensus also removes some genuine circle points near intersections. In this seed, fitting the circle first still finds a plausible circle but removes 12 genuine line points.')
    md('## 3. Planar homography and alpha blending\nTwo selected web image pairs advertise imaginary space tourism: a NASA/JPL Mars poster in a street display and The Grand Tour in a wall display. Four destination corners are selected manually in top-left, top-right, bottom-right, bottom-left order. H maps the artwork to that quadrilateral; a warped mask limits the blend. The second artwork receives side margins to preserve its portrait proportions.\n\n[Source links and credits](my_images/SOURCES.md). The optional pick_corners.py utility supports clicking alternative corners and saving them to my_images/corners.json. The supplied Wadham/flag example is also generated.')
    code(functions['composite'])
    code("""q3 = composite()
for i, pair in enumerate(q3['pairs'], 1):
    print(f"Pair {i}: {pair['overlay']} -> {pair['background']}")
    print('Destination corners:', np.asarray(pair['destination_points']))
    print('Homography:')
    print(np.array2string(np.array(pair['homography']), precision=7))
    print()
display(Markdown('**Supplied Wadham/flag example:**'))
display(Image(filename=str(OUT/'q3_wadham.png'), width=800))
""")
    md('### Interpretation\nBoth displays are nearly frontal, so the projective distortion is modest. A single homography assumes one plane and cannot reproduce fabric wrinkles, curved surfaces or foreground occluders. Blending at alpha 0.94 retains a little background illumination, but does not physically reproduce shadows or reflections.')
    md('## 4. PCB alignment and image differences\nAECC1.jpg is the template. All five other PCB images are processed using SIFT (up to 12,000 features), BFMatcher/L2 and the nearest-neighbour ratio test (0.75). Homography RANSAC uses a 3-pixel threshold, up to 20,000 trials and confidence 0.999.\n\nRaw per-channel absolute differences are saved losslessly as PNG. A warped support mask, eroded by three pixels, excludes unsupported borders. The overview shows matches, aligned images and raw differences for every comparison.')
    code(functions['pcb'])
    code("""q4 = pcb()
print("Image        Matches   H-inliers   Median reproj(px)    Overlap MAE")
for row in q4:
    print(f"{row['image']:12s} {row['ratio_matches']:7d} {row['ransac_inliers']:11d}"
          f" {row['median_reprojection_px']:19.3f} {row['overlap_mae']:14.3f}")
""")
    md('### Raw differences and enhanced views for all PCB images\nThe first row is the actual absolute difference. The second uses **4× display gain**, clipped to [0,255], to make small residuals visible. Gain is not used to compute MAE. The template self-difference is checked separately.')
    code("""fig, axes = plt.subplots(2, len(q4), figsize=(13, 6.8))
for col, row in enumerate(q4):
    stem = Path(row['image']).stem
    raw = cv.imread(str(OUT/f'q4_{stem}_difference.png'))
    for plot_row, gain in enumerate([1, 4]):
        shown = np.clip(raw.astype(float)*gain, 0, 255).astype(np.uint8)
        axes[plot_row, col].imshow(cv.cvtColor(shown, cv.COLOR_BGR2RGB))
        axes[plot_row, col].set_title(stem + (' raw' if gain == 1 else ' (4x gain)'), fontsize=10)
        axes[plot_row, col].axis('off')
savefig('q4_differences.png')
self_difference = cv.imread(str(OUT/'q4_AECC1_difference.png'))
assert not self_difference.any()
print('AECC1 template self-difference: exactly zero.')
""")
    md('### 4(e). Why do the USB port, capacitors and reset button differ strongly?\nThese components protrude above the board. Viewpoint changes create parallax that one homography fitted to the board plane cannot remove. Metallic reflections, shadows, illumination changes and occlusion add photometric residuals. Actual component or layout changes may also contribute. A difference map alone cannot identify a defect.\n\nAAAA1 has far fewer inliers than the AECC images, so its alignment deserves additional caution. Low reprojection error on matched features does not guarantee every raised component aligns.')
    md('## Final numerical checks and saved measurements\nThese checks cover geometry and output support, rather than claiming visual defect-detection accuracy.')
    code("""assert np.isclose(np.linalg.norm(q2['line_refined'][:2]), 1)
assert np.linalg.norm(np.array(q2['circle_refined'][:2]) - [2, 3]) < 1
assert abs(q2['circle_refined'][2] - 10) < 1
assert len(q4) == 5
assert all(row['median_reprojection_px'] < 3 for row in q4)
for i, pair in enumerate(q3['pairs']):
    art = cv.imread(str(ROOT/'my_images'/pair['overlay']))
    h, w = art.shape[:2]
    if i == 1:
        h, w = 900, 1480
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    projected = cv.perspectiveTransform(corners.reshape(-1, 1, 2),
                                       np.array(pair['homography'])).reshape(-1, 2)
    assert np.max(np.linalg.norm(projected-pair['destination_points'], axis=1)) < 1e-3
for row in q4:
    stem = Path(row['image']).stem
    diff = cv.imread(str(OUT/f'q4_{stem}_difference.png'))
    support = cv.imread(str(OUT/f'q4_{stem}_support.png'), cv.IMREAD_GRAYSCALE) > 0
    assert not diff[~support].any()
metrics = {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4}
(OUT/'metrics.json').write_text(json.dumps(metrics, indent=2))
print('All notebook numerical checks passed.')
print('Measured results saved to', OUT/'metrics.json')
""")
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'EN3160 A02 (Python 3)','language':'python','name':'en3160-a02'}})
    target=ROOT/'230436X_a02.ipynb'
    nbf.write(nb,target)
    print('Created',target,'with',sum(c.cell_type=='code' for c in cells),'code cells.')
    return target

if __name__=='__main__':
    build_notebook()

