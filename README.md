# EN3160 Assignment 2 — Fitting and Alignment

NAHG Nettikumara · 230436X
GitHub profile: https://github.com/hasith2003
Repository: https://github.com/hasith2003/Assignment_02

## Deliverables

- 230436X_a02_detailed.pdf — eight-page detailed report requested by the student.
- 230436X_a02.pdf — four-page submission report matching the assignment limit.
- 230436X_a02.ipynb — executed notebook with code-cell results and embedded figures.
- assignment.py — complete, deterministic experiment implementation.
- results/metrics.json — measured parameters, consensus sizes and alignment statistics.
- results/validation.json — numerical verification and exact package versions.
- results/q4_*_difference.png — lossless differences for all five non-template PCBs,
  plus the zero self-difference for AECC1.
- my_images/SOURCES.md — image sources, photographers and NASA/JPL credits.

The brief specifies a two-mark penalty per page beyond four. Use the four-page
version if submitting under that rule. The detailed report explains the results
more fully and includes scale-response plots, model ordering and raw/enhanced
difference comparisons.

## Run on this computer

The compatible Python environment is on D: because C: was almost full.
In PowerShell, from D:\Ard\CV:

    & D:\en3160_a02_env\venv\Scripts\python.exe assignment.py
    & D:\en3160_a02_env\venv\Scripts\python.exe prepare_deliverables.py
    & D:\en3160_a02_env\venv\Scripts\python.exe build_report.py

The first script performs all experiments. The second generates supplementary
figures and runs numerical checks. The third creates PDFs.
For the notebook, select D:\en3160_a02_env\venv\Scripts\python.exe as your Python
environment in an editor that supports notebooks. The notebook is executed from a fresh Jupyter kernel and saved with embedded
outputs. To rebuild and execute it again, run:

    & D:\en3160_a02_env\venv\Scripts\python.exe execute_notebook.py

Execution status is saved to results/notebook_execution.json.
prepare_deliverables.py preserves the executed notebook.

## Run on another computer

Use Python 3.12 and install requirements.txt in your own virtual environment.
Then run the same three scripts with that environment's Python executable.
PyMuPDF is used by verify_delivery.py to inspect and render the PDFs.

Input paths are relative to the scripts and do not depend on D:.
Keep the supplied a2images/a2images tree and the four files in my_images/.

## Methods and practical limits

Q1: positive scale-normalized LoG extrema in grayscale detect dark central flower
disks. A yellow annulus rejects many foliage blobs. Radius sqrt(2)*sigma describes
the central disk, not the outer petal boundary. There is no Hough circle detector.
The 60 detections are candidates, not an evaluated count of every flower.

Q2: RANSAC is implemented from minimal samples. Both residuals are geometric.
Line refinement uses TLS; circle refinement minimizes radial residuals. Consensus
sets shown are the winning sample's sets, fixed during refinement. Ground-truth
labels are used only for evaluation, never to choose the models.

Q3: two NASA/JPL poster/photo pairs are warped with manually selected corner
coordinates and alpha-blended. The Grand Tour receives side margins to preserve
its portrait aspect ratio on a landscape panel. The optional pick_corners.py
utility lets you click new corners and save them in my_images/corners.json.
Both photographs are nearly frontal; the fabric is only approximately planar.

Q4: SIFT/BFMatcher, ratio 0.75, homography RANSAC (3 pixels, 20,000 iterations,
confidence 0.999). Raw differences are saved as PNGs. Unsupported warped borders
are excluded. Fourfold gain in figures is only a display adjustment, not a change
to MAE. AAAA1 has just 14 homography inliers and deserves extra caution. High
differences can reflect parallax, shadows and illumination as well as real changes.
No defect labels or measured defect-detection accuracy are claimed.

## Development history

The instructor requests genuine regular commits over a reasonable duration.
No dates or development history have been fabricated. Review the work, understand
the methods and make real commits as you develop or revise it. Adding a GitHub
link to a PDF does not itself upload files to GitHub.

