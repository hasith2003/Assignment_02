# EN3160 Assignment 2: Fitting and Alignment

**NAHG Nettikumara | 230436X**

[GitHub profile](https://github.com/hasith2003)

## Notebook

Open [230436X_a02.ipynb](230436X_a02.ipynb) to view the implementation,
executed results, figures and discussion for all four questions:

1. Sunflower central-disk detection with scale-normalized LoG.
2. Custom RANSAC line and circle fitting, including a circle-first comparison.
3. Two homography-based image composites.
4. SIFT-based PCB alignment and difference images.

The notebook contains all experiment code. Its 17 code cells have saved execution
counts and eight embedded figure outputs.

## Run

Use Python 3.12 with Jupyter Notebook, JupyterLab or VS Code.

1. Install the packages in requirements.txt in your Python environment.
2. Open the notebook from the repository root and select that environment.
3. Restart the kernel and run all cells in order.

Keep the a2images/a2images and my_images folders in their existing locations.
Input paths are relative to the notebook working directory. Results are saved
under results/.

## Reports and results

- [PDF exported directly from the executed notebook](230436X_a02.pdf)
- [Measured results](results/metrics.json)
- [Image sources and credits](my_images/SOURCES.md)

The PDF is a direct 14-page conversion of the notebook, including its code,
explanations and all eight saved figures. It was exported with nbconvert to
self-contained HTML and printed to A4 PDF, with code wrapping to prevent clipping.
Export provenance and checks are recorded in results/pdf_export.json.

The sunflower circles describe dark central disks, not outer petals. Candidate
detections are not a validated flower count. PCB differences can include parallax,
lighting and reflections; AAAA1 has relatively few reliable homography inliers.

