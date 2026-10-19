"""Execute every notebook cell using this Python environment and save outputs."""
from pathlib import Path
import json, os, sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from build_notebook import build_notebook

ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.jupyter/mplconfig')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter/runtime')
for folder in ['.jupyter/mplconfig','.jupyter/runtime','.jupyter/kernels/en3160-a02']:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
spec_dir=ROOT/'.jupyter/kernels'
(spec_dir/'en3160-a02/kernel.json').write_text(json.dumps({
    'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'EN3160 A02 (Python 3)','language':'python'
},indent=2))
path=build_notebook()
nb=nbformat.read(path,as_version=4)
manager=KernelManager(kernel_name='en3160-a02',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_dir)]))
def on_cell_start(cell,cell_index,**kwargs):
    if cell.cell_type=='code':
        print(f'Executing cell {cell_index+1}/{len(nb.cells)}...',flush=True)
client=NotebookClient(nb,km=manager,timeout=600,allow_errors=False,
                      resources={'metadata':{'path':str(ROOT)}},on_cell_start=on_cell_start)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
code=[c for c in nb.cells if c.cell_type=='code']
errors=[o for c in code for o in c.outputs if o.output_type=='error']
images=sum(o.output_type in ('display_data','execute_result') and 'image/png' in o.get('data',{}) for c in code for o in c.outputs)
assert all(c.execution_count is not None for c in code)
assert not errors
assert images>=7
record={'executed_code_cells':len(code),'execution_counts':[c.execution_count for c in code],
        'embedded_png_outputs':images,'error_outputs':len(errors),'all_cells_executed':True,
        'python_executable':sys.executable,'notebook':path.name}
(ROOT/'results/notebook_execution.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)

