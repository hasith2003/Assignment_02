"""Inspect deliverable PDFs and notebook; render review sheets."""
from pathlib import Path
import json, ast
import pymupdf
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
records={}
for name,expected in [('230436X_a02.pdf',4),('230436X_a02_detailed.pdf',8)]:
    doc=pymupdf.open(ROOT/name)
    assert len(doc)==expected,(name,len(doc))
    pages=[]; lengths=[]
    for i,page in enumerate(doc):
        text=page.get_text()
        assert '230436X' in text
        assert len(text)>300
        assert '\ufffd' not in text
        for block in page.get_text('blocks'):
            assert block[1]>=12 and block[3]<=page.rect.height-12,(name,i,block[:4])
        pix=page.get_pixmap(matrix=pymupdf.Matrix(.72,.72))
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        pages.append(im); lengths.append(len(text))
    assert 'NAHG Nettikumara' in doc[0].get_text()
    sheet=Image.new('RGB',(pages[0].width*2,(pages[0].height+24)*((expected+1)//2)), '#dce3e8')
    draw=ImageDraw.Draw(sheet)
    for i,im in enumerate(pages):
        x=(i%2)*im.width; y=(i//2)*(im.height+24)
        draw.text((x+10,y+5),f'Page {i+1}',fill='black')
        sheet.paste(im,(x,y+24))
    target=OUT/(Path(name).stem+'_review.jpg');sheet.save(target,quality=90)
    records[name]={'pages':len(doc),'text_characters_per_page':lengths,'review_sheet':target.name}
notebook=json.loads((ROOT/'230436X_a02.ipynb').read_text(encoding='utf-8'))
count=0
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        compile(''.join(cell['source']),'<notebook-cell>','exec');count+=1
assert count>=9
tree=ast.parse((ROOT/'assignment.py').read_text())
assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='HoughCircles' for node in ast.walk(tree))
code_cells=[c for c in notebook['cells'] if c['cell_type']=='code']
execution_counts=[c['execution_count'] for c in code_cells]
assert execution_counts==list(range(1,count+1))
error_outputs=[o for c in code_cells for o in c['outputs'] if o.get('output_type')=='error']
assert not error_outputs
images=sum('image/png' in o.get('data',{}) for c in code_cells for o in c['outputs'])
assert images>=7
records['notebook']={'valid_json':True,'code_cells_compile':count,
                     'executed_code_cells':count,'execution_counts':execution_counts,
                     'error_outputs':len(error_outputs),'embedded_png_outputs':images}
records['assignment']={'no_hough_circles_call':True}
(OUT/'delivery_checks.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))

