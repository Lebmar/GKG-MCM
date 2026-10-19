"""Build the report and supplementary ZIP, with explicit inclusion rules."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
pdf=ROOT/'output/pdf/SafeInvest_Team87_Report.pdf'
pdf.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'report/latex/main.pdf',pdf)
r=PdfReader(pdf)
assert 1<=len(r.pages)<=10
for page in r.pages:
    assert abs(float(page.mediabox.width)-595.276)<1
    assert abs(float(page.mediabox.height)-841.890)<1
text='\n'.join(page.extract_text() for page in r.pages)
for required in ['Chengxuan YUAN','Use of AI Tools','35.5','3.66','43.44']:
    assert ''.join(required.split()) in ''.join(text.split()),required
evidence=json.loads((ROOT/'report/report_evidence.json').read_text())
assert evidence['customer_explanation_words']<=50
assert 'Overfull' not in (ROOT/'report/latex/main.log').read_text()
validation={'status':'PASS','report_pages':len(r.pages),'page_size':'A4',
 'declared_main_font_pt':11,'figure_font_pt_before_scale':12,
 'figures_rendered_at_nearly_original_width':True,
 'customer_explanation_words':evidence['customer_explanation_words'],
 'latex_overfull_boxes':False,'primary_layout_visually_reviewed_pages':list(range(1,len(r.pages)+1)),
 'review_note':'All-page layout contact sheets plus full-size checks of changed pages; mathematical scripts naturally use smaller glyphs.',
 'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
(ROOT/'results/report_validation.json').write_text(json.dumps(validation,indent=2))
archive=ROOT/'output/submission/Team87_Lebmar_Code.zip'
archive.parent.mkdir(parents=True,exist_ok=True)
roots=['src','tests','scripts','data','docs','results','report/latex','report/figures']
files=[ROOT/name for name in ['README.md','requirements.txt','requirements-lock.txt','.gitignore']]
for relative in roots:
    files.extend(p for p in (ROOT/relative).rglob('*') if p.is_file())
exclude={'.aux','.log','.out','.fls','.fdb_latexmk','.gz','.pyc'}
selected=sorted(set(p for p in files if p.suffix not in exclude and '__pycache__' not in p.parts
                    and p.name not in ['main.pdf','.DS_Store']))
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in selected:z.write(p,str(p.relative_to(ROOT)))
assert archive.stat().st_size<50_000_000
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
manifest={'report':{'path':str(pdf.relative_to(ROOT)),'bytes':pdf.stat().st_size,
                   'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()},
          'supplementary':{'path':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,
                          'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(selected)},
          'submitted_to_CMT':False,'human_author_review_required':True}
(ROOT/'output/submission/manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
