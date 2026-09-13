"""Verify data provenance and delivered PDF structure, without rerunning geometry."""
import csv
import hashlib
import json
from pathlib import Path
import re
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDFS = ['fig01_pipeline.pdf','fig02_confirmation.pdf','fig03_table.pdf','fig04_code.pdf','map-denoise-visual-atlas.pdf']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    prov=json.loads((HERE/'PROVENANCE.json').read_text(encoding='utf-8'))
    for source in prov['source_files']:
        assert sha(ROOT/source['path'])==source['sha256'], source['path']
    excerpt=''.join((ROOT/'reconstruction_v22/operator.py').read_text(encoding='utf-8').splitlines(keepends=True)[118:138])
    assert excerpt==(HERE/'data/operator_excerpt.py').read_text(encoding='utf-8')
    run=ROOT/'evidence/progress_v22/runs/reconstruction-v22-qayc8gft'
    summary=json.loads((run/'SUMMARY.json').read_text(encoding='utf-8'))['results']['confirmation']
    comparison=json.loads((HERE/'data/identity_comparison.json').read_text(encoding='utf-8'))
    assert comparison=={'identity':summary['comparisons']['identity']}
    means=json.loads((HERE/'data/table_means_verified.json').read_text(encoding='utf-8'))
    independent=list(csv.DictReader((HERE/'data/table_summary.csv').open(encoding='utf-8-sig')))
    assert len(independent)==9
    for row in independent:
        m=means[row['method']]
        for a,b,factor in [('mae_mm','accuracy_mm',1),('recall_pct','recall',100),('fscore','fscore',1),('displacement_rms_mm','displacement_rms_mm',1)]:
            assert abs(float(row[a])-m[b]*factor)<1e-12,(row['method'],a)
    results=[]
    for name in PDFS:
        path=HERE/name
        with fitz.open(path) as doc:
            expected=4 if 'atlas' in name else 1
            assert len(doc)==expected,(name,len(doc))
            item={'file':name,'pages':len(doc),'sha256':sha(path),'page_checks':[]}
            for p in doc:
                assert len(p.get_images())==0,(name,'raster image found')
                assert len(p.get_text())>100
                assert '\ufffd' not in p.get_text(),(name,'replacement character')
                outside=[]
                for b in p.get_text('dict')['blocks']:
                    for line in b.get('lines',[]):
                        for span in line['spans']:
                            box=fitz.Rect(span['bbox'])
                            if not (p.rect+(-1,-1,1,1)).contains(box):outside.append(span['text'])
                assert not outside,(name,'out-of-page text',outside[:3])
                item['page_checks'].append({'size_pt':[p.rect.width,p.rect.height],
                    'embedded_raster_images':0,'vector_drawing_groups':len(p.get_drawings()),
                    'text_characters':len(p.get_text()),'out_of_page_text':0})
            results.append(item)
    for name in ['fig01_pipeline','fig03_table','fig04_code']:
        log=(HERE/f'{name}.log').read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'Overfull|Missing character|^! ',log,re.M),name
    report={'status':'PASS','provenance_sources':len(prov['source_files']),
        'independent_numeric_crosschecks':36,'code_excerpt_exact':True,'published_json_exact':True,
        'scope':'Saved-score reaggregation, source correspondence and PDF layout/structure. No new geometric experiment.',
        'rendered_files':results,
        'compiler_notes':'Portable Tectonic uses installed Windows fonts. First build downloaded TeX packages; fontconfig/lineno package diagnostics retained in logs. No missing-glyph or overfull errors.'}
    (HERE/'VALIDATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json'] and '__pycache__' not in p.parts]
    (HERE/'MANIFEST.json').write_text(json.dumps({'files':[{'path':p.relative_to(HERE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]},indent=2),encoding='utf-8')
    print(f'PASS: 6 frozen inputs, 36 independent numerical crosschecks, exact source/JSON, {len(PDFS)} vector PDFs / 8 pages.')

if __name__=='__main__':main()
