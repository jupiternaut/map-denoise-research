import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_book as b


class BookContracts(unittest.TestCase):
    def test_source_change_cannot_silently_rebind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'book';root.mkdir()
            source=Path(tmp)/'REPORT.md';source.write_text('first evidence')
            with patch.object(b,'ROOT',root),patch.object(b,'EXTRA',[]):
                first,_=b.snapshot_sources([{'sources':[str(source)]}])
                before=(root/'sources/MANIFEST.json').read_bytes()
                source.write_text('changed evidence')
                with self.assertRaisesRegex(RuntimeError,'Source drift'):
                    b.snapshot_sources([{'sources':[str(source)]}])
                self.assertEqual(before,(root/'sources/MANIFEST.json').read_bytes())
                self.assertEqual((root/first[0]['snapshot']).read_text(),'first evidence')

    def test_source_ids_stable_when_earlier_path_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'book';root.mkdir()
            old=Path(tmp)/'z.md';old.write_text('old')
            new=Path(tmp)/'a.md';new.write_text('new')
            with patch.object(b,'ROOT',root),patch.object(b,'EXTRA',[]):
                first,_=b.snapshot_sources([{'sources':[str(old)]}])
                second,_=b.snapshot_sources([{'sources':[str(old),str(new)]}])
                self.assertEqual(first[0]['id'],next(x['id'] for x in second if x['origin']==str(old)))

    def test_same_content_not_independent_source_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'book';root.mkdir()
            sources=[Path(tmp)/name for name in ('a.md','b.md')]
            for p in sources:p.write_text('identical report')
            with patch.object(b,'ROOT',root),patch.object(b,'EXTRA',[]):
                manifest,_=b.snapshot_sources([{'sources':list(map(str,sources))}])
                self.assertEqual(manifest[0]['snapshot'],manifest[1]['snapshot'])
                self.assertNotEqual(manifest[0]['id'],manifest[1]['id'])

    def test_generated_index_not_treated_as_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'records').mkdir()
            (root/'records/INDEX.json').write_text('{"generated":true}')
            (root/'records/empty.json').write_text('[]')
            with patch.object(b,'ROOT',root):self.assertEqual(b.read_records(),[])

    def test_read_records_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'records').mkdir()
            row={key:'' for key in b.FIELDS}
            for key in ('did','not_done','result','interpretation','decision','sources','tags','predecessors','corrections'):row[key]=[]
            row.update(id='DUP',evidence_level='report_read')
            (root/'records/a.json').write_text(json.dumps([row,row]))
            with patch.object(b,'ROOT',root),self.assertRaisesRegex(AssertionError,'duplicate IDs'):b.read_records()


if __name__=='__main__':unittest.main()
