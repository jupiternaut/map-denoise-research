"""Seal the completed local experiment package without rewriting prior artifacts."""
import common as c
import json


def main():
    c.check_host()
    folders=[c.OUT/'evaluation',c.OUT/'posthoc_field_capacity',c.ROOT/'figures',c.ROOT/'figures_capacity']
    for folder in folders:c.verify_seal(folder)
    record=dict(host='liekkas',root=str(c.ROOT),artifacts=str(c.OUT),
                experimental_role='exposed replay plus explicitly posthoc capacity diagnosis',files={})
    for root in (c.ROOT,c.OUT):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path!=c.ROOT/'MANIFEST.json':
                record['files'][str(path)]=c.sha(path)
    c.save_json(c.ROOT/'MANIFEST.json',record)
    print('FINAL MANIFEST',len(record['files']),'files')

if __name__=='__main__':main()
