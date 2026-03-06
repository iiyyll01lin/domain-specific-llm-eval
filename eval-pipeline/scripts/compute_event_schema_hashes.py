#!/usr/bin/env python3
import json, hashlib, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
registry_path = ROOT / 'events' / 'schema_registry.json'

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    data = json.loads(registry_path.read_text())
    changed = False
    for ev in data.get('events', []):
        schema_file = ROOT / ev['schema_file']
        digest = sha256_file(schema_file)
        if ev.get('sha256') != digest:
            ev['sha256'] = digest
            changed = True
    if changed:
        registry_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        print('Registry updated with new hashes.')
    else:
        print('No changes.')

if __name__ == '__main__':
    sys.exit(main())
