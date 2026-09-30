#!/usr/bin/env python3
"""Read-only LAN endpoint for the already sealed research book."""
import argparse
import functools
import ipaddress
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


class BookHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        requested = unquote(urlsplit(path).path)
        if requested.startswith('/increment/'):
            return str(self.overlay / requested.removeprefix('/increment/'))
        if unquote(urlsplit(path).path) == '/READ.html':
            rendered = Path(__file__).with_name('reader.html')
            if rendered.is_file():
                return str(rendered)
        return super().translate_path(path)

    def __init__(self, *args, root, allowed, network, overlay, **kwargs):
        self.root = root
        self.allowed = allowed
        self.network = network
        self.overlay = overlay
        super().__init__(*args, directory=str(root), **kwargs)

    def permitted(self):
        if ipaddress.ip_address(self.client_address[0]) not in self.network:
            self.send_error(403, 'Local network access only')
            return False
        requested = unquote(urlsplit(self.path).path)
        if '\x00' in requested or '\\' in requested or '..' in PurePosixPath(requested).parts:
            self.send_error(403, 'Invalid book path')
            return False
        name = requested.lstrip('/') or 'READ.html'
        if name.startswith('increment/'):
            manifest_path = self.overlay / 'MANIFEST.json'
            if not manifest_path.is_file():
                self.send_error(404, 'Increment unavailable')
                return False
            manifest = json.loads(manifest_path.read_text())
            overlay_allowed = {entry['path'] for entry in manifest['files']} | {'MANIFEST.json'}
            rel = name.removeprefix('increment/')
            target = (self.overlay / rel).resolve()
            if rel not in overlay_allowed or not target.is_relative_to(self.overlay) or not target.is_file():
                self.send_error(404, 'Not a published increment file')
                return False
            return True
        if name not in self.allowed:
            self.send_error(404, 'Not a published book file')
            return False
        target = (self.root / name).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            self.send_error(404, 'Book file unavailable')
            return False
        if requested == '/':
            self.path = '/READ.html'
        return True

    def do_GET(self):
        if self.permitted():
            super().do_GET()

    def do_HEAD(self):
        if self.permitted():
            super().do_HEAD()

    def list_directory(self, path):
        self.send_error(403, 'Directory listing disabled')

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Security-Policy',
                         "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; "
                         "script-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        super().end_headers()

    def guess_type(self, path):
        if Path(path).suffix.lower() in ('.md', '.txt', '.py', '.yaml', '.jsonl'):
            return 'text/plain; charset=utf-8'
        return super().guess_type(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bind', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--network', default='192.168.6.0/24')
    parser.add_argument('--overlay', type=Path,
                        default=Path('/home/grf/Documents/Codex/2026-09-30/geometry-closeout-20260930/memory/published'))
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text())
    allowed = {entry['path'] for entry in manifest['files']} | {'PACKAGE_MANIFEST.json'}
    assert 'READ.html' in allowed
    handler = functools.partial(BookHandler, root=root, allowed=allowed,
                                network=ipaddress.ip_network(args.network), overlay=args.overlay.resolve())
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f'Research book: http://{args.bind}:{args.port}/ ; root={root}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
