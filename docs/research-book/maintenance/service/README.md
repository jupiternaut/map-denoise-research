# Research book LAN service

Exact host: liekkas. Book: `/home/grf/Documents/Codex/2026-09-29/geometry-meta-research-gitbook`.

URL: `http://192.168.6.201:8765/`.

This service serves the existing sealed book without editing it. The root opens READ.html; only files in its package manifest can be read. Directory listing, parent traversal and writes are unavailable. It binds only the LAN IPv4 address, accepts the 192.168.6.0/24 subnet, and does not create a public tunnel or open router/firewall ports. Devices on that LAN can read the book without a password.

The HTTP READ.html view is rendered from those same Markdown sources into this service directory by `render.py`, using the installed Python Markdown and BeautifulSoup libraries. It replaces the offline reader's raw-text fallback with formatted tables, links and available images. RENDER_RECEIPT.json identifies the source manifest and rendered bytes. The sealed source directory and ZIP stay unchanged.

The transient user service survives the terminal/session that started it, but is not installed for automatic startup after reboot.

```bash
systemctl --user status research-book-lan.service
journalctl --user -u research-book-lan.service -n 30
systemctl --user stop research-book-lan.service
```

Start (only when the unit is absent):

```bash
systemd-run --user --unit=research-book-lan --collect --description='Research book LAN reader' --property=Restart=on-failure --property=RestartSec=3 --property=NoNewPrivileges=yes /usr/bin/python3 -u /home/grf/Documents/Codex/2026-09-29/research-book-lan-service/serve.py --root /home/grf/Documents/Codex/2026-09-29/geometry-meta-research-gitbook --bind 192.168.6.201 --port 8765
```

The historical book's statement that it had not been published describes its sealed build-time state. This service adds LAN access only, not GitHub/GitBook cloud publication.

## September 30 increment

The same root reader now includes an append-only overlay from
`/home/grf/Documents/Codex/2026-09-30/geometry-closeout-20260930/memory/published`.
Paths under `/increment/` are separately allowlisted by that overlay's manifest.
The original package manifest, book and ZIP remain unchanged.

Refresh the complete reader through the memory publisher (which first renders
the base and then appends all frozen increments):

```bash
/usr/bin/python3 -B /home/grf/Documents/Codex/2026-09-30/geometry-closeout-20260930/memory/memory.py publish
/usr/bin/python3 -B /home/grf/Documents/Codex/2026-09-30/geometry-closeout-20260930/memory/memory.py verify
```

`render.py` by itself renders only the sealed base. The memory README documents
explicit byte snapshots, complete experiment registration and duplicate-research
search; refreshing the reader does not automatically acquire more conversation.
