"""Record TUM-TLS-24 metadata only. Do not download the ~95 GB archive."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import urllib.request

from hashutil import dump_json
from paths import TUM_META, require_liekkas

PAGE = "https://tum2t.win/datasets/pc-tls"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 research-data-access"


def main():
    require_liekkas()
    TUM_META.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(PAGE, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        html = resp.read(200_000).decode("utf-8", "replace")
        final = resp.geturl()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    links = {
        "subsampled": next((h for h in hrefs if "Subsampled" in h), None),
        "original": next((h for h in hrefs if "Original" in h), None),
        "complete_tum_tls_24": next((h for h in hrefs if "TUM-TLS-24" in h), None),
    }
    report = {
        "host": "liekkas",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "official_page": PAGE,
        "final_url": final,
        "published_links": links,
        "z_offset_note": "Official page: TLS clouds need +0.7551 m in Z to align with other TUM2TWIN models; to be fixed in a later release.",
        "subset_possible": (
            "The page separately lists subsampled clouds, full-resolution clouds, "
            "and the complete dataset with metadata. That is enough to say a later "
            "round can try a small station subset instead of the ~95 GB bundle. "
            "This round does not download any of those archives."
        ),
        "downloaded": False,
        "reason_not_downloaded": "task forbids downloading TUM-TLS-24 (~95 GB) this round",
        "later_checks_required": [
            "re-check the published version",
            "station list and poses",
            "cross-model coordinate offset",
            "licence",
            "do not treat registration residual as independent absolute geometry",
        ],
    }
    dump_json(TUM_META / "TUM_TLS24_METADATA.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
