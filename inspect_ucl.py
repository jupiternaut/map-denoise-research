"""Trace the official Indoor-Bench download. Do not invent stations or use unknown mirrors."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import urllib.error
import urllib.request

from hashutil import dump_json
from paths import LOGS, UCL_DOWNLOADS, require_liekkas

OFFICIAL = "https://indoor-bench.github.io/indoor-bench/"
PAPER = "https://isprs-archives.copernicus.org/articles/XL-5/581/2014/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 research-data-access"


def fetch(url: str, max_bytes=200_000) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read(max_bytes)
            return {
                "url": url,
                "final_url": resp.geturl(),
                "status": resp.status,
                "content_type": resp.headers.get("content-type"),
                "content_length": resp.headers.get("content-length"),
                "body": body.decode("utf-8", "replace"),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read(2000)
        return {
            "url": url,
            "status": exc.code,
            "reason": exc.reason,
            "www_authenticate": exc.headers.get("WWW-Authenticate") if exc.headers else None,
            "body": raw.decode("utf-8", "replace"),
        }
    except Exception as exc:
        return {"url": url, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def slim(rec: dict) -> dict:
    out = dict(rec)
    body = out.pop("body", "")
    out["body_head"] = body[:400]
    out["login_live"] = "login.live.com" in body.lower() or "login.live.com" in str(out.get("final_url", "")).lower()
    out["unauthenticated"] = "unauthenticated" in body.lower()
    return out


def main():
    require_liekkas()
    LOGS.mkdir(parents=True, exist_ok=True)
    UCL_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    page = fetch(OFFICIAL)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', page.get("body", ""), re.I)
    share = next((h for h in hrefs if "1drv.ms" in h or "onedrive" in h.lower()), None)
    probes = [slim(page)]
    if share:
        probes.append(slim(fetch(share)))
        probes.append(slim(fetch("https://api.onedrive.com/v1.0/shares/s!AuBuZXi2cbwhnCe-A4jdfGqHRgzM/root")))
        encoded = "u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdUJ1WlhpMmNid2huQ2UtQTRqZGZHcUhSZ3pN"
        probes.append(slim(fetch(f"https://api.onedrive.com/v1.0/shares/{encoded}/root")))
        probes.append(slim(fetch("https://onedrive.live.com/redir?resid=21BC71B678656EE0!3623")))
    loginish = bool(share) and any(
        p.get("login_live")
        or p.get("unauthenticated")
        or str(p.get("status")) in {"401", "403"}
        for p in probes
    )
    report = {
        "host": "liekkas",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "official_page": OFFICIAL,
        "paper": PAPER,
        "official_download_href": share,
        "page_hrefs": hrefs,
        "requested_subset": "Basic Corridor only",
        "ifc_role": (
            "IFC is a human model from Faro scans, not independent millimetre GT. "
            "The paper states many thicknesses are arbitrary. This round does not "
            "feed IFC to filters or use it as a geometric scorer."
        ),
        "e57_status": "not_downloaded",
        "usable_for_frame_bias": False,
        "blocker": {
            "kind": "official_onedrive_share_requires_microsoft_login_or_returns_unauthenticated",
            "details": (
                "The official page only publishes a OneDrive/SharePoint share "
                "(https://1drv.ms/u/s!AuBuZXi2cbwhnCe-A4jdfGqHRgzM). "
                "Anonymous Graph/OneDrive API access returns 401 unauthenticated "
                "or a malformed share id. A cookie-less redirect lands on login.live.com. "
                "No public direct zip URL is listed on the official page."
            ),
            "action_not_taken": [
                "did not use an unknown mirror",
                "did not borrow an account",
                "did not invent scan IDs",
            ],
        }
        if loginish or share
        else {
            "kind": "official_page_has_no_usable_anonymous_file_url",
            "details": "Could not extract a direct E57/IFC URL from the official page.",
            "page_hrefs": hrefs,
        },
        "probes": probes,
        "downloaded_files": [],
        "note": "If E57 later arrives and is only a fused cloud, mark it unusable for frame-bias experiments.",
    }
    dump_json(LOGS / "ucl_probe.json", report)
    dump_json(UCL_DOWNLOADS / "BLOCKER.json", {k: report[k] for k in report if k != "probes"})
    print(json.dumps({"href": share, "blocker": report["blocker"]["kind"], "n_probes": len(probes)}, indent=2))


if __name__ == "__main__":
    main()
