"""Per-FILE label operations on the ga-fdp channel.

    python channel_tools.py retire  <package> <version-being-staged> <archive-dir>
    python channel_tools.py promote <package> <version>

Spec: docs/specs/2026-09-15-staging-and-promotion.md sections 4 and 5.

WHY THIS EXISTS INSTEAD OF `anaconda move`
------------------------------------------
`anaconda move --from-label staging --to-label main ga-fdp/pkg/X.Y.Z` does
not move X.Y.Z. It moves **everything in the source label**. Demonstrated
against the live API: two versions in one label, "move" one, both moved.

The CLI is not at fault -- it passes `version` through. The server ignores
it. Granularity, established empirically:

    {package}                        -> every file in the package
    {package, version}               -> version IGNORED, still every file
    {package, version, basename}     -> exactly that file

So every call here passes a basename, and a caller that wants a version
enumerates its files first.

Two more behaviours that shaped this file:

  * `anaconda move` EXITS 0 ON FAILURE. Its implementation catches the
    exception and calls logger.exception(). So nothing here trusts a return
    code: every mutation is verified by reading the labels back.
  * Labels are lowercased server-side. `Attic` and `attic` are one label.
"""

import hashlib
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

OWNER = os.environ.get("ANACONDA_OWNER", "ga-fdp")
STAGING = os.environ.get("STAGING_LABEL", "staging")
PROTECTED = os.environ.get("PROTECTED_LABEL", "main")
ATTIC = os.environ.get("ATTIC_LABEL", "attic")
API = "https://api.anaconda.org"


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the token at the redirect hop.

    A download URL 302s to signed object storage. Forwarding an anaconda
    token there is both a 400 and a credential handed to a third party.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.headers = {k: v for k, v in new.headers.items()
                           if k.lower() != "authorization"}
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_StripAuthOnRedirect)


def _token():
    tok = os.environ.get("ANACONDA_API_TOKEN", "")
    if not tok:
        sys.exit("ANACONDA_API_TOKEN is unset. Refusing to guess.")
    return tok


def files_for(package, token):
    """Every published file of a package, with labels and checksums.

    From the REST API, never repodata: repodata is CDN-cached and can lag,
    and acting on stale labels is how this touches the wrong thing.
    """
    req = urllib.request.Request(f"{API}/package/{OWNER}/{package}",
                                 headers={"Authorization": "token " + token})
    try:
        return json.load(urllib.request.urlopen(req)).get("files", [])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise


def labels_of(package, token):
    """``{version: {basename: {label, ...}}}`` -- the shape decisions need."""
    out = {}
    for f in files_for(package, token):
        out.setdefault(f["version"], {})[f["basename"]] = set(f.get("labels") or [])
    return out


def _label_call(method, label, package, version, basename, token):
    body = json.dumps({"package": package, "version": version,
                       "basename": basename}).encode()
    req = urllib.request.Request(
        f"{API}/channels/{OWNER}/{label.lower()}", data=body, method=method,
        headers={"Authorization": "token " + token,
                 "Content-Type": "application/json"})
    urllib.request.urlopen(req)          # raises on non-2xx, unlike the CLI


def relabel(package, version, from_label, to_label, token):
    """Move one version's files between labels, and prove it happened.

    Adds before removing: if the process dies between the two, the artifact
    is in both labels rather than neither. Discoverable beats lost.
    """
    before = labels_of(package, token)
    if version not in before:
        sys.exit(f"{package} {version} is not on the channel at all, so it "
                 f"cannot be moved from '{from_label}'.")

    targets = [b for b, labels in before[version].items() if from_label in labels]
    if not targets:
        sys.exit(f"{package} {version} is not on '{from_label}' "
                 f"(it has {sorted(set().union(*before[version].values()))}). "
                 f"Refusing: a move that finds nothing must not report success.")

    for basename in targets:
        _label_call("POST", to_label, package, version, basename, token)
        _label_call("DELETE", from_label, package, version, basename, token)

    after = labels_of(package, token)
    for basename in targets:
        got = after.get(version, {}).get(basename, set())
        if to_label not in got or from_label in got:
            sys.exit(f"{package} {version} {basename} is {sorted(got)} after "
                     f"the move, expected '{to_label}' without '{from_label}'.")

    # The failure that made this module necessary: a label operation reaching
    # versions it was never asked about.
    for other, files in after.items():
        if other == version:
            continue
        for basename, got in files.items():
            was = before.get(other, {}).get(basename, set())
            if got != was:
                sys.exit(f"COLLATERAL: {package} {other} {basename} changed "
                         f"from {sorted(was)} to {sorted(got)} while moving "
                         f"{version}. Stop and investigate.")
    print(f"    {package} {version}: {len(targets)} file(s) "
          f"'{from_label}' -> '{to_label}'")
    return targets


def archive(record, into, token):
    """Download one file and prove it arrived intact. Returns its path."""
    blob = _opener.open(urllib.request.Request(
        "https:" + record["download_url"],
        headers={"Authorization": "token " + token})).read()

    actual, expected = hashlib.md5(blob).hexdigest(), record.get("md5")
    if expected and actual != expected:
        raise SystemExit(
            f"REFUSING to retire {record['full_name']}: downloaded bytes hash "
            f"{actual}, the channel says {expected}. The archive would not be "
            f"the artifact, so nothing is relabelled.")
    if len(blob) != record.get("size", len(blob)):
        raise SystemExit(
            f"REFUSING to retire {record['full_name']}: got {len(blob)} bytes, "
            f"expected {record['size']}.")

    into.mkdir(parents=True, exist_ok=True)
    path = into / record["basename"].split("/")[-1]
    path.write_bytes(blob)
    print(f"    archived {path.name} ({len(blob)} bytes, md5 {actual})")
    return path


def retire(package, keep, archive_dir):
    """Keep at most one unpromoted version of a package in staging.

    Nothing is deleted. A failed candidate is evidence -- "why did the gate
    reject 2.18.0?" is unanswerable once the bytes are gone -- and
    `anaconda remove` has no --label flag, so the obvious command would take
    every version including the ones users install.
    """
    token = _token()
    by_version = {}
    for f in files_for(package, token):
        by_version.setdefault(f["version"], []).append(f)
    if not by_version:
        print(f"{package}: not on the channel yet; nothing to retire")
        return 0

    for version, records in sorted(by_version.items()):
        labels = set().union(*(set(r.get("labels") or []) for r in records))
        if version == keep or STAGING not in labels:
            continue
        if PROTECTED in labels:
            print(f"REFUSING to retire {package} {version}: it is also on "
                  f"'{PROTECTED}'. Promotion moves rather than copies, so "
                  f"this state should be impossible -- investigate before "
                  f"retrying.", file=sys.stderr)
            return 1

        print(f"{package}: retiring superseded {version} from '{STAGING}'")
        for record in records:
            archive(record, pathlib.Path(archive_dir) / package / version, token)
        relabel(package, version, STAGING, ATTIC, token)
    return 0


def promote(package, version):
    """Move one version from staging to main. Nothing else moves."""
    token = _token()
    relabel(package, version, STAGING, PROTECTED, token)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "retire" and len(sys.argv) == 5:
        sys.exit(retire(sys.argv[2], sys.argv[3], sys.argv[4]))
    if len(sys.argv) >= 2 and sys.argv[1] == "promote" and len(sys.argv) == 4:
        sys.exit(promote(sys.argv[2], sys.argv[3]))
    sys.exit(__doc__.strip().splitlines()[2] + "\n" +
             __doc__.strip().splitlines()[3])
