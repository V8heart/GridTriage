#!/usr/bin/env bash
# Download the MITRE sources of the Stage2 knowledge base, pinned to the versions used for the
# reported results (ATT&CK Enterprise/ICS 19.2, ATLAS 5.6.0), and verify their SHA-256.
#
#   bash scripts/fetch_kb_sources.sh            # -> dataset/kb/raw/
#   python -m pipeline.kb.build_kb               # -> dataset/kb/kb_docs.jsonl (1,091 docs)
#
# ATLAS: the parser reads the legacy single-file format (dist/ATLAS.yaml), pinned to the
# atlas-data commit that last changed it (version 5.6.0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${KB_RAW_DIR:-$ROOT/dataset/kb/raw}"
mkdir -p "$DEST"

fetch() {  # name url sha256
  local name="$1" url="$2" sha="$3" out="$DEST/$1"
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --status; then
    echo "ok (cached) $name"; return
  fi
  echo "downloading $name"
  curl -fsSL --retry 3 -o "$out.part" "$url"
  echo "$sha  $out.part" | sha256sum -c --status || { echo "checksum mismatch: $name" >&2; rm -f "$out.part"; exit 1; }
  mv "$out.part" "$out"
  echo "ok $name"
}

fetch enterprise-attack.json \
  "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack-19.2.json" \
  dc1639caa5501d720e280cf1cbd8fbe009884a0c9b3e6e9ed9d0c25166c3d8f4
fetch ics-attack.json \
  "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack-19.2.json" \
  08b83d2cea6b6d6752468ef0e62e2ab2a53c9443ef72c439ecccb07ab9e89da9
fetch ATLAS.yaml \
  "https://raw.githubusercontent.com/mitre-atlas/atlas-data/da9ebf9b66e6902ad97c267e2a20af0bd996a60f/dist/ATLAS.yaml" \
  c9c23971d0db07e2782423d795a20e78d5c2c9bcdc78fc67e24530657daaec41

echo "KB sources ready in $DEST (MITRE ATT&CK / ATLAS: see their terms of use)"
