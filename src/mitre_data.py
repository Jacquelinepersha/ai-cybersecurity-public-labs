
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ATTACK_ENTERPRISE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/"
    "attack-stix-data/master/enterprise-attack/enterprise-attack.json"
)

ATLAS_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-atlas/"
    "atlas-navigator-data/main/dist/stix-atlas.json"
)

def load_json_url(url, timeout=30):
    """Load JSON over HTTPS using Python's standard library."""
    request = Request(
        url,
        headers={"User-Agent": "AI-Cybersecurity-Masterclass/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def load_json_file(path):
    """Load a local JSON bundle."""
    return json.loads(Path(path).read_text())

def load_remote_or_local(url, local_path=None):
    """Try the official remote URL first, then an optional local fallback."""
    try:
        return load_json_url(url)
    except Exception:
        if local_path is not None and Path(local_path).exists():
            return load_json_file(local_path)
        raise

def external_id(obj):
    """Return the first MITRE-style external ID, when present."""
    for ref in obj.get("external_references", []):
        value = ref.get("external_id")
        if value:
            return value
    return None

def active_objects(bundle, object_type=None):
    """Return active STIX objects, optionally filtered by type."""
    objects = bundle.get("objects", [])
    result = []

    for obj in objects:
        if object_type is not None and obj.get("type") != object_type:
            continue
        if obj.get("revoked") is True:
            continue
        if obj.get("x_mitre_deprecated") is True:
            continue
        result.append(obj)

    return result

def technique_catalog(bundle, id_prefix=None):
    """Extract active attack-pattern objects into a DataFrame."""
    rows = []

    for obj in active_objects(bundle, "attack-pattern"):
        ext_id = external_id(obj)
        if id_prefix and (not ext_id or not ext_id.startswith(id_prefix)):
            continue

        tactics = sorted({
            phase.get("phase_name")
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("phase_name")
        })

        rows.append({
            "stix_id": obj.get("id"),
            "external_id": ext_id,
            "name": obj.get("name"),
            "tactics": ", ".join(tactics),
            "description": obj.get("description", ""),
            "modified": obj.get("modified"),
        })

    return pd.DataFrame(rows)

def object_catalog(bundle, object_type):
    """Extract a lightweight catalog for STIX objects such as groups/software."""
    rows = []

    for obj in active_objects(bundle, object_type):
        rows.append({
            "stix_id": obj.get("id"),
            "external_id": external_id(obj),
            "name": obj.get("name"),
            "description": obj.get("description", ""),
            "modified": obj.get("modified"),
        })

    return pd.DataFrame(rows)

def relationship_table(bundle):
    """Extract active STIX relationships."""
    rows = []

    for obj in active_objects(bundle, "relationship"):
        rows.append({
            "relationship_type": obj.get("relationship_type"),
            "source_ref": obj.get("source_ref"),
            "target_ref": obj.get("target_ref"),
            "description": obj.get("description", ""),
        })

    return pd.DataFrame(rows)
