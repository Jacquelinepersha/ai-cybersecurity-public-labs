
import pandas as pd

from .mitre_data import (
    object_catalog,
    relationship_table,
    technique_catalog,
)

def group_technique_edges(bundle):
    """Return ATT&CK group-to-technique relationships with readable names."""
    groups = object_catalog(bundle, "intrusion-set").rename(
        columns={
            "stix_id": "group_stix_id",
            "external_id": "group_id",
            "name": "group_name",
        }
    )

    techniques = technique_catalog(bundle, id_prefix="T").rename(
        columns={
            "stix_id": "technique_stix_id",
            "external_id": "technique_id",
            "name": "technique_name",
        }
    )

    rel = relationship_table(bundle)
    rel = rel[rel["relationship_type"] == "uses"].copy()

    edges = (
        rel.merge(
            groups[["group_stix_id", "group_id", "group_name"]],
            left_on="source_ref",
            right_on="group_stix_id",
            how="inner",
        )
        .merge(
            techniques[
                ["technique_stix_id", "technique_id", "technique_name", "tactics"]
            ],
            left_on="target_ref",
            right_on="technique_stix_id",
            how="inner",
        )
    )

    return edges[
        [
            "group_id",
            "group_name",
            "technique_id",
            "technique_name",
            "tactics",
        ]
    ].drop_duplicates()

def technique_usage_frequency(bundle):
    """Count how many ATT&CK groups are associated with each technique."""
    edges = group_technique_edges(bundle)

    return (
        edges.groupby(
            ["technique_id", "technique_name"],
            as_index=False,
        )
        .agg(group_count=("group_id", "nunique"))
        .sort_values("group_count", ascending=False)
    )

def group_profile(bundle, group_name):
    """Return all mapped techniques for a named ATT&CK group."""
    edges = group_technique_edges(bundle)

    mask = edges["group_name"].str.contains(
        group_name,
        case=False,
        na=False,
        regex=False,
    )

    return edges[mask].sort_values(
        ["tactics", "technique_id"]
    ).reset_index(drop=True)

def tactic_coverage_for_group(profile):
    """Count mapped techniques per tactic string for a selected group profile."""
    if profile.empty:
        return pd.DataFrame(columns=["tactic", "technique_count"])

    exploded = profile.copy()
    exploded["tactic"] = exploded["tactics"].str.split(", ")
    exploded = exploded.explode("tactic")
    exploded = exploded[
        exploded["tactic"].notna()
        & (exploded["tactic"] != "")
    ]

    return (
        exploded.groupby("tactic", as_index=False)
        .agg(technique_count=("technique_id", "nunique"))
        .sort_values("technique_count", ascending=False)
    )
