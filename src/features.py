
import pandas as pd

def drop_identifier_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        c for c in ["id", "ID", "row_id", "index"] if c in df.columns
    ]
    return df.drop(columns=identifiers, errors="ignore")

def add_safe_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"sbytes", "dbytes", "dur"}.issubset(out.columns):
        total_bytes = out["sbytes"].fillna(0) + out["dbytes"].fillna(0)
        out["bytes_per_second_safe"] = total_bytes / (out["dur"].abs() + 1e-6)

    if {"sbytes", "dbytes"}.issubset(out.columns):
        out["src_dst_byte_ratio_safe"] = (
            out["sbytes"].fillna(0) + 1.0
        ) / (
            out["dbytes"].fillna(0) + 1.0
        )

    if {"spkts", "dpkts"}.issubset(out.columns):
        out["src_dst_packet_ratio_safe"] = (
            out["spkts"].fillna(0) + 1.0
        ) / (
            out["dpkts"].fillna(0) + 1.0
        )

    return out
