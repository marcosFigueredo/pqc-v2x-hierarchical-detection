import h5py
import numpy as np
import pandas as pd

from src.config import RAW_H5_PATH
from src.features import audit_leakage

H5_ONLY_DERIVED_FIELDS = [
    "ZoneRisk_tunnel",
    "cos_timestamp",
    "sin_timestamp",
    "decryption_time",
    "enc_dec_ratio",
    "latency_ber_ratio",
    "risk_amp",
    "size_anomaly",
    "snr_packet_loss_ratio",
]

if __name__ == "__main__":
    n = 300_000
    with h5py.File(RAW_H5_PATH, "r") as f:
        data = {col: f[col][:n] for col in H5_ONLY_DERIVED_FIELDS}
        data["is_attack"] = f["is_attack"][:n].astype(int)
    df = pd.DataFrame(data)

    result = audit_leakage(df, candidate_cols=H5_ONLY_DERIVED_FIELDS)
    print(result.to_string(index=False))
