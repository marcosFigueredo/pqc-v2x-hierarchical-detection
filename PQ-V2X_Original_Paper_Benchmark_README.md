# PQ-V2X Original Paper Benchmark Dataset

## Overview

**PQ-V2X Original Paper Benchmark** is the paper-era version of the PQ-V2X dataset used for the original machine-learning benchmark experiments.

- **Primary data file:** `pq_v2x_realistic.csv`
- **Total records:** 5,000,000
- **Stored columns:** 55
- **Total classes:** 24
- **Attack classes:** 23
- **Benign class:** 1 (`normal`)
- **Observed real cryptographic traces:** 502,700 records (10.0540%)
- **Observed synthetic traces:** 4,497,300 records (89.9460%)
- **SHA-256:** `9239419680a342e3b14478cf2ebb2ac9e62bd76c019c1a7db032aec10cc47893`

This release is retained specifically for reproducibility of the original PQ-V2X paper-era benchmark. A later extended PQ-V2X release contains 65 stored columns and should not be silently substituted when reproducing the original experiments.

## Data Format

The dataset is distributed in CSV format. Each row represents one labeled V2X security observation with mobility, communication, cryptographic, attack, and trace-related attributes.

## Class Distribution

### Paper-level attack families

| Family | Records | Percentage |
|---|---:|---:|
| Normal | 3,006,500 | 60.1300% |
| Post-Quantum | 894,700 | 17.8940% |
| Hybrid | 615,100 | 12.3020% |
| Classical | 483,700 | 9.6740% |

For paper-level reporting, several quantum-inspired and transition attack classes are grouped under the **Hybrid** family. The original per-class labels remain unchanged in the dataset.

## Attack Taxonomy

The dataset contains 23 attack classes plus the normal class.

| Class label | Attack type | Paper reporting family |
|---:|---|---|
| 0 | `normal` | Normal |
| 1 | `PQ_Kyber_ReusedKey` | Post-Quantum |
| 2 | `PQ_Dilithium_SideChannel` | Post-Quantum |
| 3 | `PQ_NTRU_KeyMismatch` | Post-Quantum |
| 4 | `PQ_SPHINCS_WeakNonce` | Post-Quantum |
| 5 | `PQ_Kyber_TimingSkew` | Post-Quantum |
| 6 | `PQ_Falcon_SignatureSizeSpike` | Post-Quantum |
| 7 | `PQ_Dilithium_InvalidFormat` | Post-Quantum |
| 8 | `PQ_SPHINCS_FallbackCollision` | Post-Quantum |
| 9 | `PQ_MalformedPublicKey` | Post-Quantum |
| 10 | `PQ_KeyReuse_AcrossZones` | Post-Quantum |
| 11 | `Grover_Approximate_Collision` | Hybrid |
| 12 | `Shor_Signature_Recovery` | Hybrid |
| 13 | `ManInTheMiddle_PQ` | Hybrid |
| 14 | `Latency_Hijack_QuantumNoise` | Hybrid |
| 15 | `Classic_Replay` | Classical |
| 16 | `Classic_DDoS` | Classical |
| 17 | `Classic_MITM` | Classical |
| 18 | `Classic_SpoofedCert` | Classical |
| 19 | `Certificate_Transparency_Bypass` | Hybrid |
| 20 | `Hybrid_PQAuth_ClassicChannel` | Hybrid |
| 21 | `Hybrid_ProtocolDesync` | Hybrid |
| 22 | `Hybrid_KeyTampering` | Hybrid |
| 23 | `Hybrid_SignatureCrossUse` | Hybrid |

## Trace Interpretation

The dataset contains an `is_real_trace` indicator and a `trace_status` field.

- `real`: 502,700 rows (10.0540%)
- `synthetic`: 4,497,300 rows (89.9460%)

**Important:** "real trace" refers to records generated through the configured real cryptographic execution/trace path. It does **not** mean that the associated vehicle trajectory or wireless-network observation was collected from a real-world road deployment.

## RSU Zone Distribution

| RSU zone | Records | Percentage |
|---|---:|---:|
| Highway | 1,265,850 | 25.3170% |
| Tunnel | 1,249,700 | 24.9940% |
| Rural | 1,245,600 | 24.9120% |
| Urban | 1,238,850 | 24.7770% |

## Stored Message-Type Distribution

| Message type | Records | Percentage |
|---|---:|---:|
| Navigation | 1,254,600 | 25.0920% |
| Status | 1,254,450 | 25.0890% |
| Emergency | 1,246,050 | 24.9210% |
| Infotainment | 1,244,900 | 24.8980% |

## Stored Protocol Values

The dataset contains records associated with these protocol labels:

- `Kyber512`
- `Dilithium2`
- `Falcon-512`
- `SPHINCS+-SHA256-128s`
- `NTRU-HPS-2048-509`
- `ECDSA-SECP256k1`
- `AES256-GCM`
- `TLS1.3`

These names reflect the software/protocol identifiers used during dataset generation.

## Column Structure

The paper-era release contains 55 stored columns:

1. `timestamp`
2. `vehicle_id`
3. `latitude`
4. `longitude`
5. `speed`
6. `acceleration`
7. `direction`
8. `rsu_zone`
9. `msg_type`
10. `source`
11. `label`
12. `class_label`
13. `source_ip`
14. `attack_type`
15. `Protocol`
16. `session_id`
17. `is_attack`
18. `data_integrity`
19. `vehicle_type`
20. `protocol`
21. `snr`
22. `packet_loss`
23. `latency`
24. `ber`
25. `attack_family`
26. `attack_severity`
27. `risk_score`
28. `is_real_trace`
29. `KeyReuseCount`
30. `EncapTimeDeviation`
31. `EntropyDeviation`
32. `KeyGuessAttempts`
33. `SignatureSize`
34. `SpoofCertMatch`
35. `TimingVariance`
36. `LeakageSignal`
37. `protocol_mismatch`
38. `zone_mismatch_flag`
39. `sequence_gap`
40. `timestamp_offset`
41. `message_frequency`
42. `source_entropy`
43. `handshake_time`
44. `cipher_suite`
45. `encryption_time`
46. `block_mode`
47. `sig_generation_time`
48. `sig_verification_time`
49. `pqc_public_key`
50. `pqc_ciphertext`
51. `pqc_signature`
52. `trace_status`
53. `synthetic_anomaly_score`
54. `detected_flag`
55. `detection_method`

## Important Modeling Note

Not every stored column should be used as a machine-learning input feature.

Columns that directly reveal the target, attack identity, or post-hoc detection outcome should be excluded from predictive inputs as appropriate. Examples include:

- `label`
- `class_label`
- `attack_type`
- `is_attack`
- `attack_family`
- `attack_severity`
- `detected_flag`
- `detection_method`

Researchers should define and report their exact feature-selection policy before training models.

## Loading Example

```python
import pandas as pd

df = pd.read_csv("pq_v2x_realistic.csv", low_memory=False)

print(df.shape)
print(df["attack_type"].value_counts())
print(df["is_real_trace"].value_counts())
```

For memory-constrained systems:

```python
import pandas as pd

for chunk in pd.read_csv(
    "pq_v2x_realistic.csv",
    chunksize=100_000,
    low_memory=False
):
    # process one chunk at a time
    pass
```

## Integrity Verification

After downloading the release, verify the file fingerprint:

```bash
sha256sum pq_v2x_realistic.csv
```

Expected SHA-256:

```text
9239419680a342e3b14478cf2ebb2ac9e62bd76c019c1a7db032aec10cc47893
```

## Intended Uses

The dataset is intended for research on:

- V2X intrusion detection
- post-quantum and hybrid-security threat classification
- machine-learning and deep-learning IDS evaluation
- attack-family analysis
- V2X cybersecurity benchmarking
- reproducibility studies

## Version Note

This documentation describes the **PQ-V2X Original Paper Benchmark** release with 55 stored columns.

A separate later extended PQ-V2X release contains 65 stored columns and additional derived fields. The two versions should be treated as distinct dataset releases when reporting experiments.

## Citation

Please cite the corresponding IEEE DataPort dataset record and the associated PQ-V2X paper when using this dataset.
