# AEGIS AI Model Governance

## Overview
Model Governance in AEGIS AI is responsible for maintaining the identity, lifecycle, and safe promotion of candidate models to the status of research champion. 

## Identity and Versioning
Models in AEGIS are strictly versioned.
- **ModelIdentity**: A deterministic identifier derived from the logical configuration (model type, feature schema, target configuration, training configuration, calibration config, and seed). Changes to any of these fields generate a new `ModelIdentity`.
- **ArtifactMetadata**: Stored alongside serialized model files. It tracks fingerprint, integrity hash, dataset used, and the specific experiment that produced it. The platform verifies artifact integrity before loading.

## Experiment Tracking
Each model is trained in the context of an `ExperimentRecord`. The record is immutable and captures the configurations used, the models generated, and the final walk-forward evaluation metrics (e.g. Mean F1, Max Drawdown).

## Reproducibility and Artifact Integrity
Before a model can be considered for promotion, its reproducibility and physical artifact integrity must be verified.
- **Reproducibility Receipts**: Generated via `ReproducibilityVerifier` to prove that the current environment (including source configuration) deterministically recreates the expected logical footprint (fingerprint) of the model.
- **Artifact Verification**: Ensured via cryptographic hashes (`integrity_hash`) when an artifact is saved or loaded. A mismatch rejects the load attempt and protects against data corruption or unauthorized modifications.

### Security Limitation: Trusted-Local Artifacts Only
> [!CAUTION]
> The current artifact format relies on `joblib` / `pickle`-style serialization.
> - **Artifacts are trusted-local artifacts only.**
> - `SHA-256` verification ensures integrity against disk corruption or internal tampering, **but does not guarantee code safety**.
> - Deserialization with `joblib` may execute arbitrary code. 
> - **External, untrusted, or remotely downloaded artifacts MUST NOT be loaded into the registry.**

## Persistent Model Registry
The governance registry uses a SQLAlchemy-backed persistent schema (`governance_registry` and `governance_audit` tables).
- A central audit trail logs every lifecycle event (REGISTER, DEMOTE, PROMOTE, REJECT, RETIRE, SUPERSEDE) for full forensics tracking.
- The registry acts as an abstraction over the persistent layer. It validates artifacts on load before caching them in-memory, ensuring that the working model instances exactly match the recorded truth.

## Model Registry Lifecycle
The registry governs the statuses of models. The statuses are:
- `CANDIDATE`: A newly registered model waiting for evaluation.
- `CHALLENGER`: A candidate being formally evaluated against the current champion.
- `PROMOTED`: A model recommended for promotion.
- `CHAMPION`: The single active research champion model.
- `REJECTED`: A candidate that failed to meet promotion criteria.
- `SUPERSEDED`: A previous champion that has been replaced.
- `RETIRED`: A model that is no longer in use.

## Promotion Policy and Decisions
To become a `CHAMPION`, a model must satisfy the explicit `PromotionPolicy`. This policy dictates mandatory criteria:
- Minimum walk-forward evaluation windows
- Minimum acceptable evaluation metrics (e.g. Mean F1)
- Maximum allowable drawdown
- A requirement to beat a simple baseline model
- Cryptographic reproducibility and artifact integrity validation
- Preservation of the strict `PREDICTION_ONLY` ExecutionMode

**Highest PnL Does Not Win:** The evaluation guarantees multi-dimensional robustness over just maximizing theoretical returns. A model with high theoretical returns but unstable confidence or excessive drawdown will be automatically rejected.

## Important Note on "Live" Models
> [!WARNING]
> PROMOTED ≠ LIVE. 
> A `PROMOTED` or `CHAMPION` model in the governance registry is strictly an "approved research model." The system's execution capability remains fully locked in `PREDICTION_ONLY` mode. No governance action can autonomously enable order submission, capital allocation, or live exchange integrations. 
