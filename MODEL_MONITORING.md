# AEGIS AI - Model Monitoring (Sprint 14)

AEGIS AI implements deterministic, offline, and research-only drift monitoring for Champion models. This ensures that the current Champion model's behavior and input data distribution remains consistent with its approved research baseline.

## Core Principles

1. **OFFLINE AND RESEARCH ONLY**: The monitoring system operates strictly on historical data windows (observations) and never connects to live broker feeds. It never executes automated live responses to drift.
2. **DETERMINISTIC EVALUATION**: The identical ReferenceProfile, MonitoringWindow, and MonitoringPolicy will unconditionally yield the identical HealthAssessment and MonitoringAlerts. Alert IDs are deterministic and derived from alert contents.
3. **NO AUTOMATED REPLACEMENT**: A degraded or invalid Champion generates governance alerts for human review. It does **NOT** automatically retrain the model, demote the Champion, or promote a Challenger.
4. **SAFETY FIRST**: Explicit handlers prevent Zero-Division, NaN, and Infinity leaks into health reporting. Minimum sample sizes are enforced.
5. **CORRUPTION HANDLING**: NaN/Infinity prediction data is explicitly treated as corruption and causes an INVALID health state with a CRITICAL alert.
6. **INSUFFICIENT LABELS**: Insufficient labeled data produces an explicit WARNING alert to maintain auditability. However, insufficient labels do not automatically invalidate the model (results in DEGRADED).

## Components

### Reference Profile
An immutable snapshot of the Champion model's statistical properties during its approved evaluation period. Once created, it cannot be mutated.
Includes:
- Feature mean, standard deviation, missingness.
- Prediction class distributions and mean confidence.
- Historical performance (F1, Accuracy, Win Rate, Max Drawdown).

### Monitoring Window
An observation period over which new predictions or historical test data are accumulated. Compared deterministically against the Reference Profile.

### Monitoring Policy
A set of defined thresholds such as `max_feature_mean_shift`, `max_prediction_divergence`, and `max_drawdown_increase`.

### Champion Health States
- **HEALTHY**: No policy thresholds were breached.
- **DEGRADED**: One or more WARNING alerts were raised (e.g., statistical drift in data or predictions).
- **INVALID**: A CRITICAL alert was raised (e.g., mismatched schemas, corrupted identities, or unparseable artifacts).

## Alerts and Governance Integration
When a HealthAssessment is generated, it encapsulates the drift report and any MonitoringAlerts. These are persisted deterministically to the governance schema.
Events `ChampionHealthAssessedEvent` and `MonitoringAlertRaisedEvent` are dispatched to notify the supervisor, maintaining strict isolation from the execution layer.
