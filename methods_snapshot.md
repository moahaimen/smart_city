# Methods Snapshot

## TCN Role
The TCN predicts next-step PM2.5 from recent multivariate pollution windows. Its predicted severity is fed into the routing priority score for TCN-PPA-LEACH.

## Severity Mapping
PM2.5 is mapped into three classes: Normal, Warning, and Hazardous using configurable AQI-style thresholds.

## AoI Definition
Age of Information at the sink increments by one each round when no fresh packet from a node arrives and resets to zero on successful delivery.

## Priority Score
The node priority score combines current severity, predicted severity, AoI, change rate, hotspot relevance, and communication cost using configurable weights.

## Cluster-Head Election
Standard LEACH uses probabilistic election. Energy-aware LEACH uses residual-energy and distance ranking. TCN-PPA-LEACH uses residual energy, predictive priority, and sink distance.

## Suppression Rule
Only TCN-PPA-LEACH applies routine suppression. Hazardous packets are never suppressed. Warning and hazardous packets always remain eligible for transmission.

## Compared Baselines
- LEACH
- EA-LEACH
- TCN-PPA-LEACH
