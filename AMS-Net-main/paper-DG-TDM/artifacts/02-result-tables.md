# DG-TDM Result Tables (TBD pending training)

No experimental result has been generated here. All TBD cells must be filled
from user-run experiments or verified public baseline reports with matching
protocol.

## Main comparison (Diving48, top-1 accuracy %)

| Method | Backbone | Frames | Top-1 (%) |
| --- | --- | --- | --- |
| Temporal action encoder (baseline) | ResNet-50 | 8 | TBD |
| Frozen CLIP + adapter (EVL) | ResNet-50 + ViT-B/16 | 8 | TBD |
| DG-TDM (ours) | ResNet-50 + frozen CLIP | 8 | TBD |

## Ablation (Diving48, top-1 accuracy %)

| Variant | Component changed | Top-1 (%) |
| --- | --- | --- |
| Full method | none | TBD |
| w/o STG | remove guidance branch | TBD |
| w/o MoDeA | remove motion deformation attention | TBD |
| w/o TSA | remove temporal state aggregation | TBD |
| w/o DG-TDM | remove both branches | TBD |

## Claim-evidence matrix

| Claim | Reviewer question | Evidence needed | Status |
| --- | --- | --- | --- |
| C1: MoDeA captures motion change | Does local deformable sampling help? | w/o MoDeA ablation | planned |
| C2: STG guidance focuses action region | Does spatial prior injection help? | w/o STG ablation | planned |
| C3: TSA models channel-to-temporal variation | Does state aggregation help beyond local motion? | w/o TSA ablation | planned |
| C4: DG-TDM improves accuracy | Does full method beat baselines? | main table | planned |
