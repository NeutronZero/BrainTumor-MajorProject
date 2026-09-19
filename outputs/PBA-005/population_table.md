# PBA-005 population / denominator table (reporting layer, frozen artifacts)

Unmeasured != zero. Each joint metric uses only the population where its
numerator AND denominator are defined. No model rerun; derived from
per_case_test.json + per_case_segtest.json linkage by image stem.

| Population | N | Classification measured | Segmentation measured | Joint outcome |
| --- | ---: | --- | --- | --- |
| All classification test | 1000 | Yes | -- | -- (no joint denominator) |
| Segmentation test population | 860 | Yes/linked | Yes | measurable |
| Matched cls+seg (joint denominator) | 860 | Yes | Yes | 846 concordant / 10 potential-localization-failure / 4 uncertainty-pathway |
| Healthy classification cases (true notumor, no seg GT) | 140 | Yes | applicability differs: no tumor GT -> unmeasured | NOT assumed zero; excluded from joint denominator |
| cls-only stems (unmeasured for joint) | 140 | Yes | No | unmeasured |
| seg-only stems | 0 | -- | -- | unmeasured (none) |

Verification: 860 matched + 140 cls-only = 1000 classification cases; all 140
cls-only stems are true-notumor in per_case_test.json; seg population is
254 glioma / 306 meningioma / 300 pituitary (sums to 860, zero healthy).
Joint rates: 10/860 matched = 1.16% of matched; 10/856 confident-tumor matched
= 1.17% (locked-eval definition). Both denominators stated; neither treats the
140 healthy cases as zero outcomes.
