# Final Demonstration Record (deterministic, synthetic inputs only)

Narrated walkthrough of the frozen system. Every step below is
pinned by committed acceptance evidence; re-running the demo suite
reproduces these outcomes exactly.

## healthy_like
- Expected contract: `{'predicted_class': 'notumor'}`
- Observed: class=notumor, state=uncertain, system=uncertain, seg=empty, quality=accept
- Contract holds: yes

## tumor_like
- Expected contract: `{'predicted_class': 'pituitary', 'classification_state': 'confident'}`
- Observed: class=pituitary, state=confident, system=tumor_unlocalized, seg=empty, quality=accept
- Contract holds: yes

## low_quality
- Expected contract: `{'quality': 'reject'}`
- Observed: class=glioma, state=confident, system=tumor_unlocalized, seg=empty, quality=reject
- Contract holds: yes

## seg_empty
- Expected contract: `{'predicted_class': 'glioma', 'segmentation_state': 'empty'}`
- Observed: class=glioma, state=confident, system=tumor_unlocalized, seg=empty, quality=reject
- Contract holds: yes

## seg_localized
- Expected contract: `{'predicted_class': 'meningioma', 'system_state': 'tumor_localized'}`
- Observed: class=meningioma, state=confident, system=tumor_localized, seg=nonempty, quality=accept
- Contract holds: yes

## uncertain
- Expected contract: `{'classification_state': 'uncertain'}`
- Observed: class=glioma, state=uncertain, system=uncertain, seg=empty, quality=reject
- Contract holds: yes

## degraded_no_segmenter
- Expected contract: `{'system_state': 'degraded'}`
- Observed: class=None, state=None, system=degraded, seg=None, quality=None
- Contract holds: yes

## Operator walkthrough
Recorded steps: start_system → load_image → read_model_output → read_observers → inspect_explanation → interpret_limitations → api_health → stop_system

## API acceptance
Schema regression: no failures. Service/API equivalence holds.

> Research prototype — not clinical. Demonstrations show system
> contracts, never diagnostic accuracy.
