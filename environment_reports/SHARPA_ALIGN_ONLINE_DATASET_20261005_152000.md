# Sharpa online dataset revision — no training

- Two groups × n=0/2/5/10 × step=1/3/5/8/12; 40 parameter cases.
- Window16, posterior-half Key membership, nearest preceding label, mixed Align merged as success.
- Original rollout split80/17/17 preserved.
- CPU only, reused repos/ProcVLM/.venv; frozen F6 boundary prefixes encoded, no decoder or reconstruction.
- 629592 grid windows verified; source sensor/feature/encoder/annotation snapshot hashes PASS.
- ready/train: symmetric7376 per class, asymmetric merged4626 per class. Val/test unbalanced reference n5 step3.
- No optimizer, backward, checkpoint, or model training.
- [Original README](../outputs/sharpa_align_online_datasets/20261005_152000/README.md)
- [Log](../logs/sharpa_align_online_dataset_20261005_152000.log)
- [Code/version record](../outputs/sharpa_align_online_datasets/20261005_152000/generation_record.json)
- [Weekly summary](../WeeklySummary/10.5/10.5.md)
