# F6 temporal sampling ablation

Complete 2026-10-05T20:02:46.477484+08:00; commit f46ce0111ed50472db2d4175786faf0ba8c66d22. Project-local ProcVLM/T-Rex Python; GPU0 frozen-feature extraction then8smallhead trainings sequential, no encoders finetuned/downloads/systemchanges. User requested fast experiment, so only required sampling/label/metric checks, no reconstruction/CPU replay/multiple seeds.

Command:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.f6_temporal_sampling --output outputs/sharpa_f6_temporal_sampling/20261005_195800 --device cuda:0 > logs/sharpa_f6_temporal_sampling_20261005_195800.log 2>&1
```

Fixed original_align,F6GRU128,Gaussian sigma8,seed42,rolloutsplit,optimizer and denseval/test. Feature-step andRaw-step each0/1/2/3 plusmulti;step0same so reusedbaseline,9uniqueconfigurations,8newruns. step=stride−1. Multi picks1variant/interaction/epoch, permuted4epoch cycle;batch/updatesperepochunchanged, totalvalidsupervisionticks differbystride. Shared dense-train normalization. Preserve originalGaussian positions andkey_index, not sigma8 rescaled bystride. Rawpast16sparsepoints only, firsttickleftpad. Feature/rawmatchedendpoint filtering dropped8invalidpoints. Stored features include exactraw tickwindows andoriginalpositions.

Verification PASS: sampledraws causal, samepairedendpoints, fixedGTval/test support, all9metrics recomputed. No broad safety checks beyond these. Allval/test GT-conditioned 3classprobabilitymean±SD curves exported; allCSVN preserved. Copies checked. Feature change cosine/relativeL2 computed ontrain only, no extra model testing.

Results: baseline testBA45.42%;Featuremulti54.53%(+9.11pp),Rawmulti40.81%(−4.61pp). Featurestep3 65.95%,Rawstep1 60.71%, no testbasedconfigselection. Seed42-only screening, cannot declare stable ordering. [OriginalREADME](../outputs/sharpa_f6_temporal_sampling/20261005_195800/README.md), [WeeklyREADME](../WeeklySummary/10.5/f6_temporal_sampling/20261005_195800/README.md).
