# F6 sampling allsteps five-seed completion

Completed 2026-10-05T20:33:09.365744+08:00; repositorycommit f46ce0111ed50472db2d4175786faf0ba8c66d22. Existing project-local ProcVLM/T-Rex Python. User authorized repeated small frozen-encoder heads; GPU0 sequential, no encoder recomputation/finetuning/systemchanges/downloads. Reuse original source data and step caches.

Command:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.f6_temporal_sampling_seeds --output outputs/sharpa_f6_temporal_sampling/20261005_203200 --device cuda:0 > logs/sharpa_f6_temporal_sampling_20261005_203200.log 2>&1
```

PASS:19uniqueconfigs×seeds42–46=95results,19seed42reused,76newruns. Feature/Rawstep0sharedbaseline,eachstep1..8,andmulti0..3bothmodes. Sigma8fixed,allval/testdense0,sameoriginalrolloutsplit,optimizer,norm,labels. All95confusion matrices recomputed fromprobabilities; config cardinality and5seed coverageverified; original/copy docsandCSV hashesmatch. Minimalchecks only peruserrequest; no expanded replay or reconstruction. Fourfull Key-relativeprobabilityplots pluscomparisonPDF/PNG; curve variance acrossseed-averagedinteractions separatefrommetricseedSD. DeduplicatedbaselineCSVentries presentinbothplotlayouts without alteringprobabilities ormetrics.

Main result: denseBA56.86±10.57%;Featuremulti50.52±4.48%,paired−6.34±12.21pp;Rawmulti45.31±4.92%,paired−11.55±9.14pp. This revises priorseed42Featuremulti improvement: not retainedacross5seeds. Rawstep8valBA43.12±3.50%,test40.89±8.15%;Featurestep8val61.61%(SD inCSV),test47.52±4.04%. Encoderwindowpadding andGRUfrequency confoundisolatedencoderattribution. No test-basedconfigselection orarchitecturechanges.

[Originalreport](../outputs/sharpa_f6_temporal_sampling/20261005_203200/README.md), [Weeklyreport](../WeeklySummary/10.5/f6_temporal_sampling/20261005_203200/README.md). Priorseed42reportslinktothiscanonicalsummary.
