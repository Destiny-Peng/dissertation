# LF3R incremental decoder latent experiments

Latents come from the **last token of the opening `<score>` tag**.
The saved vector is that token's final normalized decoder hidden state, used
by the model to predict the next `+`, `-`, `0`, etc. token. When the tag spans
multiple tokens, select the token completing the tag; do not advance one token.
The extraction uses the existing vLLM model, prompt, multimodal inputs, sampling
parameters and generation call. Only incremental mode captures features; fused
runs still generate all three perspectives using their existing behavior.

## Generate latents manually

In WebUI Review, select Robo-Dopamine and open Configure baseline. Enable
**Extract latent / re-inference (incremental)** and choose **Fused** for the
three-way comparison. Launch the rollout using the existing Run controls.
For a batch, enable the same option and choose result filter **All** so completed
progress outputs are included in re-inference. New runs use their own output
folders. Existing progress-only runs do not acquire latent features retroactively.

The Review result card reports `Latent: generated` or `Latent: missing`.
Old `score_next` artifacts are shown as an old token position and must be
re-extracted. Training rejects these artifacts rather than mixing positions.
If missing, enable extraction and rerun. Incremental-only runs can save latents,
but Localization Lab comparisons require fused outputs from that same run.

The existing LF3R baseline CLI also accepts `--robo-extract-latent` with
`--robo-eval-mode fused` (or incremental). Direct upstream Python callers can
construct `GRMInference(model_path, extract_latent=True)` and run the normal
pipeline with `eval_mode="incremental"`.

Each incremental output directory contains `latent_features.npz` alongside
`sample.json` and `pred_vllm.json`. The archive contains:

- `features`: float32 `[samples, 2560]`.
- `sample_ids`, zero-based `sample_indices`, original `before_frame_indices`
  and `frame_indices` (the after frame, matching the progress sample).
- `token_indices`: zero-based indices in the generated token sequence.
- `mode=incremental`, `position=score_start`, and `schema_version=1`.

Missing score tags, truncated/unprocessed selected tokens, wrong hidden widths,
and incomplete sample sets produce errors rather than substitute features.
The hook uses `examples.latent.LatentWorkerExtension` and named collective RPCs,
so it works with vLLM's default safe serialization. Returned features use native
containers over RPC; the saved archive still contains float32 arrays.
The hook targets the installed vLLM 0.11 V1 GPU model runner, with no speculative
or pipeline-parallel decoding. Capturing copies decoder states to CPU during
incremental generation and adds overhead. Extraction is opt-in.

## Train manually in Localization Lab

Load one of the following built-in presets, then use the existing Run button:

| Preset | Input | BiLSTM input width |
| --- | --- | --- |
| fused progress → BiLSTM | Existing fused `[progress, hop]` | 2 |
| PCA-32 latent → BiLSTM | Incremental hidden state projected by PCA | 32 |
| PCA-32 latent + fused → BiLSTM | PCA latent plus fused `[progress, hop]` | 34 |
| PCA-64 latent → BiLSTM (default) | Incremental hidden state projected by PCA | 64 |
| PCA-64 latent + fused → BiLSTM (default) | PCA latent plus fused `[progress, hop]` | 66 |
| PCA-128 latent → BiLSTM | Incremental hidden state projected by PCA | 128 |
| PCA-128 latent + fused → BiLSTM | PCA latent plus fused `[progress, hop]` | 130 |

The latent modes are `robodopamine_latent` and
`robodopamine_latent_plus_fused`. The default `data.pca_components=64` is
editable in the builder. Start with the PCA-64 presets. PCA-32/64/128 require at least 33/65/129 training samples respectively.
Each repeat fits PCA after the rollout split, using only failure train rollouts
and any selected success-negative training rollouts. Validation/test samples
never fit PCA. The fitted mean/components and fit rollout IDs are saved in the
checkpoint's `latent_pca`; checkpoint consumers reuse them without refitting.

All comparison presets share seeds, rollout split settings, labels, loss, model size,
and training settings. Keep those settings equal for comparisons. Latent and
fused inputs must come from the same run with identical native frame indices and
incremental sample IDs. Incomplete latent coverage fails explicitly, so a latent
experiment cannot silently use fewer rollouts than the fused anchor.

## Verification

Code checks use CPU fixtures with simulated generation and a mocked training
function. No real Robo-Dopamine inference, GPU job, or optimizer training was run
for this change. Actual extraction and training should be launched manually.

### Small annotated cohort analysis

Latent extraction alone does not supply localization labels. Choose trajectories
with existing failure onset annotations when testing the localization head.
In Localization Lab, set **Baseline run path (optional)** to the timestamped
Robo-Dopamine output directory containing `run.json`. In JSON this is
`base.data.source_run_root`; leave it empty to use the historical baseline pool.
Use the same path for fused, latent and latent+fused comparisons. All configurations
share this source; changing the path in a sweep or variant is rejected.
The analysis output includes `data_preflight.json` with annotation and signal
exclusions, including when the dataset is rejected before training.
