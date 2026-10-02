"""Capture final decoder activations on the existing vLLM generation path.

vLLM 0.11 V1 compute_logits receives the final, normalized decoder output
for the *input* token. It predicts the next token; do not confuse these two
positions. Capture generated-token inputs and select the tag boundary after
normal generation finishes. No prompt or sampling parameter is changed.
"""
from __future__ import annotations
import re
import numpy as np


def install_capture(worker):
    runner = worker.model_runner
    if not hasattr(runner, "input_batch") or not hasattr(runner, "seq_lens"):
        raise RuntimeError("Latent capture requires vLLM 0.11 V1 GPUModelRunner")
    if runner.vllm_config.speculative_config is not None:
        raise ValueError("Latent capture does not support speculative decoding")
    if runner.vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Latent capture does not support pipeline parallelism")
    if hasattr(runner, "_lf3r_latents"):
        return
    runner._lf3r_latents = None
    original = runner.model.compute_logits

    def compute_logits(hidden_states, *args, **kwargs):
        if runner._lf3r_latents is not None:
            batch = runner.input_batch
            req_ids = list(batch.req_ids)
            if hidden_states.shape[0] != len(req_ids):
                raise RuntimeError("Latent capture expected one logit row per request")
            indices = (runner.seq_lens.np[:len(req_ids)] - 1
                       - batch.num_prompt_tokens[:len(req_ids)])
            selected = [i for i, index in enumerate(indices) if index >= 0]
            if selected:
                values = hidden_states[selected].detach().float().cpu().numpy()
                for i, value in zip(selected, values):
                    runner._lf3r_latents.setdefault(req_ids[i], {})[int(indices[i])] = value.copy()
        return original(hidden_states, *args, **kwargs)

    runner.model.compute_logits = compute_logits


def start_capture(worker):
    worker.model_runner._lf3r_latents = {}


def finish_capture(worker):
    saved = worker.model_runner._lf3r_latents
    worker.model_runner._lf3r_latents = None
    # Untyped UtilityResult payloads do not reconstruct ndarray types under
    # vLLM's default safe Msgpack transport. Return only native containers.
    return {request_id: {index: value.tolist() for index, value in tokens.items()}
            for request_id, tokens in saved.items()}


class LatentWorkerExtension:
    """Named worker RPCs, importable without serializing Python callables."""

    def lf3r_install_latent_capture(self):
        install_capture(self)

    def lf3r_start_latent_capture(self):
        start_capture(self)

    def lf3r_finish_latent_capture(self):
        return finish_capture(self)


def score_token_index(tokenizer, token_ids, position="score_start"):
    marker = "</score>" if position == "score_end" else "<score>"
    for i in range(len(token_ids)):
        if marker in tokenizer.decode(token_ids[:i + 1], skip_special_tokens=False):
            index = i + 1 if position == "score_next" else i
            if index >= len(token_ids):
                raise ValueError("No generated token after <score>")
            return index
    raise ValueError(f"Generation did not complete {marker}; no latent emitted")


def select_features(outputs, workers, tokenizer, position):
    features = []
    token_indices = []
    for output in outputs:
        index = score_token_index(tokenizer, output.outputs[0].token_ids, position)
        copies = [w.get(output.request_id, {}).get(index) for w in workers]
        copies = [value for value in copies if value is not None]
        if not copies:
            raise ValueError("Score boundary token was not decoded (e.g. max_tokens truncation); rerun required")
        value = np.asarray(copies[0], dtype=np.float32)
        if value.shape != (2560,) or not np.isfinite(value).all():
            raise ValueError("Expected a finite 2560D final decoder hidden state")
        features.append(value)
        token_indices.append(index)
    return features, token_indices


def save_features(path, rows, features, token_indices, position):
    frames = []
    before = []
    for row in rows:
        match = re.search(r"af_(\d+)$", row["id"])
        bf = re.search(r"bf_(\d+)-af_", row["id"])
        if match is None or bf is None:
            raise ValueError("Incremental sample ID has no before/after frame index")
        frames.append(int(match.group(1)))
        before.append(int(bf.group(1)))
    if len(features) != len(rows) or not features:
        raise ValueError("Latent extraction produced an incomplete or empty sample set")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, features=np.stack(features),
            sample_ids=np.asarray([row["id"] for row in rows]),
            sample_indices=np.arange(len(rows)), frame_indices=np.asarray(frames),
            before_frame_indices=np.asarray(before), token_indices=np.asarray(token_indices),
            mode=np.asarray("incremental"), position=np.asarray(position),
            schema_version=np.asarray(1))
    temporary.replace(path)
