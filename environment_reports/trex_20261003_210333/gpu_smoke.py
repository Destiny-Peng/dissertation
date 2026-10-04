import json
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from qwen_vla import Qwen3VLVLAModel
from tactile_vqvae.models.tactile_vqvae import TactileVQVAE, TactileVQVAEConfig

torch.manual_seed(0)
assert torch.cuda.is_available()
device = torch.device("cuda:0")
config = Qwen3VLTextConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=128)
config.attention_bias = False
vq_config = TactileVQVAEConfig(granularity="finger", hidden_channels=16,
    bottleneck_channels=32, embed_dim=32, codebook_size=64)
model = Qwen3VLVLAModel(config, action_dim=62, action_chunk=2, tacf6_dim=60,
    use_tactile_vqvae=True, vqvae_config=vq_config.to_dict()).to(device=device, dtype=torch.bfloat16).eval()
with torch.inference_mode():
    codes = model.encode_tactile_f6_history(torch.rand(1, 16, 10, 6, device=device))
    assert codes.shape == (1, 10)
    assert ((codes >= 0) & (codes < 64)).all()
    latent = torch.randn(1, 4, 256, device=device, dtype=torch.bfloat16)
    actions = model.x_embedder(torch.randn(1, 2, 62, device=device, dtype=torch.bfloat16))
    tactile = model.tactile_code_embedder(codes)
    inputs = torch.cat([latent, actions, tactile], dim=1)
    position_ids = torch.arange(4, device=device).view(1, 1, 4).expand(3, 1, 4)
    output = model.model(inputs_embeds=inputs, position_ids=position_ids,
        latent_indexes=torch.arange(4, device=device),
        action_indexes=torch.arange(4, 6, device=device),
        tactile_indexes=torch.arange(6, 16, device=device), use_cache=True)
    assert output.last_hidden_state.shape == (1, 16, 256)
    assert torch.isfinite(output.last_hidden_state).all()
    assert output.past_key_values.get_seq_length() == 16
    velocity = model.final_layer(output.last_hidden_state[:, 4:6])
    assert velocity.shape == (1, 2, 62) and torch.isfinite(velocity).all()
    vq = TactileVQVAE(vq_config).to(device).eval()
    recon = vq(torch.rand(1, 16, 5, 6, device=device))
    assert recon["recon"].shape == (1, 16, 5, 6)
    assert torch.isfinite(recon["total_loss"])
    torch.cuda.synchronize()
print(json.dumps({"status": "PASS", "torch": torch.__version__, "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(), "tactile_codes_shape": list(codes.shape),
    "mot_output_shape": list(output.last_hidden_state.shape), "action_velocity_shape": list(velocity.shape),
    "vqvae_reconstruction_shape": list(recon["recon"].shape),
    "peak_allocated_MiB": round(torch.cuda.max_memory_allocated()/2**20, 2),
    "scope": "Small randomly initialized official model; no pretrained checkpoint or robot rollout"}, indent=2))
