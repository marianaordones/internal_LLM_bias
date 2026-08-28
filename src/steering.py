"""Loading and applying TalkTuner controlling-probe directions."""

from pathlib import Path

from src.config import ATTRIBUTE_CONFIG


def resolve_controlling_probe_dir(path):
    """Accept both normal and double-nested checkpoint layouts."""
    path = Path(path)
    candidates = [path, path / "controlling_probe"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*_probe_at_layer_*.pth")):
            return candidate
    raise FileNotFoundError(
        f"No control probe checkpoints found in {path} or {path / 'controlling_probe'}"
    )


def load_steer_vectors(probe_dir, attribute, from_idx, to_idx, device):
    import torch

    config = ATTRIBUTE_CONFIG[attribute]
    vectors = {}
    for layer in range(from_idx, to_idx):
        # TalkTuner checkpoint numbering is one ahead of model decoder indices.
        checkpoint = probe_dir / f"{config['probe_name']}_probe_at_layer_{layer + 1}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        weights = state["proj.0.weight"].to(torch.float32)
        expected = len(config["classes"])
        if weights.shape[0] != expected:
            raise ValueError(
                f"{checkpoint.name} has {weights.shape[0]} classes; expected {expected}."
            )
        vectors[layer] = [weights[i].to(device) for i in range(expected)]
    return vectors


def make_hook(vector):
    import torch

    def hook(module, inputs, output):
        # Llama decoder layers return hidden states as the first tuple item.
        hidden = output[0]
        hidden[:, -1, :] = (
            hidden[:, -1, :].to(torch.float32) + vector
        ).to(hidden.dtype)
        return output

    return hook


def option_distribution(model, input_ids, candidate_ids):
    import torch

    with torch.inference_mode():
        logits = model(input_ids).logits[0, -1]
    candidates = torch.tensor(candidate_ids, device=input_ids.device)
    return torch.softmax(logits[candidates].float(), dim=-1).cpu().numpy()


def steered_distribution(model, input_ids, candidate_ids, vectors, class_idx, magnitude):
    # Magnitude zero is the unmodified model baseline.
    if magnitude == 0:
        return option_distribution(model, input_ids, candidate_ids)

    handles = []
    try:
        # Hooks are active for exactly one forward pass and are always removed.
        for layer, class_vectors in vectors.items():
            vector = magnitude * class_vectors[class_idx]
            handles.append(
                model.model.layers[layer].register_forward_hook(make_hook(vector))
            )
        return option_distribution(model, input_ids, candidate_ids)
    finally:
        for handle in handles:
            handle.remove()
