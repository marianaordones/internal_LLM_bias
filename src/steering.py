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


def load_steer_vectors(
    probe_dir, attribute, from_idx, to_idx, device, *, probe_name=None,
    class_indices=None, expected_num_classes=None, expected_hidden_size=None,
):
    import torch

    config = ATTRIBUTE_CONFIG.get(attribute)
    if config is None and probe_name is None:
        raise ValueError(f"Unknown attribute {attribute!r}; provide probe_name explicitly.")
    checkpoint_prefix = probe_name or config["probe_name"]
    selected_indices = class_indices or list(range(len(config["classes"])))
    num_classes = expected_num_classes or len(config["classes"])
    vectors = {}
    for layer in range(from_idx, to_idx):
        # TalkTuner checkpoint numbering is one ahead of model decoder indices.
        checkpoint = probe_dir / f"{checkpoint_prefix}_probe_at_layer_{layer + 1}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        weights = state["proj.0.weight"].to(torch.float32)
        if weights.shape[0] != num_classes:
            raise ValueError(
                f"{checkpoint.name} has {weights.shape[0]} classes; expected {num_classes}."
            )
        if expected_hidden_size is not None and weights.shape[1] != expected_hidden_size:
            raise ValueError(
                f"{checkpoint.name} has hidden size {weights.shape[1]}; "
                f"the model uses {expected_hidden_size}."
            )
        if any(index < 0 or index >= num_classes for index in selected_indices):
            raise ValueError(f"Invalid class indices {selected_indices} for {checkpoint.name}.")
        vectors[layer] = [weights[index].to(device) for index in selected_indices]
    return vectors


def make_hook(vec):
    import torch
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.dim() == 3:
            h[:, -1, :] = (h[:, -1, :].to(torch.float32) + vec).to(h.dtype)
        elif h.dim() == 2:
            h[-1, :] = (h[-1, :].to(torch.float32) + vec).to(h.dtype)
        else:
            raise RuntimeError(f"formato inesperado: {tuple(h.shape)}")
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
