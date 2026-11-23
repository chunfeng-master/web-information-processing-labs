from pathlib import Path
import torch


def save_checkpoint(model, path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
