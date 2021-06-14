import torch


def save_model(path, model):
    torch.save({
        'model_state_dict': model.state_dict()
    }, path)


def load_model(model, path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
