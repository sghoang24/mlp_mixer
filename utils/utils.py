"""Utils."""

import os
import random

import torch
from torchvision import transforms


def load_model(model, optimizer, checkpoint_path: str = None):
    """Load model."""
    if checkpoint_path is None:
        # Load from scratch
        return model, optimizer, 0

    # Load from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f"Loaded checkpoint from epoch {epoch}")

    return model, optimizer, epoch


def save_model(args, model):
    """Save model."""
    folder_path = args.model_folder
    model_name = args.model
    i = 1
    new_model_name = model_name
    while os.path.exists(os.path.join(folder_path, f'mixer_{new_model_name}' + ".pt")):
        new_model_name = model_name + "_" + str(i)
        i += 1

    path = os.path.join(folder_path, new_model_name + ".pt")

    torch.save(model, path)

    print(f"Model saved to `{new_model_name}.pt`")


def save_checkpoint(model, optimizer, epoch, loss, args):
    """Save checkpoint."""
    os.makedirs(args.model_folder, exist_ok=True)
    checkpoint_path = os.path.join(args.model_folder, f'mixer_{args.model}' + '_' + str(epoch) + ".pt")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to `{checkpoint_path}`")


def predict_image(image_path, model, args):
    """Predict image."""
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]),
    ])
    class_map = {
        "daisy" : 0, "dandelion": 1,
        "roses" : 2, "sunflowers": 3, "tulips" : 4
    }
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(dim=0).to(args.device)

    with torch.no_grad():
        outputs = model(image_tensor)
        pred_idx = torch.argmax(outputs, dim=1)
    pred_label = pred_idx.item()
    pred_class = [key for key, value in class_map.items() if value == pred_label][0]


def display_image(image_path, predicted_label):
    """Display image."""
    image = Image.open(image_path)
    image.show()
    print(f"Predicted label: {predicted_label}")


def predict_folder(folder_path):
    """Predict folder."""
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_paths)
    selected_images = image_paths[:15]
    rows, cols = 3, 5
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(selected_images):
                image_path = selected_images[index]
                predicted_label = predict_image(image_path)
                display_image(image_path, predicted_label)
                print()
            else:
                break
