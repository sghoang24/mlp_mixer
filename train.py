"""Train script."""

import argparse
import wandb
import torch
import torch.nn as nn

from src.data_loader import *
from src.mlp_mixer import *
from utils.utils import *


def train_model(args, model, **kwargs):
    """Train model."""
    print("Start training...")
    if not args.logger:
        if not args.id_name:
            wandb.init(id=args.id_name, project='MLP-Mixer', resume="must")
        else:
            wandb.init(project='MLP-Mixer', name=args.model)

        wandb.watch(model)

    # Train
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(args.device)
            label = label.squeeze().type(torch.LongTensor).to(args.device)
            output = model(data)

            loss = criterion(output, label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # Calculate accuracy
            acc = (output.argmax(dim=1) == label).float().mean()

            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            torch.cuda.empty_cache()

        train_accuracy.append(epoch_accuracy.item())
        train_losses.append(epoch_loss.item())

        # Log
        print(
            "Epoch : {}, train accuracy : {}, train loss : {}".format(
                epoch + 1, epoch_accuracy, epoch_loss
            )
        )

        if not args.logger:
            wandb.log({'Train Accuracy': epoch_accuracy.item(), 'Train Loss': epoch_loss.item(), 'Epoch': epoch + 1})

        # Validation
        with torch.no_grad():
            epoch_val_accuracy, epoch_val_loss = 0, 0
            for data, label in val_loader:
                data = data.to(args.device)
                label = label.squeeze().type(torch.LongTensor).to(args.device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (output.argmax(dim=1) == label).float().mean()
                
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += loss / len(val_loader)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        val_accuracy.append(epoch_val_accuracy.item())
        val_losses.append(epoch_val_loss.item())

        # Log
        print(
            "Epoch : {}, val_accuracy : {}, val_loss : {}".format(
                epoch + 1, epoch_val_accuracy, epoch_val_loss
            )
        )
        if not args.logger:
            wandb.log({'Val Accuracy': epoch_accuracy.item(), 'Val Loss': epoch_loss.item(), 'Epoch': epoch + 1})

        if save_interval is not None:
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(model, optimizer, epoch+1, loss, args)

    if args.logger is not None:
        wandb.finish()
        
    save_model(args, model)
    print('Save model successfully...')

def main(args):
    """Main."""
    model = mixer(**vars(args)).to(args.device)
    train_transforms, val_transforms = get_data_transform(image_size=args.image_size)
    train_loader, val_loader = get_data_loader(
        train_data=train_data,
        valid_data=valid_data,
        batch_size=args.batch_size,
    )
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    running_loss_train, running_loss_val = 0, 0
    running_accuracy_train, running_accuracy_val = 0, 0

    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []

    model, optimizer, start_epoch = load_model(
        model=model,
        optimizer=optimizer,
        checkpoint_path=args.resume
    )

    train_model(
        args,
        model,
        optimizer=optimizer,
        criterion=criterion,
        start_epoch=start_epoch,
        epochs=args.epochs,
        save_interval=args.save_interval,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == '__main__':
    home_dir = '/content'
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument('--logger', default=None, help='Logger.')
    parser.add_argument("--id-name", default=None, type=str)
    parser.add_argument("--train-folder", default='{}/data/train'.format(home_dir), type=str)
    parser.add_argument("--valid-folder", default='{}/data/valid'.format(home_dir), type=str)
    parser.add_argument("--model-folder", default='{}/model'.format(home_dir), type=str)
    parser.add_argument("--resume", default=None, type=str, help='weight to resume training')
    parser.add_argument("--save-interval", default=None, type=int)
    parser.add_argument("--model", default=None, type=str, help='s_16, s_32, b_16, b_32, l_16, l_32, h_14, custom')
    parser.add_argument("--num-classes", default=5, type=int)
    parser.add_argument("--num-mlp-blocks", default=8, type=int)
    parser.add_argument("--patch-size", default=16, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int, help='Projection units')
    parser.add_argument("--tokens-mlp-dim", default=256, type=int, help='Token-mixing units')
    parser.add_argument("--channels-mlp-dim", default=2048, type=int, help='Channel-mixing units')
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--device", default='cuda', type=str, help='cuda, cpu')

    args = parser.parse_args()
    main(args)
