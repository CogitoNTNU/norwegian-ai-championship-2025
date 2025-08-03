from datetime import datetime
import logging
import sys
import os
from dotenv import load_dotenv
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)


from monai.visualize import plot_2d_or_3d_image
import wandb
from tumor_dataset import create_tumor_dataset


def main(dataset_dir):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    monai.config.print_config()

    NUM_EPOCHS = 10
    BATCH_SIZE = 2
    LR = 1e-3
    PATCH_SIZE = (96, 96)

    # W&B run initialization
    run = wandb.init(
        project="tumor-segmentation",
        name=f"BasicUnetPlusPlus_{datetime.now():%Y%m%d_%H%M%S}",
        entity="nm-i-ki",
        config=dict(
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            patch_size=PATCH_SIZE,
            architecture="UNet",
            loss="Dice(sigmoid=True)",
            optimizer="Adam",
        ),
        save_code=True,
        sync_tensorboard=True,
        tags=["monai", "segmentation"],
    )

    # Create datasets
    train_ds, val_ds = create_tumor_dataset(dataset_dir=dataset_dir)

    print(f"Amount of images train: {len(train_ds)} val: {len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate
    )

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.BasicUNetPlusPlus(
        spatial_dims=2,
        features=(64, 128, 256, 512, 1024, 128),
        in_channels=4,
        out_channels=4,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), LR)

    # Attach gradients & parameters to W&B
    wandb.watch(model, log="all", log_freq=10)

    # Train model
    print("Starting training...")
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    tensorboard_writer = SummaryWriter()
    best_metric = -1.0
    best_epoch = -1

    os.makedirs("models", exist_ok=True)
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_loader, 1):
            imgs, segs = batch["img"].to(device), batch["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)[0]  # endring her, gjør den om til en tensor
            loss = loss_function(outputs, segs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log per step
            global_step += 1
            wandb.log(
                {"train_loss": loss.item(), "epoch": epoch + 1, "step": global_step}
            )
            tensorboard_writer.add_scalar(
                "train_loss", loss.item(), epoch * len(train_loader) + step
            )
        avg_epoch_loss = epoch_loss / step
        print(f"  Train avg loss: {avg_epoch_loss:.4f}")
        wandb.log({"train_avg_loss": avg_epoch_loss, "epoch": epoch + 1})

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs, val_segs = (
                        val_batch["img"].to(device),
                        val_batch["seg"].to(device),
                    )
                    sw_out = sliding_window_inference(val_imgs, PATCH_SIZE, 4, model)
                    preds = [post_trans(x) for x in decollate_batch(sw_out)]
                    dice_metric(y_pred=preds, y=val_segs)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                tensorboard_writer.add_scalar("val_mean_dice", metric, epoch + 1)
                print(f"  Val mean Dice: {metric:.4f}")

                # Log validation metric & sample images
                wandb.log({"val_mean_dice": metric, "epoch": epoch + 1})

                # Log first channel of first image / pred / label as examples
                img_np = val_imgs[0, 0].cpu().float().numpy()
                pred_np = preds[0][0].cpu().float().numpy()
                label_np = val_segs[0, 0].cpu().float().numpy()
                wandb.log(
                    {
                        "example_input": wandb.Image(img_np, caption="input"),
                        "example_pred": wandb.Image(pred_np, caption="prediction"),
                        "example_label": wandb.Image(label_np, caption="ground truth"),
                        "epoch": epoch + 1,
                    }
                )

                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch + 1
                    best_model_path = f"models/best_model_{metric:.4f}.pth"
                    torch.save(model.state_dict(), best_model_path)
                    run.save(best_model_path)
                    print(
                        f"  Best model saved with Dice {best_metric:.4f} at epoch {best_epoch}"
                    )

                    # Save model to W&B as an artifact
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(best_model_path)
                    run.log_artifact(artifact)

                # Continue logging to TensorBoard if desired
                plot_2d_or_3d_image(
                    val_imgs, epoch + 1, tensorboard_writer, index=0, tag="image"
                )
                plot_2d_or_3d_image(
                    val_segs, epoch + 1, tensorboard_writer, index=0, tag="label"
                )
                plot_2d_or_3d_image(
                    preds, epoch + 1, tensorboard_writer, index=0, tag="output"
                )

    print(f"\nTraining done! Best Dice {best_metric:.4f} reached on epoch {best_epoch}")
    tensorboard_writer.close()
    run.finish()


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if not WANDB_API_KEY:
        raise ValueError(
            "WANDB_API_KEY not found in environment variables. Please set it in .env file."
        )
    print(
        f"WANDB_API_KEY: {WANDB_API_KEY[:4]}..."
    )  # Print only the first 4 characters for security
    main("../data/raw/tumor-segmentation")
