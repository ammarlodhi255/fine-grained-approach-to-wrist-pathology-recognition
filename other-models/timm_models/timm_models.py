import argparse
from pathlib import Path

import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torch
import torchmetrics
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from pytorch_accelerated.callbacks import SaveBestModelCallback
from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS


def create_datasets(image_size, data_mean, data_std, train_path, val_path):
    train_transforms = timm.data.create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        # auto_augment="rand-m7-mstd0.5-inc1",
    )

    eval_transforms = timm.data.create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )

    train_dataset = timm.data.dataset.ImageDataset(
        train_path, transform=train_transforms
    )
    eval_dataset = timm.data.dataset.ImageDataset(val_path, transform=eval_transforms)

    test_dataset = timm.data.dataset.ImageDataset("./Split/test", transform=eval_transforms) ##
    return train_dataset, eval_dataset, test_dataset


class TimmMixupTrainer(Trainer):
    def __init__(self, eval_loss_fn, mixup_args, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = timm.data.Mixup(**mixup_args)

        # self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        torchmetrics_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=4)
        self.accuracy = torchmetrics_accuracy
        self.ema_accuracy = torchmetrics_accuracy
        self.ema_model = None

    def create_scheduler(self):
        return timm.scheduler.CosineLRScheduler(
            self.optimizer,
            t_initial=self.run_config.num_epochs,
            cycle_decay=0.5,
            lr_min=1e-6,
            t_in_epochs=True,
            warmup_t=3,
            warmup_lr_init=1e-4,
            cycle_limit=1,
        )

    def training_run_start(self):
        # Model EMA requires the model without a DDP wrapper and before sync batchnorm conversion
        self.ema_model = timm.utils.ModelEmaV2(
            self._accelerator.unwrap_model(self.model), decay=0.9
        )
        if self.run_config.is_distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

    def train_epoch_end(
        self,
    ):
        self.ema_model.update(self.model)
        self.ema_model.eval()

        if hasattr(self.optimizer, "sync_lookahead"):
            self.optimizer.sync_lookahead()

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            xb, yb = batch
            outputs = self.model(xb)
            val_loss = self.eval_loss_fn(outputs, yb)
            self.accuracy.update(outputs.argmax(-1), yb)

            ema_model_preds = self.ema_model.module(xb).argmax(-1)
            self.ema_accuracy.update(ema_model_preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()

        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.run_history.update_metric(
            "ema_model_accuracy", self.ema_accuracy.compute().cpu()
        )
        self.accuracy.reset()
        self.ema_accuracy.reset()



def test_model(model, test_dataset, loss_fn, num_classes):

    # Load the saved model's state dictionary
    checkpoint = torch.load("./best_model.pt")

    # Load the state dictionary onto the model
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    test_loss = 0.0
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Adjust batch_size as needed

    with torch.no_grad():
        for batch in test_loader:
            xb, yb = batch
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            test_loss += loss.item()
            accuracy.update(outputs.argmax(-1), yb)
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy.compute().cpu()
    accuracy.reset()
    
    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Assuming you have a test DataLoader named 'test_loader' and 'validate_loss_fn' defined previously
# You can call the function like this after training:
# test_model(model, test_loader, validate_loss_fn, num_classes)


def main(data_path, is_train):

    # Set training arguments, hardcoded here for clarity
    image_size = (384, 384)
    lr = 5e-3
    smoothing = 0.1
    mixup = 0.2
    cutmix = 1.0
    batch_size = 64
    bce_target_thresh = 0.2
    num_epochs = 100
    data_path = Path(data_path)
    train_path = data_path / "train"
    val_path = data_path / "val"
    num_classes = len(list(train_path.iterdir()))

    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes,
    )

    # Create model using timm  
    model = timm.create_model(
        "resnet101", pretrained=True, num_classes=num_classes, drop_path_rate=0.05)   


    # Load data config associated with the model to use in data augmentation pipeline
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config["mean"]
    data_std = data_config["std"]

    # Create training and validation datasets
    train_dataset, eval_dataset, test_dataset = create_datasets(
        train_path=train_path,
        val_path=val_path,
        image_size=image_size,
        data_mean=data_mean,
        data_std=data_std,
    )

    # Create optimizer
    optimizer = timm.optim.create_optimizer_v2(
        model, opt="lookahead_AdamW", lr=lr, weight_decay=0.01
    )

    # As we are using Mixup, we can use BCE during training and CE for evaluation
    train_loss_fn = timm.loss.BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    # Create trainer and start training

    if is_train:
        trainer = TimmMixupTrainer(
            model=model,
            optimizer=optimizer,
            loss_func=train_loss_fn,
            eval_loss_fn=validate_loss_fn,
            mixup_args=mixup_args,
            num_classes=num_classes,
            callbacks=[
                *DEFAULT_CALLBACKS,
                SaveBestModelCallback(watch_metric="accuracy", greater_is_better=True),
            ],
        )

        trainer.train(
            per_device_batch_size=batch_size,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=num_epochs,
            create_scheduler_fn=trainer.create_scheduler,
        )
    else:
        test_model(model, test_dataset, validate_loss_fn, 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    parser.add_argument("--train", required=True, help="Whether to do training")
    args = parser.parse_args()
    main(args.data_dir, args.train)
