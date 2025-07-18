import os
import sys
import yaml
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import model_provider
from Utils.dataLoader import CholecT50Dataset
from Utils.trainer import Trainer
from Utils.logger import getLogger
import Utils.ivtmetrics.recognition as ivt_metrics
from Utils.reproducabiltiy import seed_everything

# ------------------------------ Load Config ------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Unpack training config
train_config = config['train']
model_config = config['model']

task_used = train_config['task']
gpu_ids = ",".join(map(str, train_config['gpu']))
recognition_length = train_config['recognitionLength']
anticipation_length = train_config['anticipationLength']
batch_size = train_config['batchSize']
optimizer_choice = train_config['optimiserUsed']
multi_optim = train_config['multi']
epochs = train_config['epochs']
num_workers = train_config['workNumber']
lr = train_config['learningRate']
momentum = train_config['momentum']
weight_decay = train_config['weightdecay']
dampening = train_config['dampening']
nesterov = train_config['nesterov']
m1 = train_config['multiChannelLoss1']
m2 = train_config['multiChannelLoss2']
freeze_net = train_config['freeze']

model_name = model_config['name']

# ------------------------------ Set seed ------------------------------
seed_everything(train_config['randomSeed'])

# ------------------------------ Metrics ------------------------------
recognise = ivt_metrics.Recognition(num_class=100)
anticipate = ivt_metrics.Recognition(num_class=100)

# ------------------------------ Training Function ------------------------------
def train_model(train_ds, val_ds, test_ds):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    use_gpu = torch.cuda.is_available()

    if not use_gpu:
        print("CUDA not available.")
        sys.exit(1)
    else:
        print(f"Using GPUs: {gpu_ids}")

    # Output Directories
    timestamp = time.strftime("%m%d-%H%M", time.localtime())
    weights_dir = Path("weights") / model_name / timestamp
    weights_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("training_logs/logger") / f"log_{timestamp}_.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = getLogger(str(log_path))

    tb_path = Path("training_logs/tensorboard_logs") / timestamp
    tb_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_path))

    logger.info(f"Training with m1={m1}, m2={m2}")

    # Dataloaders
    def create_loader(dataset): return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = create_loader(train_ds)
    val_loader = create_loader(val_ds)
    test_loader = create_loader(test_ds)

    # Model setup
    model = model_provider(recognition_length=recognition_length, anticipation_length=anticipation_length, batch_size=batch_size, **model_config)
    if use_gpu:
        model = DataParallel(model).cuda()

    # Optimizer
    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum,
                dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
            )
        else:
            logger.info("Using AdamW optimizer")
            optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        logger.info("Using multi optimizer setup")
        params = [
            {'params': model.module.share.parameters()},
        ]

        if optimizer_choice == 0:
            params += [
                {'params': model.module.lstm.parameters(), 'lr': lr},
                {'params': model.module.fc1.parameters(), 'lr': lr},
            ]
            optimizer = optim.SGD(params, lr=lr / 10, momentum=momentum,
                                  dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        else:
            params += [
                {'params': getattr(model.module, name).parameters(), 'lr': lr}
                for name in [
                    "features2fixed_i", "features2fixed_v", "linear_layer_i",
                    "linear_layer_v", "classifier_1", "classifier_2", "classifier_3",
                    "classifier_4", "fc_h1", "fc_h2"
                ]
            ]
            optimizer = optim.AdamW(params, lr=lr / 10)

    trainer = Trainer(model, train_ds, m1, m2, recognition_length, anticipation_length, use_gpu)

    best_model_wts = deepcopy(model.module.state_dict() if use_gpu else model.state_dict())
    best_epoch, best_val_map = 0, 0.0

    # ------------------------------ Training Loop ------------------------------
    for epoch in range(epochs):
        start = time.time()
        tl_total, tl_recog, tl_antic = trainer.train_model(train_loader, optimizer, recognise, anticipate)

        train_map_recog = recognise.compute_AP('ivt')["mAP"]
        train_map_antic = anticipate.compute_AP('ivt')["mAP"]
        recognise.reset(), anticipate.reset()

        vl_total, vl_recog, vl_antic = trainer.validate_model(val_loader, recognise, anticipate)
        val_map_recog = recognise.compute_AP('ivt')["mAP"]
        val_map_antic = anticipate.compute_AP('ivt')["mAP"]
        recognise.reset(), anticipate.reset()

        # Log losses and metrics
        writer.add_scalars('Loss/Total', {'Train': tl_total / len(train_loader), 'Val': vl_total / len(val_loader)}, epoch)
        writer.add_scalars('Loss/Recognition', {'Train': tl_recog / len(train_loader), 'Val': vl_recog / len(val_loader)}, epoch)
        writer.add_scalars('Loss/Anticipation', {'Train': tl_antic / len(train_loader), 'Val': vl_antic / len(val_loader)}, epoch)

        writer.add_scalars('mAP/Recognition_IVT', {'Train': train_map_recog, 'Val': val_map_recog}, epoch)
        writer.add_scalars('mAP/Anticipation_IVT', {'Train': train_map_antic, 'Val': val_map_antic}, epoch)

        # Log to file
        logger.info(f"[Epoch {epoch+1}/{epochs}] Time: {int(time.time() - start)}s")
        logger.info(f"Train Loss: {tl_total/len(train_loader):.4f} | Val Loss: {vl_total/len(val_loader):.4f}")
        logger.info(f"Val Anticipation mAP (IVT): {val_map_antic:.4f}")

        if val_map_antic > best_val_map:
            best_val_map = val_map_antic
            best_epoch = epoch + 1
            best_model_wts = deepcopy(model.module.state_dict() if use_gpu else model.state_dict())
            logger.info(f"New best model saved at epoch {best_epoch}")

    # Save Best Model and archtecture
    best_path = weights_dir / f"best_model_epoch_{best_epoch}.pth"
    torch.save({
        'model_state_dict': best_model_wts,
        'config': config,   # Save the config dict to recreate the model
        'epoch': best_epoch,
        'best_val_map': best_val_map
    }, best_path)
    logger.info(f"Best Model Saved | Epoch: {best_epoch}, mAP: {best_val_map:.4f}")


    # ------------------------------ Testing ------------------------------
    model.module.load_state_dict(best_model_wts) if use_gpu else model.load_state_dict(best_model_wts)
    test_loss_total, test_loss_recog, test_loss_anticp = trainer.test_model(test_loader, recognise, anticipate)

    test_loss_total /= len(test_loader)
    test_loss_recog /= len(test_loader)
    test_loss_anticp /= len(test_loader)


    # === Compute metrics ===
    test_recog_ap_i = recognise.compute_AP('i')["mAP"]
    test_recog_ap_v = recognise.compute_AP('v')["mAP"]
    test_recog_ap_t = recognise.compute_AP('t')["mAP"]
    test_recog_ap_it = recognise.compute_AP('it')["mAP"]
    test_recog_ap_iv = recognise.compute_AP('iv')["mAP"]
    test_recog_ap_ivt = recognise.compute_AP('ivt')["mAP"]


    test_anticipate_ap_i = anticipate.compute_AP('i')["mAP"]
    test_anticipate_ap_v = anticipate.compute_AP('v')["mAP"]
    test_anticipate_ap_t = anticipate.compute_AP('t')["mAP"]
    test_anticipate_ap_it = anticipate.compute_AP('it')["mAP"]
    test_anticipate_ap_iv = anticipate.compute_AP('iv')["mAP"]
    test_anticipate_ap_ivt = anticipate.compute_AP('ivt')["mAP"]

    # === Log results ===
    logger.info(f"Test Loss Total: {test_loss_total:.4f}")
    logger.info(f"Test Loss Recognition: {test_loss_recog:.4f}")
    logger.info(f"Test Loss Anticipation: {test_loss_anticp:.4f}")

    logger.info(f"Test mAP Recognition I: {test_recog_ap_i:.4f}")
    logger.info(f"Test mAP Recognition V: {test_recog_ap_v:.4f}")
    logger.info(f"Test mAP Recognition T: {test_recog_ap_t:.4f}")
    logger.info(f"Test mAP Recognition IT: {test_recog_ap_it:.4f}")
    logger.info(f"Test mAP Recognition IV: {test_recog_ap_iv:.4f}")
    logger.info(f"Test mAP Recognition IVT: {test_recog_ap_ivt:.4f}")

    logger.info(f"Test mAP Anticipation I: {test_anticipate_ap_i:.4f}")
    logger.info(f"Test mAP Anticipation V: {test_anticipate_ap_v:.4f}")
    logger.info(f"Test mAP Anticipation T: {test_anticipate_ap_t:.4f}")
    logger.info(f"Test mAP Anticipation IT: {test_anticipate_ap_it:.4f}")
    logger.info(f"Test mAP Anticipation IV: {test_anticipate_ap_iv:.4f}")
    logger.info(f"Test mAP Anticipation IVT: {test_anticipate_ap_ivt:.4f}")

    print("Evaluation complete.")

# ------------------------------ Main ------------------------------
def main():
    train_ds = CholecT50Dataset("Data", recognition_length, anticipation_length, mode='train')
    val_ds = CholecT50Dataset("Data", recognition_length, anticipation_length, mode='val')
    test_ds = CholecT50Dataset("Data", recognition_length, anticipation_length, mode='test')
    train_model(train_ds, val_ds, test_ds)

if __name__ == "__main__":
    main()