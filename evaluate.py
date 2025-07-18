import os
import torch
import time
from pathlib import Path

from models import model_provider
import Utils.ivtmetrics.recognition as ivt_metrics
from Utils.dataLoader import CholecT50Dataset
from Utils.trainer import Trainer
from Utils.logger import getLogger
from Utils.reproducabiltiy import seed_everything


def main():
    # === Load checkpoint and config from inside ===
    weights_path = "weights/McDecoder/0717-2129/best_model_epoch_1.pth"  # or wherever your model is

    assert os.path.isfile(weights_path), f"Checkpoint not found at {weights_path}"
    checkpoint = torch.load(weights_path, map_location='cpu')  # safer to start on CPU
    config = checkpoint['config']
    model_state = checkpoint['model_state_dict']
    model_config = config['model']
    train_config = config['train']

    # === Extract training config values ===
    gpu_usg = ",".join(map(str, train_config['gpu']))
    recognition_length = train_config['recognitionLength']
    anticipation_length = train_config['anticipationLength']
    batch_size = train_config['batchSize']
    m1 = train_config['multiChannelLoss1']
    m2 = train_config['multiChannelLoss2']
    workers = train_config['workNumber']

    print(train_config['randomSeed'])

    # === Set random seed ===
    seed_everything(train_config['randomSeed'])

    # === Set device ===
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")

    # === Logging ===
    time_now = time.strftime("%m%d-%H%M", time.localtime())
    Path("eval_logs").mkdir(exist_ok=True)
    logger = getLogger(f"eval_logs/eval_{time_now}.log")

    # === Load test dataset ===
    test_dataset = CholecT50Dataset(
        'Data',
        recognition_length=recognition_length,
        anticipation_length=anticipation_length,
        mode='test'
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False,
        drop_last=False
    )

    # === Build model and load weights ===
    model = model_provider(
        recognition_length=recognition_length,
        anticipation_length=anticipation_length,
        batch_size=batch_size,
        **model_config
    )
    model = model.to(device)
    model.load_state_dict(model_state)

    # === Set to eval mode ===
    model.eval()

    # === Metrics ===
    recognise = ivt_metrics.Recognition(num_class=100)
    anticipate = ivt_metrics.Recognition(num_class=100)

    # === Run Evaluation ===
    trainer = Trainer(model, test_dataset, m1, m2, recognition_length, anticipation_length, use_gpu=use_gpu)
    test_loss_total, test_loss_recog, test_loss_anticp = trainer.test_model(test_loader, recognise, anticipate)

    # === Normalize loss ===
    dataset_len = len(test_dataset)
    test_loss_total /= dataset_len
    test_loss_recog /= dataset_len
    test_loss_anticp /= dataset_len

    # === Compute and Log mAPs ===
    def log_ap(metric, prefix):
        logger.info(f"{prefix} mAP I: {metric.compute_AP('i')['mAP']:.4f}")
        logger.info(f"{prefix} mAP V: {metric.compute_AP('v')['mAP']:.4f}")
        logger.info(f"{prefix} mAP T: {metric.compute_AP('t')['mAP']:.4f}")
        logger.info(f"{prefix} mAP IT: {metric.compute_AP('it')['mAP']:.4f}")
        logger.info(f"{prefix} mAP IV: {metric.compute_AP('iv')['mAP']:.4f}")
        logger.info(f"{prefix} mAP IVT: {metric.compute_AP('ivt')['mAP']:.4f}")

    logger.info(f"Test Loss Total: {test_loss_total:.4f}")
    logger.info(f"Test Loss Recognition: {test_loss_recog:.4f}")
    logger.info(f"Test Loss Anticipation: {test_loss_anticp:.4f}")

    log_ap(recognise, "Test Recognition")
    log_ap(anticipate, "Test Anticipation")

    print("Evaluation complete.")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
