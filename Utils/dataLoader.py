import os
import json
import numpy as np
import random
import csv
import warnings

from PIL import Image
import torch
from torch.utils.data import Dataset

from Utils.dataTransformation import transformations


class CholecT50Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        recognition_length: int = 16,
        anticipation_length: int = 0,
        mode: str = 'test'
    ):
        """
        Dataset for CholecT50 surgical videos.

        Args:
            root_dir (str): Path to the CholecT50 dataset.
            recognition_length (int): Number of frames per sample.
            anticipation_length (int): Number of future frames to anticipate.
            mode (str): One of 'train', 'val', 'test'.
        """
        self.root_dir = root_dir
        self.recognition_length = recognition_length
        self.anticipation_length = anticipation_length
        self.mode = mode

        # Validate inputs
        if self.recognition_length < 1:
            warnings.warn("recognition_length must be >= 1 for the model to function.")
        if self.mode not in ('train', 'val', 'test'):
            warnings.warn("mode should be one of 'train', 'val', or 'test'.")

        self.transform = transformations(self.mode)

        self.video_dir = os.path.join(root_dir, 'videos')
        self.label_dir = os.path.join(root_dir, 'labels')

        video_names = sorted(
            v for v in os.listdir(self.video_dir)
            if os.path.isdir(os.path.join(self.video_dir, v))
        )
        self.video_names = video_names

        self.samples = self._load_samples()

        self._compute_pos_weights()

    def _load_samples(self):
        """
        Load frame sequences and labels as samples using a sliding window.
        Shuffle and split samples by mode at the sample level (not video-level).
        """
        samples = []

        for video_name in self.video_names:
            video_path = os.path.join(self.video_dir, video_name)
            label_path = os.path.join(self.label_dir, f"{video_name}.json")

            if not os.path.isdir(video_path) or not os.path.exists(label_path):
                continue

            frame_files = sorted(f for f in os.listdir(video_path) if f.endswith('.png'))
            frame_paths = [os.path.join(video_path, f) for f in frame_files]

            with open(label_path, 'r') as f:
                labels = json.load(f)["annotations"]

            if isinstance(labels, dict):
                labels = list(labels.values())

            # Extract IVT triplets and convert to individual labels
            labels = [[item[0] for item in group] for group in labels]
            labels = [[[x] for x in item] for item in labels]
            labels = self._triplet_to_indiv_label(labels)

            # Determine sliding window range
            end_limit = min(len(frame_paths), len(labels)) - (self.recognition_length + self.anticipation_length) + 1
            if end_limit < 0:
                continue  # skip videos too short

            # Sliding window samples
            for start in range(end_limit):
                frames = frame_paths[start : start + self.recognition_length]
                lbls = labels[start : start + self.recognition_length + self.anticipation_length]
                samples.append((frames, lbls))

        random.shuffle(samples)

        # Sample-level split rationale:
        # - Model only sees short clips, no long-term context.
        # - Shuffling breaks temporal label repetition bias.
        # - Better class balance possible than video-level splits.
        # - Seeded randomness ensures reproducibility.
        num_samples = len(samples)
        train_end = int(0.7 * num_samples)
        val_end = int(0.9 * num_samples)

        if self.mode == 'train':
            samples = samples[:train_end]
        elif self.mode == 'val':
            samples = samples[train_end:val_end]
        elif self.mode == 'test':
            samples = samples[val_end:]

        return samples

    def __len__(self):
        return len(self.samples)

    def _triplets_to_indiv_labels(self, nested_ivt_list, label_map):
        """
        Convert triplet labels (IVT) to individual labels (instrument, verb, target).
        """
        for i, outer in enumerate(nested_ivt_list):
            for j, inner in enumerate(outer):
                triplet = label_map[inner[0]]
                nested_ivt_list[i][j] = [
                    triplet['i'],
                    triplet['v'],
                    triplet['t'],
                    inner[0]
                ]
        return nested_ivt_list

    def _triplet_to_indiv_label(self, nested_ivt_list):
        label_map = {}
        mapping_path = os.path.join(self.root_dir, 'label_mapping.txt')
        with open(mapping_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ivt_code = int(row['# IVT'])
                label_map[ivt_code] = {
                    'i': int(row[' I']),
                    'v': int(row[' V']),
                    't': int(row[' T']),
                }
        return self._triplets_to_indiv_labels(nested_ivt_list, label_map)

    def _compute_pos_weights(self):
        """
        Compute positive class weights for imbalance handling.
        """
        instrument_counts = torch.zeros(6)
        verb_counts = torch.zeros(10)
        target_counts = torch.zeros(15)
        ivt_counts = torch.zeros(100)
        total_frames = 0

        for _, label_seq in self.samples:
            i_hot, v_hot, t_hot, ivt_hot = self._one_hot_encode_label(label_seq)
            instrument_counts += torch.tensor(i_hot, dtype=torch.float).sum(dim=0)
            verb_counts += torch.tensor(v_hot, dtype=torch.float).sum(dim=0)
            target_counts += torch.tensor(t_hot, dtype=torch.float).sum(dim=0)
            ivt_counts += torch.tensor(ivt_hot, dtype=torch.float).sum(dim=0)
            total_frames += i_hot.shape[0]

        eps = 1e-6
        raw = {
            'i': (total_frames - instrument_counts) / (instrument_counts + eps),
            'v': (total_frames - verb_counts) / (verb_counts + eps),
            't': (total_frames - target_counts) / (target_counts + eps),
            'ivt': (total_frames - ivt_counts) / (ivt_counts + eps),
        }
        MAX_WEIGHT = 50.0
        self.pos_weights = {k: torch.clamp(w, max=MAX_WEIGHT) for k, w in raw.items()}

    def _one_hot_encode_label(
        self,
        label_triplets,
        i_classes=6,
        v_classes=10,
        t_classes=15,
        ivt_classes=100
    ):
        num_frames = len(label_triplets)

        instrument_onehots = np.zeros((num_frames, i_classes), dtype=int)
        verb_onehots = np.zeros((num_frames, v_classes), dtype=int)
        target_onehots = np.zeros((num_frames, t_classes), dtype=int)
        ivt_onehots = np.zeros((num_frames, ivt_classes), dtype=int)

        for f_idx, label_group in enumerate(label_triplets):
            for label in label_group:
                i, v, t, ivt = label
                instrument_onehots[f_idx, i] = 1
                verb_onehots[f_idx, v] = 1
                target_onehots[f_idx, t] = 1
                ivt_onehots[f_idx, ivt] = 1

        return instrument_onehots, verb_onehots, target_onehots, ivt_onehots

    def __getitem__(self, idx):
        frame_paths, label_seq = self.samples[idx]

        # One-hot encode labels
        instrument_onehot, verb_onehot, target_onehot, ivt_onehot = self._one_hot_encode_label(label_seq)

        # Convert to tensors
        instrument_onehot = torch.tensor(instrument_onehot, dtype=torch.float)  # (T, 6)
        verb_onehot = torch.tensor(verb_onehot, dtype=torch.float)            # (T, 10)
        target_onehot = torch.tensor(target_onehot, dtype=torch.float)        # (T, 15)
        ivt_onehot = torch.tensor(ivt_onehot, dtype=torch.float)              # (T, 100)

        # Load and transform frames
        frames = [self.transform(Image.open(p).convert('RGB')) for p in frame_paths]
        frames = torch.stack(frames)  # (T, C, H, W)

        return frames, instrument_onehot, verb_onehot, target_onehot, ivt_onehot


def main():
    """Basic test to verify dataset loading and batching."""
    dataset = CholecT50Dataset('Data', recognition_length=9, anticipation_length=1, mode='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for frames, i, v, t, ivt in loader:
        print(frames.shape)  # [B, T, C, H, W]
        print(i.shape)       # [B, T, 6]
        break


if __name__ == '__main__':
    main()
