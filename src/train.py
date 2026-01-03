import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from utils import (
    StoryDataset,
    get_image_transform,
    get_tokenizer,
    load_storyreasoning_dataset_streaming
)


# Configuration

BATCH_SIZE = 2
RESULTS_DIR = "results/figures_tables"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Main Script

def main():

    print("[INFO] Loading StoryReasoning dataset (streaming)...")

    # IMPORTANT: streaming loader takes NO arguments
    dataset_stream = load_storyreasoning_dataset_streaming()

    # Limit samples manually to avoid disk issues
    dataset_raw = []
    for i, item in enumerate(dataset_stream):
        if i >= 10:
            break
        dataset_raw.append(item)

    print(f"[OK] Loaded {len(dataset_raw)} streamed samples")

    tokenizer = get_tokenizer()
    transform = get_image_transform()

    dataset = StoryDataset(dataset_raw, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    image_seq_lengths = []
    text_seq_lengths = []

    print("[INFO] Iterating through dataset...\n")

    for i, batch in enumerate(dataloader):
        img_len = batch["input_images"].shape[1]
        txt_len = batch["input_ids"].shape[1]

        image_seq_lengths.append(img_len)
        text_seq_lengths.append(txt_len)

        print(
            f"[BATCH {i}] "
            f"Images shape: {batch['input_images'].shape} | "
            f"Text shape: {batch['input_ids'].shape}"
        )

    save_sequence_plot(image_seq_lengths, text_seq_lengths)

    print("\n[DONE] train.py executed successfully")
    print(f"Results saved to → {RESULTS_DIR}")


# Plot Function

def save_sequence_plot(img_lens, txt_lens):

    plt.figure()
    plt.plot(img_lens, label="Image sequence length")
    plt.plot(txt_lens, label="Text sequence length")
    plt.xlabel("Batch index")
    plt.ylabel("Sequence length")
    plt.title("Input Sequence Length Statistics")
    plt.legend()

    path = os.path.join(RESULTS_DIR, "input_sequence_statistics.png")
    plt.savefig(path)
    plt.close()

    print(f"[OK] Saved plot {path}")


# Entry Point

if __name__ == "__main__":
    main()
