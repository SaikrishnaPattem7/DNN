import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import os


# Custom Dataset

class StoryDataset(Dataset):
    """
    Returns sequences of images and corresponding tokenized story texts.
    """

    def __init__(self, hf_dataset, tokenizer, transform, seq_len=4):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.seq_len = seq_len
        self.needed_len = seq_len + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        images_list = item["images"]
        texts_list = item["story"]

        if isinstance(texts_list, str):
            texts_list = [texts_list]

        # Pad images
        if len(images_list) < self.needed_len:
            images_list += [images_list[-1]] * (self.needed_len - len(images_list))

        # Pad text
        if len(texts_list) < self.needed_len:
            texts_list += [texts_list[-1]] * (self.needed_len - len(texts_list))

        imgs = images_list[-self.needed_len:]
        txts = texts_list[-self.needed_len:]

        processed_imgs = []
        for im in imgs:
            if im.mode != "RGB":
                im = im.convert("RGB")
            processed_imgs.append(self.transform(im))

        img_tensor = torch.stack(processed_imgs)

        encoded = self.tokenizer(
            txts,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        return {
            "input_images": img_tensor[:-1],
            "input_ids": encoded["input_ids"][:-1],
            "attention_mask": encoded["attention_mask"][:-1],
            "target_image": img_tensor[-1],
            "target_ids": encoded["input_ids"][-1]
        }


# Transforms & Tokenizer

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def load_storyreasoning_dataset_streaming():
    """
    Load StoryReasoning dataset using streaming to avoid disk usage.
    """
    return load_dataset(
        "daniel3303/StoryReasoning",
        split="train",
        streaming=True
    )


# Visualization Helpers

def save_image_tensor(img_tensor, filename, folder):
    os.makedirs(folder, exist_ok=True)

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.axis("off")

    path = os.path.join(folder, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved image → {path}")

def save_sample_table(raw_dataset, filename, folder, num_rows=5):
    os.makedirs(folder, exist_ok=True)

    rows = []
    for i, item in enumerate(raw_dataset):
        if i >= num_rows:
            break

        rows.append({
            "num_images": len(item["images"]),
            "num_sentences": len(item["story"]),
            "first_sentence": item["story"][0]
        })

    df = pd.DataFrame(rows)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)

    print(f"[OK] Saved dataset table → {path}")

def plot_dataset_distribution(raw_dataset, folder):
    os.makedirs(folder, exist_ok=True)

    img_counts = []
    story_counts = []

    for item in raw_dataset:
        img_counts.append(len(item["images"]))
        story_counts.append(len(item["story"]))

    # Images distribution
    plt.figure()
    plt.hist(img_counts, bins=10)
    plt.title("Images per Sample")
    plt.xlabel("Number of Images")
    plt.ylabel("Frequency")
    path1 = os.path.join(folder, "images_per_sample.png")
    plt.savefig(path1)
    plt.close()

    print(f"[OK] Saved plot → {path1}")

    # Story distribution
    plt.figure()
    plt.hist(story_counts, bins=10)
    plt.title("Sentences per Story")
    plt.xlabel("Number of Sentences")
    plt.ylabel("Frequency")
    path2 = os.path.join(folder, "story_per_sample.png")
    plt.savefig(path2)
    plt.close()

    print(f"[OK] Saved plot → {path2}")


# MAIN DEMO (RUN THIS FILE DIRECTLY)

if __name__ == "__main__":

    OUTPUT_DIR = "results/figures_tables"


    print("Running dataset visualization demo...")
  

    print("[INFO] Loading dataset (STREAMING mode)")
    dataset_stream = load_storyreasoning_dataset_streaming()

    # Take only first 10 samples (NO DISK)
    raw_samples = list(dataset_stream.take(10))
    print(f"[INFO] Loaded {len(raw_samples)} samples")

    tokenizer = get_tokenizer()
    transform = get_image_transform()
    dataset = StoryDataset(raw_samples, tokenizer, transform)

    print("[INFO] Saving sample images...")
    for i in range(3):
        sample = dataset[i]
        save_image_tensor(
            sample["input_images"][0],
            f"sample_image_{i}.png",
            OUTPUT_DIR
        )

    print("[INFO] Saving dataset table...")
    save_sample_table(raw_samples, "sample_table.csv", OUTPUT_DIR)

    print("[INFO] Creating distribution plots...")
    plot_dataset_distribution(raw_samples, OUTPUT_DIR)

    print("All figures & tables saved to:")
    print("results/figures_tables/")
 
