import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel



# Contrastive Loss

def contrastive_loss(text_feat, image_feat, temperature=0.07):
    """
    Symmetric contrastive loss between text and image embeddings
    """
    text_feat = F.normalize(text_feat, dim=1)
    image_feat = F.normalize(image_feat, dim=1)

    sim = torch.matmul(text_feat, image_feat.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)

    loss_t2i = F.cross_entropy(sim, labels)
    loss_i2t = F.cross_entropy(sim.T, labels)

    return 0.5 * (loss_t2i + loss_i2t)



# Visual Encoder

class VisualEncoder(nn.Module):
    """
    ResNet-50 based visual feature encoder
    """

    def __init__(self, embed_dim):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, images):
        """
        images: [B, S, 3, 224, 224]
        """
        B, S, C, H, W = images.shape
        x = images.view(B * S, C, H, W)

        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        feats = self.fc(feats)

        return feats.view(B, S, -1)



# Text Encoder

class TextEncoder(nn.Module):
    """
    DistilBERT based text encoder
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B, S, L]
        """
        B, S, L = input_ids.shape
        ids = input_ids.view(B * S, L)
        mask = attention_mask.view(B * S, L)

        out = self.bert(ids, attention_mask=mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        proj = self.fc(cls_emb)

        return proj.view(B, S, -1)



# Story Reasoning Model

class StoryReasoningModel(nn.Module):
    """
    Multimodal Story Reasoning Network
    """

    def __init__(self, config):
        super().__init__()

        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.vocab_size = config["vocab_size"]

        # Encoders
        self.visual_encoder = VisualEncoder(self.embed_dim)
        self.text_encoder = TextEncoder(self.embed_dim)

        # Fusion
        self.fusion_fc = nn.Linear(self.embed_dim * 2, self.hidden_dim)

        # Temporal reasoning
        self.temporal_lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Text decoder
        self.text_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.text_decoder = nn.GRU(
            self.embed_dim,
            self.hidden_dim,
            batch_first=True
        )
        self.text_out = nn.Linear(self.hidden_dim, self.vocab_size)

        # Image decoder
        self.image_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, images, input_ids, attention_mask, target_ids=None):

        # Encode modalities
        v = self.visual_encoder(images)
        t = self.text_encoder(input_ids, attention_mask)

        B, S, E = v.shape

        # Contrastive alignment loss
        align_loss = contrastive_loss(
            t.view(B * S, E),
            v.view(B * S, E)
        )

        # Fusion
        fused = torch.cat([v, t], dim=-1)
        fused = F.relu(self.fusion_fc(fused))

        # Temporal reasoning
        _, (h, _) = self.temporal_lstm(fused)
        context = h[-1]

        # Image prediction
        pred_image = self.image_decoder(context)

        # Text prediction
        text_logits = None
        if target_ids is not None:
            emb = self.text_embedding(target_ids)
            hidden = context.unsqueeze(0)
            dec_out, _ = self.text_decoder(emb, hidden)
            text_logits = self.text_out(dec_out)

        return pred_image, text_logits, align_loss



# Standalone Test

if __name__ == "__main__":
    print("[INFO] Running model sanity check...")

    CONFIG = {
        "embed_dim": 256,
        "hidden_dim": 512,
        "vocab_size": 30522
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StoryReasoningModel(CONFIG).to(device)
    model.eval()

    B, S, L = 2, 4, 32

    images = torch.randn(B, S, 3, 224, 224).to(device)
    ids = torch.randint(0, CONFIG["vocab_size"], (B, S, L)).to(device)
    mask = torch.ones(B, S, L).to(device)
    tgt = torch.randint(0, CONFIG["vocab_size"], (B, L)).to(device)

    with torch.no_grad():
        pi, tl, al = model(images, ids, mask, tgt)

    print("[OK] Forward pass successful")
    print("Predicted image shape:", pi.shape)
    print("Text logits shape:", tl.shape)
    print("Contrastive loss:", float(al))
