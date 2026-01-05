# Visual Storytelling with Multimodal Sequence Modelling

## Quick Links
- **[Experiments Notebook](experiment_notebook.ipynb)** – Full experimental workflow and implementation  
- **[Evaluation Results](results/)** – All The results are in this folder 
- **[Model Architecture](src/)** – Encoders, fusion, temporal modelling, and decoders  
---

## Innovation Summary
**This project introduces an explicit contrastive multimodal alignment loss to enforce semantic consistency between visual and textual embeddings in a multimodal sequence modelling task. The innovation improves narrative coherence and cross-modal alignment when predicting the next element of a visual story.**

---

## Executive Summary
This project investigates multimodal sequence modelling for visual story reasoning, a task requiring the integration of visual perception, natural language understanding, and temporal modelling. Given four image–text pairs forming a narrative, the system predicts the fifth image and its corresponding textual description while preserving narrative coherence.

The model combines a convolutional neural network for visual encoding, a transformer-based language model for text encoding, multimodal feature fusion, and a recurrent temporal model. A dual-decoder design enables simultaneous generation of the next image and text from a shared narrative representation. Transfer learning is employed to reduce computational requirements, and an explicit contrastive alignment loss is introduced to strengthen semantic correspondence between visual and textual embeddings.

Experiments conducted on a subset of the StoryReasoning dataset demonstrate that the model successfully captures narrative structure and semantic continuity, achieving competitive performance under limited data and training epochs.

---

## Key Results

| Metric       | Score |
|--------------|-------|
| BLEU         | **0.57** |
| ROUGE-L      | **0.68** |
| METEOR       | **0.74** |

These results indicate strong lexical overlap, semantic consistency, and narrative coherence in the generated textual outputs.

---

## Most Important Finding
- The explicit contrastive multimodal alignment loss significantly improves semantic consistency between images and text, reducing modality drift and preserving narrative flow across temporal steps.
- Qualitative evaluation shows that generated descriptions remain contextually aligned with prior story elements, despite limited training data.

---

## Model Architecture
The architecture follows a modular multimodal design:
- **Visual Encoder:** ResNet-50 pretrained on ImageNet, projecting features into a 256-dimensional embedding space.
- **Text Encoder:** DistilBERT, extracting contextualized [CLS] representations mapped to the same embedding space.
- **Fusion & Temporal Modelling:** Multimodal embeddings are fused and processed using a recurrent neural network to model narrative progression.
- **Dual Decoders:**  
  - Image decoder reconstructs the next image using upsampling layers.  
  - Text decoder generates the next textual description using sequence generation.

---

## Multimodal Alignment Innovation
A contrastive alignment loss is applied between visual and textual embeddings within each batch. The loss enforces bidirectional semantic consistency and is jointly optimized with image reconstruction and text generation losses. This explicit alignment mechanism represents the primary methodological contribution of the project.

---

## Training Strategy
- Optimizer: AdamW  
- Learning Rate: 1 × 10⁻⁴  
- Epochs: 2  
- Batch Size: 2  
- Transfer learning applied by freezing ResNet-50 and lower DistilBERT layers  

Training loss consistently decreased from **9.93** to **4.89**, indicating stable convergence despite limited data.

---

## Evaluation and Results
Text generation quality is evaluated using BLEU, ROUGE-L, and METEOR metrics. The model demonstrates strong semantic overlap with reference narratives. While image generation is limited by decoder simplicity and training time, generated images maintain basic structural consistency with story context.

---

## Limitations and Future Work
- Small training subset limits generalization.
- Image decoder struggles with fine-grained visual detail.
- Recurrent temporal modelling lacks an explicit attention mechanism.

Future work may incorporate transformer-based temporal models, attention mechanisms, and more advanced image generation architectures.

---

## How to Reproduce
1. Install dependencies  
   ```bash
   pip install -r requirements.txt
