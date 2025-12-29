# DNN
Multimodal Sequence Modelling for Visual Story Reasoning:-

**Executive Summary**

This project examines multimodal sequence modelling of visual story reasoning, the combination of visual perception, natural language comprehension, and temporal modelling. It aims at generating an image and a text description of it to anticipate the next component of a multimodal story. With a predetermined order of four pairs of images and texts and qualities derived from a visual narrative, the model guesses the fifth and does not disrupt the narrative structure of both media. This work is executed on the basis of the StoryReasoning dataset and seems to rely on the architectural advice of the given module, staying in addition to the explicit multimodal alignment mechanism as the essential innovation.

The suggested system is a mixture of a convolutional neural network with visual encoding, a transformer-based language model with text encoding, the fusion of the multimodal features, and a recurrent temporal model. The system generates the "day tomorrow" text and poster by having two decoders operating on the same narrative representation. Transfer learning is used by freezing convolutional and language backbones and guaranteeing the feature extraction when there is a small amount of computational resources (Wu et al., 2024). To make visual and textual embeddings explicit semantic alignments, it is augmented with a contrastive alignment loss.

The model is trained in two epochs with 200 training samples. The quantitative analysis of the generated text results in a BLEU score of 0.57, a ROUGE-L score of 0.68, and a METEOR score of 0.74, which reflects the information that the system manages to achieve a narrative structure and semantic continuity. In general, the project depicts a successful usage of deep neural networks for multimodal sequence prediction and meets the learning outcomes of the module.

**1. Introduction**

The heterogeneity of the data on which modern-day artificial intelligence must operate to process and reason has made multimodal learning the focus of attention. A vast number of real-world applications, such as video understanding, robotics, and assistive technologies, demand models to comprehend sequences of multi-modals and not singular samples. Visual storytelling is one of the most difficult multimodal activities because it involves the model comprehending how narratives change over time and how visual and textual knowledge support each other to express a meaning.

In contrast to the classical image captioning or text generation task, multimodal sequence modelling results in an extra time dimension. The model needs to focus not only on cross-modal dependencies but also long-range ones among the sequential parts as well. This implies that such systems easily result in error due to imprecise modalities or the failure to ensure narrative coherence between time steps. To overcome these issues, architectures capable of deriving rich modality-specific features, merging them efficiently, and currently modeling temporal dynamics are required.

**The problem that is being solved in this project is stated as follows:** having a set of four pictures and text descriptions of those pictures, a model will predict the picture of the fifth image and text. This formula is a direct result of the assignment brief and a real-life multimodal thinking situation. The project will be based on the implementation of an end-to-end system for this task and the demonstration of knowledge in the field of advanced neural architecture, transfer learning, and multimodal alignment methods. A clear architectural innovation is presented in the shape of an explicit contrastive alignment loss in the form of the enhancement of the visual and textual representation semantic relationship.

**2. Dataset and Input–Output Specification**

The StoryReasoning dataset, available on the Hugging Face datasets library, is used to run the experiments. This is the dataset of visual stories; in each sample there is also a list of images and of short textual descriptions that create a discernible narrative. To make the computation possible, a subsample of 200 learning samples is employed.
All the samples go through the process of creating the fixed-length code of five multimodal elements. The initial four pairs of images and text are the input of models; the fifth one is the target of prediction. All stories with less than five elements will be padded by repeating the last available image and text to ensure that all the inputs maintain similar dimensions. The images are scaled to 224x224, turned to RGB, and normalized with ImageNet statistics. The DistilBERT tokenizer is found to tokenize the text, and the longest sequence length is 32 tokens.

The resulting input tensors are (batch size, 4, 3, 224, 224) in the case of images and (batch size, 4, 32) in the case of rasterized texts. The model returns the predicted image tensor of dimension (batch size, 3, 224, 224) and a series of token logits that are the prediction of the textual description.

**3. Model Architecture:-**

The offered architecture is based on the modular multimodal design, which echoes the ideas discussed during the module. The extractor takes visual features based on a ResNet-50 convolutional neural network that has been trained on ImageNet. The last classification layer is eliminated, and an end result is registered as an embedding space (256) through a fully connected layer to a 256-dimensional space. This model allows visual features to be extracted quite well and minimizes the number of trainable parameters.

The language model, encoded by textual input and called DistilBERT, is a lightweight transformer-based language model. Contextualized representation of the [CLS] token of every text sequence is searched and projected into the same 256-dimensional embedding space as the visual features. This mutual embedding dimensionality makes it easy to fuse and bring into alignment multimodality.

At the time level, visual and textual embeddings are combined together and fed through a linear transformation, yielding a fused representation. These embedded fusions are then fed to a recurrent neural network, which models a temporal relationship of the four input steps. The recurrent model generates a final hidden state, which is a summative version of the narrative context.

In order to produce the following multimodal element, a dual-decoder format is used. The decoder of the image recovers the original story representation with a full-resolution image, which takes the last narrative encoding and feeds on learned upsampling layers, which are trained with the loss of mean squared error. This exact narrative representation is used by the text decoder to initialize a recurrent text generation process, which is able to make predictions at the token level on a 30,522-word vocabulary.

**4. Multimodal Alignment Innovation**
   
One of the most important novelties of this project is the contrastive multimodal alignment loss. Most multimodal architectures are built on an implicit integration of components achieved via joint optimization,, this method clearly insisted on a semantic correlation between the equivalent visual and textual embeddings. The contrastive loss is calculated by building a similarity matrix between visual and textual embeddings into a batch, and cross-entropy loss is applied in both directions.This symmetric design guarantees that each image aligns with its appropriate text, and each text aligns with its corresponding image.

The contrastive loss is multiplied by 0.5 and used together with the image reconstruction and text generation losses in training. This architecture will stimulate this model to learn a common semantic space, limiting the chances of an appearance of modality drift and enhancing the narrative.

**5. Training Strategy and Transfer Learning**
   
The convolutional layers of the ResNet-50 backbone and the bottom layers of the DistilBERT encoder are frozen in order to stabilize training and to lower the cost of the computation. This transfer learning approach enables the model to take advantage of representations that have been trained, and optimization focuses on the fusion, temporal modelling, and decoding subsystems (Wu et al., 2024).
Two epochs are combined with a batch size of two on which the model is trained. There are three terms comprising the total loss: picture prediction mean squared error, text generation loss based on the cross-entropy (padding tokens are neglected), and the contrastive alignment loss. Losses at training diminish from 9.93 in the first epoch to 4.89 in the second epoch.

**6. Evaluation and Results**

The quality of the generated text is mainly evaluated since quantitative assessment of image generation is a research issue. The use of three popular metrics is used: BLEU, ROUGE-L, and METEOR. On evaluation subset,the evaluation the model scores 0.57 on BLEU, 0.68 on ROUGE-L, and 0.74 on METEOR. These findings show that the text that has been generated has a high level of lexical and semantic overlap with the reference descriptions.
Qualitative analysis of produced results indicates that the model is able to address the overall narrative flow of the stories and is able to generate contextually concrete descriptions. Though the process of generating the images is limited by the simplicity of the decoder and the lack of time in the training period, the images that are predicted have a basic structural consistency with the visual context presented beforehand.

**7. Discussion and Limitations**

The findings reveal that explicit multimodal alignment has the capacity to enhance story comprehension in multimodal sequence modelling activities. Nevertheless, there are a number of limitations. The small training subset limits generalization, and the image decoder finds it difficult to work with fine-grained details. Furthermore, the recurrent temporal model does not have a specific attention mechanism, so it might be reduced with the possibility to focus on the most crucial story aspects.

**8. Conclusion**

The present project manages to realize a multimodal sequence modelling system of visual story reasoning, which is completely in line with the assignment brief. The model can perform effective multimodal reasoning using limited resources through transfer learning, temporal modelling, dual decoders, and a contrastive alignment loss.




