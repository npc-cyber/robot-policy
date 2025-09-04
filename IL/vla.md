# pi0的预训练模型结构

这个不会确实离谱，可以看hugface的readme

[PaliGemma](https://hf-mirror.com/google/paligemma-3b-pt-22)

PaliGemma is a versatile and lightweight vision-language model (VLM) inspired by PaLI-3 and based on open components such as the SigLIP vision model and the Gemma language model. 

PaliGemma is the composition of a Transformer decoder and a Vision Transformer image encoder, with a total of 3 billion params. The text decoder is initialized from Gemma-2B. The image encoder is initialized from SigLIP-So400m/14. PaliGemma is trained following the PaLI-3 recipes.

可以发现包含 SigLIP(视觉编码器) 和 Gemma(语言解码器)

补充说明一下 

SigLIP（Sigmoid Contrastive Language-Image Pretraining）是Google开发的一种视觉语言模型，其核心架构确实基于‌Vision Transformer（ViT）‌，但针对多模态任务进行了专项优化。


# smolvla 用的 smolvlm 和siglip好像

# gr00t好像用的eagle2

# umi的硬件设计 umi夹爪为什么要带编码
https://umi-gripper.github.io/#paper
Hardware Design

	
