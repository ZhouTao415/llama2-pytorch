# LLaMA 2 from Scratch (PyTorch) — RoPE · RMSNorm · MQA · KV Cache · GQA · SwiGLU + Inference

> Full re-implementation of core LLaMA 2 building blocks in PyTorch with clear, well-commented code and math notes. Includes multiple decoding strategies (greedy, beam search, temperature, top-k, nucleus/top-p, random sampling). PDF slides included.

[简体中文在下方 ⬇](#-简体中文说明)

![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

---

## ✨ Features
- **Architecture**: Token/position embeddings (RoPE), **RMSNorm**, **SwiGLU** FFN  
- **Attention**: **Self-Attention with KV Cache**, **Multi-Query Attention (MQA)**, **Grouped Query Attention (GQA)**  
- **Inference**: Greedy · Beam Search · Temperature · Random Sampling · **Top-k** · **Top-p (Nucleus)**  
- **Math Notes**: Step-by-step derivation for **Rotary Positional Embedding (RoPE)**  
- **Utilities**: Weight loading hooks, config system, reproducible seeds, profiling helpers  
- **Slides**: PDF slides summarizing concepts and code mapping

---

## 🗂️ Repository Structure

## 📈 Roadmap

- [ ] Multi-GPU / FSDP training script  
- [ ] FlashAttention / xFormers optional kernels  
- [ ] Quantization (int8/4) loaders  
- [ ] Export to ONNX / TensorRT  
- [ ] LoRA fine-tuning example
