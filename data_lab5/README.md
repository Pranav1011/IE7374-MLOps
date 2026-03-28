# Lab 2 — Streaming LM Pipeline

## Overview

This lab demonstrates a streaming language modeling data pipeline using Hugging Face Datasets and PyTorch. The pipeline processes large-scale text corpora without loading entire datasets into memory.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `wikitext-103-raw-v1` (raw text, 103M tokens) |
| Tokenizer | GPT-2 (GPT2TokenizerFast) |
| Block size | 512 tokens |
| Batch size | 16 |
| Mode | Streaming |

## Pipeline Architecture

```
1. Stream Dataset  →  2. Tokenize (lazy)  →  3. Rolling Buffer  →  4. IterableDataset  →  5. DataLoader
   (wikitext-103)       (GPT-2 tokenizer)     (group into 512)     (PyTorch wrapper)      (batch_size=16)
```

### Steps

1. **Load dataset** — Stream `wikitext-103-raw-v1` without loading into RAM
2. **Tokenize** — Lazy tokenization via `.map()`, no padding/truncation
3. **Rolling buffer** — Concatenate tokens across documents, yield 512-token chunks
4. **IterableDataset** — Wrap generator for PyTorch DataLoader compatibility
5. **Collate & batch** — Produce `input_ids`, `attention_mask`, and `labels` tensors

## Output Format

Each batch contains:
- `input_ids`: `[batch_size, block_size]` = `[16, 512]`
- `attention_mask`: `[16, 512]`
- `labels`: copy of `input_ids` for LM loss

## Installation

```bash
pip install datasets transformers torch
```

## Usage

Run all cells in order in Jupyter / VS Code / Colab:

```bash
jupyter notebook Lab2.ipynb
```

## Directory Structure

```
data_lab5/
├── Lab2.ipynb       # Streaming LM pipeline notebook
└── README.md
```
