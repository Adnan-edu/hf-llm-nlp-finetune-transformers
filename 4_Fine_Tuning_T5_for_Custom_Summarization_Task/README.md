# Fine-Tuning T5 for Custom Summarization

This project demonstrates the process of fine-tuning the T5 (Text-to-Text Transfer Transformer) model for abstractive text summarization. It transitions from using pre-trained baselines on news articles to fine-tuning a custom model on dialogue-based data.

## 🚀 Project Overview

The objective is to leverage the power of the T5 architecture to generate concise, human-like summaries. The project walks through:
1. **Baseline Evaluation**: Testing pre-trained `t5-small` and `BART` models on the CNN/DailyMail dataset.
2. **Custom Fine-Tuning**: Adapting `t5-small` specifically for conversation summarization using the DialogSum dataset.
3. **Inference**: Deploying the fine-tuned model for real-world summarization tasks using Hugging Face Pipelines.

## 📊 Datasets

- **CNN/DailyMail**: A large-scale dataset of news articles used for initial baseline testing. It focuses on well-structured journalistic text.
- **DialogSum (samsum)**: A dataset containing human-annotated dialogues and their summaries. This is used for fine-tuning to handle the nuances of conversational language.

## 🛠️ Workflow Architecture

The notebook follows a structured pipeline:

### 1. Environment Setup
Installation of essential libraries including `transformers`, `datasets`, `accelerate`, and `sentencepiece`.

### 2. Exploratory Analysis
- Loading the `cnn_dailymail` and `dialogsum` datasets.
- Visualizing distributions of dialogue and summary lengths to determine optimal padding and truncation strategies.

### 3. Baseline Comparison
Running zero-shot summarization using:
- `ubikpt/t5-small-finetuned-cnn`
- `facebook/bart-large-cnn`

### 4. Data Preprocessing
- Tokenization using `AutoTokenizer` for `t5-small`.
- Formatting the input prefix as `"summarize: "` (standard for T5).
- Managing attention masks and label IDs for sequence-to-sequence training.

### 5. Fine-Tuning
- **Model**: `t5-small` (Seq2Seq architecture).
- **Optimizer**: AdamW with weight decay.
- **Strategy**: Gradient accumulation (steps=350) to manage memory efficiency on smaller hardware.
- **Trainer**: Utilizing the Hugging Face `Trainer` API for efficient training across 10 epochs.

### 6. Verification & Inference
Saving the fine-tuned model and testing it on custom dialogue snippets using the `summarization` pipeline.

## 💻 How to Use

1. **Requirements**: Ensure you have a GPU environment (like Google Colab) for training.
2. **Execution**: Run the `Fine_Tuning_T5_for_Custom_Summarization.ipynb` notebook sequentially.
3. **Model Weights**: After training, the model is saved to the `t5_samsum_summarization` directory, which can be loaded for future inference.

## 🔧 Model Specifications

- **Architecture**: `AutoModelForSeq2SeqLM` (`t5-small`)
- **Parameters**: ~60 million
- **Checkpoint**: `t5-small`
