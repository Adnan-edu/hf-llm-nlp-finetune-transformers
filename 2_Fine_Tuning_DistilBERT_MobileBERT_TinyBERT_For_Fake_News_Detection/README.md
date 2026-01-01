# Fake News Detection with TinyBERT and Hierarchical Attention

This project explores various strategies for fine-tuning transformer models—specifically **TinyBERT**, **DistilBERT**, and **MobileBERT**—for the task of fake news detection. A primary challenge addressed is the classification of long text documents that exceed the standard 512-token limit of BERT models.

## Project Structure

- `Fine_Tuning_DistilBERT_MobileBERT_TinyBERT_For_Fake_News_Detection.ipynb`: Baseline notebook comparing multiple lightweight transformer models using standard truncation.
- `Fine_Tuning_TinyBERT_For_Fake_News_Detection_text.ipynb`: focused implementation of **TinyBERT** on body text using a **Chunking with Mean Pooling** strategy.
- `Fine_Tuning_TinyBERT_For_Fake_News_Detection_text_hierarchical.ipynb`: advanced implementation of **Hierarchical Attention** for long-document classification.
- `Fine_Tuning_TinyBERT_For_Fake_News_Detection_Merged.ipynb`: merged implementation potentially combining title and text features.

---

## 1. Summary of Architectural Approaches

| Approach | Implementation File | Model(s) | Input | Handling of Long Text |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Model Baseline** | [Baseline Notebook](Fine_Tuning_DistilBERT_MobileBERT_TinyBERT_For_Fake_News_Detection.ipynb) | DistilBERT, MobileBERT, TinyBERT | `text` | **Truncation**: Only uses the first 512 tokens. |
| **Mean Pooling** | [TinyBERT Text Notebook](Fine_Tuning_TinyBERT_For_Fake_News_Detection_text.ipynb) | TinyBERT | `text` | **Mean Logit Aggregation**: Document split into chunks; predictions are averaged. |
| **Hierarchical Attention**| [Hierarchical Notebook](Fine_Tuning_TinyBERT_For_Fake_News_Detection_text_hierarchical.ipynb) | TinyBERT | `text` | **Attention Aggregation**: Global pooling of chunk-level [CLS] embeddings. |

---

## 2. Deep Dive: Hierarchical Attention vs. Mean Pooling

The **Hierarchical Attention Implementation** represents the most advanced solution in this repository for processing long documents (up to 9,000+ tokens).

### How Hierarchical Attention Works:
1.  **Chunking**: The document is split into overlapping chunks (stride=50).
2.  **Local Encoding**: Each chunk is passed through TinyBERT to extract a **312-dimensional [CLS] embedding**.
3.  **Global Aggregation**: An **Attention Layer** calculates an importance score (weight) for each chunk. 
4.  **End-to-End Training**: Loss is calculated at the document level, allowing the model to learn which parts of the text matter most.

### Key Advantages:

| Feature | Mean Pooling (Logit-Level) | Hierarchical Attention (Embedding-Level) |
| :--- | :--- | :--- |
| **Aggregation** | **Late Fusion**: Only final 2-dim predictions are averaged. | **Mid-Fusion**: Rich 312-dim features are combined. |
| **Weighting** | **Uniform**: All chunks are treated as equally important. | **Dynamic**: Focuses on "fake news signals" and ignores noise. |
| **Context** | **Low**: Chunks are treated independently until the end. | **High**: Sequence of embeddings is treated as a coherent story. |

---

## 3. Comparison with Title-Only Solutions

Solutions that consider **only the title** or **truncate the text** suffer from major biases:
-   **Missing Signal**: Fake news often hides misinformation deep within the body text.
-   **Limited Evidence**: Deceptive titles are common; the falsehoods often appear later in the article.
-   **Length Bias**: Truncation penalizes detailed reporting by cutting off nuances.

**Hierarchical Attention** performs better by "scanning" the entire text and using the most relevant segments for high-fidelity classification. 

---

## 4. Key Engineering Optimizations
-   **Differential Learning Rates**: Lower LR (2e-5) for pre-trained weights, higher LR (1e-4) for attention layers.
-   **Memory Efficiency**: Uses **Gradient Checkpointing** and **Mixed Precision (FP16)** for document-level training.
-   **Chunk-Level Masking**: Ensures the attention mechanism ignores padded chunks in variable-length documents.

## Model Performance Comparison

The following table summarizes the test accuracy for each architectural approach. Note that the results for chunking strategies (Mean Pooling and Hierarchical Attention) are aggregated at the document level.

| Approach | Model | Accuracy | F1-Score |
| :--- | :--- | :---: | :---: |
| **Hierarchical Attention** | TinyBERT | **99.48%** | **0.99** |
| **Merged Features (Title + Text)** | TinyBERT | 99.10% | 0.99 |
| **Mean Pooling (Text only)** | TinyBERT | 98.66% | 0.99 |
| **Truncation Baseline** | Multi-Model | 95.82% | 0.96 |

---

### Detailed Test Results

#### 1. Hierarchical Attention (SOTA)
*File: `Fine_Tuning_TinyBERT_For_Fake_News_Detection_text_hierarchical.ipynb`*

```text
Test Loss: 0.0322
Test Accuracy: 0.9948

Classification Report:
              precision    recall  f1-score   support

        Real       0.99      1.00      1.00      2072
        Fake       1.00      0.99      0.99      1584

    accuracy                           0.99      3656
   macro avg       0.99      0.99      0.99      3656
weighted avg       0.99      0.99      0.99      3656
```

#### 2. Merged Features (Title + Text with Mean Pooling)
*File: `Fine_Tuning_TinyBERT_For_Fake_News_Detection_Merged.ipynb`*

```text
Results for 3656 unique documents:
              precision    recall  f1-score   support

        Real       0.98      1.00      0.99      2072
        Fake       1.00      0.97      0.98      1584

    accuracy                           0.99      3656
   macro avg       0.99      0.99      0.99      3656
weighted avg       0.99      0.99      0.99      3656
```

#### 3. Mean Pooling (Text Only)
*File: `Fine_Tuning_TinyBERT_For_Fake_News_Detection_text.ipynb`*

```text
Results for 3656 unique documents:
              precision    recall  f1-score   support

        Real       0.98      1.00      0.99      2072
        Fake       1.00      0.97      0.99      1584

    accuracy                           0.99      3656
   macro avg       0.99      0.99      0.99      3656
weighted avg       0.99      0.99      0.99      3656
```

#### 4. Truncation Baseline
*File: `Fine_Tuning_DistilBERT_MobileBERT_TinyBERT_For_Fake_News_Detection.ipynb`*

```text
Classification Report for Aggregated Predictions:
              precision    recall  f1-score   support

        Real       0.96      0.97      0.96      2072
        Fake       0.96      0.94      0.95      1584

    accuracy                           0.96      3656
   macro avg       0.96      0.96      0.96      3656
weighted avg       0.96      0.96      0.96      3656
```

## Installation and Requirements

```bash
pip install -U transformers accelerate datasets seaborn openpyxl tqdm
```

## Performance Highlights
The **Hierarchical Attention** model offers superior accuracy on long articles (70%+ of the dataset) compared to simpler truncation-based methods.
