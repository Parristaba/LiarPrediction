# LIAR Fact Classification with BERT & Metadata Fusion

This project explores multiple transformer-based models for classifying political statements from the LIAR dataset into six truthfulness categories using RoBERTa and a variety of metadata fusion techniques.

## Problem Statement

The LIAR dataset contains over 12,000 short political statements labeled with six classes of truthfulness:

- true
- mostly-true
- half-true
- barely-true
- false
- pants-fire

The goal is to build a robust classifier that can predict the label of a given statement, optionally enriched with speaker metadata.

## Project Structure

- 1. Data Preparation: Loads and cleans the data, performs label mapping, and handles missing metadata.
- 2. Dataset Creation: Creates separate datasets for text-only and metadata-augmented models.
- 3. Model Experiments:
  - Text-Only BERT: Uses only the cleaned statement text.
  - Text + Metadata Concatenation: Concatenates metadata string to the text input.
  - Special Tokens Model: Introduces special tokens for metadata fields.
  - Fused Metadata Model: Embeds metadata with a small MLP and fuses it with BERT output.

## Setup

```bash
# Create environment
conda create -n liar-classification python=3.9
conda activate liar-classification

# Install dependencies
pip install -r requirements.txt
```

Ensure GPU is available for faster training.

## Experiments

| Model                     | Test Accuracy | Macro F1 |
|--------------------------|---------------|----------|
| Text-Only RoBERTa        | 28.2%         | 28.1%    |
| Text + Metadata Concat   | 25.6%         | 25.7%    |
| Special Token Injection  | 20.6%         | 15.9%    |
| Metadata Fused RoBERTa   | 24.6%         | 24.4%    |

Current performance is above random (~16.7%) but far from perfect. Metadata alone did not significantly boost model accuracy in this iteration.

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- Macro Precision, Recall, and F1-score
- Confusion Matrix

These metrics are used due to class imbalance.

## Data

Download the LIAR dataset and place the TSV files in a folder named `liar_dataset/`.

Files used:

- train.tsv
- valid.tsv
- test.tsv

## Notable Features

- Weighted CrossEntropyLoss based on class frequency
- Dynamic and static metadata integration
- Special token augmentation
- Custom model with metadata MLP + frozen BERT layers
- Progress tracking via tqdm and confusion matrices

## How to Run

1. Preprocess data:
   ```bash
   python preprocess.py
   ```

2. Train and test any model from:
   - train_text_only_model.py
   - train_metadata_concat_model.py
   - train_special_token_model.py
   - train_fused_model.py

3. Results will be printed at the end of each run, including performance metrics and confusion matrix.

## Future Work

- Experiment with longer training (10–15 epochs + early stopping)
- Try alternate models like DeBERTa or DistilRoBERTa
- Apply label grouping (e.g., reduce 6 to 3 classes)
- Add more metadata (e.g., job title, context)
- Explore attention-based metadata fusion

## Author

Fevzi Kagan Becel