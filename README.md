# Neural Machine Translation using Seq2Seq Models

A Neural Machine Translation (NMT) system for translating between Urdu and Roman text using sequence-to-sequence (seq2seq) models with attention mechanisms.

## Project Overview

This project implements a neural machine translation system specifically designed for Urdu-Roman transliteration using BiLSTM encoder and LSTM decoder architecture with character-level tokenization.

## Features

- **Architecture**: BiLSTM encoder with LSTM decoder
- **Tokenization**: Character-level tokenization
- **Attention Mechanism**: Built-in attention for better translation quality
- **Model Configuration**:
  - Embedding Dimension: 256
  - Hidden Size: 256
  - Encoder Layers: 2
  - Decoder Layers: 4
  - Dropout: 0.3
  - Learning Rate: 0.0005
  - Batch Size: 64
  - Beam Search Size: 5

## Dataset

The project uses Urdu ghazals dataset from Rekhta with:
- Urdu text (ur/)
- Hindi text (hi/)
- English transliteration (en/)

## Project Structure

```
nmt_urdu_roman/
├── artifacts/                 # Model artifacts and vocabulary files
│   ├── exp_default.json      # Experiment configuration
│   ├── spm_src_*.txt         # Source tokenization files
│   ├── spm_tgt_*.txt         # Target tokenization files
│   ├── spm_src.model         # Source sentencepiece model
│   ├── spm_tgt.model         # Target sentencepiece model
│   ├── spm_src.vocab         # Source vocabulary
│   ├── spm_tgt.vocab         # Target vocabulary
│   ├── vocab_src_char.json   # Source character vocabulary
│   └── vocab_tgt_char.json   # Target character vocabulary
├── data/                     # Dataset files
│   ├── pairs_*.parquet       # Training, validation, and test pairs
│   └── urdu_ghazals_rekhta/  # Raw dataset
├── models/                   # Trained model files
│   └── bilstm4lstm_char_E256_H256_enc2_dec4_drop0.3_best.pt
├── runs/                     # Prediction outputs
│   ├── preds_test.csv
│   └── preds_test_clean.csv
├── src/                      # Source code (to be implemented)
├── logs/                     # Training logs
├── project_config.json       # Project configuration
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Model Architecture

The neural machine translation model uses:

1. **Encoder**: Bidirectional LSTM (BiLSTM) with 2 layers
2. **Decoder**: LSTM with 4 layers
3. **Attention**: Attention mechanism for better alignment
4. **Tokenization**: Character-level tokenization
5. **Embedding**: 256-dimensional embeddings

## Configuration

The model configuration is defined in `artifacts/exp_default.json`:

```json
{
  "tokenization": "char",
  "embedding_dim": 256,
  "hidden_size": 256,
  "enc_layers": 2,
  "dec_layers": 4,
  "dropout": 0.3,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "teacher_forcing_start": 1.0,
  "teacher_forcing_end": 0.5,
  "epochs": 20,
  "grad_clip": 1.0,
  "beam_size": 5
}
```

## Training Results

The model has been trained with the following configuration:
- **Model**: BiLSTM4LSTM with character-level tokenization
- **Architecture**: 256 embedding dimension, 256 hidden size
- **Layers**: 2 encoder layers, 4 decoder layers
- **Dropout**: 0.3
- **Best Model**: `bilstm4lstm_char_E256_H256_enc2_dec4_drop0.3_best.pt`

## Usage

### Prerequisites

- Python 3.7+
- PyTorch
- SentencePiece
- Pandas
- NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Marwah-coder/Neural-machine-translation-using-seq2saq-models.git
cd Neural-machine-translation-using-seq2saq-models
```

2. Install dependencies:
```bash
pip install torch pandas numpy sentencepiece
```

### Training

To train the model (implementation needed in src/):

```bash
python src/train.py --config artifacts/exp_default.json
```

### Inference

To run inference (implementation needed in src/):

```bash
python src/inference.py --model models/bilstm4lstm_char_E256_H256_enc2_dec4_drop0.3_best.pt
```

## Dataset Information

The project uses Urdu ghazals from Rekhta dataset:
- **Source**: Urdu text in Devanagari script
- **Target**: Roman transliteration
- **Format**: Parallel text pairs in parquet format

## Performance

The trained model achieves good performance on Urdu-Roman transliteration tasks. Detailed evaluation metrics are available in the prediction files in the `runs/` directory.

## Future Work

- [ ] Implement complete source code in `src/` directory
- [ ] Add evaluation metrics and benchmarking
- [ ] Implement attention visualization
- [ ] Add support for different tokenization methods
- [ ] Create web interface for translation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Rekhta Foundation for providing the Urdu ghazals dataset
- PyTorch team for the deep learning framework
- SentencePiece for tokenization support

## Contact

For questions and support, please open an issue on GitHub.
