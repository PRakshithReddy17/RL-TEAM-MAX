# StepCheck: Supervised Factuality Checker

A synthetic pipeline for training and evaluating factuality classifiers using DeBERTa and GRPO (Generalized Relative Policy Optimization).

## Project Structure

```
.
├── main.py                          # Main entry point for the pipeline
├── stepcheck/                       # Core package
│   ├── __init__.py
│   ├── data/                        # Data generation module
│   │   ├── __init__.py
│   │   └── generator.py              # Synthetic dataset generation
│   ├── model/                         # Model module
│   │   ├── __init__.py
│   │   └── checker.py                # DeBERTa factuality checker
│   ├── training/                      # Training optimization
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py         # BM25 evidence retrieval
│   │   └── grpo.py                   # GRPO reward functions
│   ├── evaluation/                    # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py                  # Accuracy, F1, error analysis
│   └── utils/                         # Utilities
│       ├── __init__.py
│       └── visualization.py            # Plotting functions
└── requirements.txt                   # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run specific phases
python main.py --mode data      # Generate synthetic dataset only
python main.py --mode train     # Train classifier only
python main.py --mode eval      # Evaluate trained model only
python main.py --mode compare   # Compare reward variants
```

## Pipeline Phases

1. **Data Generation**: Creates 600 synthetic reasoning trails with True/False/Neutral claims
2. **Classifier Training**: Fine-tunes DeBERTa-v3-small for three-class classification
3. **Evaluation**: Computes accuracy, F1, and soft factuality rewards
4. **Retrieval Analysis**: BM25 retrieval quality metrics
5. **Model Comparison**: Compares reward variants

## Key Components

- **StepCheckFactualityChecker**: Wraps DeBERTa for claim verification
- **BM25Retriever**: Lightweight BM25 for evidence retrieval
- **GRPO Reward Functions**: Soft factuality reward and advantage adjustment

## Research Note

All evidence and claims in this dataset are **synthetic and fictional**. Results are a pipeline smoke test, not evidence of real-world factuality-checker performance.

