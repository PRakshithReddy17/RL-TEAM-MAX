# -*- coding: utf-8 -*-
"""
DeBERTa Factuality Checker Model

Fine-tunes a DeBERTa-v3-small model for three-class classification:
True, False, Neutral
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset, DatasetDict
import inspect
import math


class StepCheckFactualityChecker:
    """Factuality checker using DeBERTa-v3-small for claim verification."""

    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LENGTH = 256
    LABEL2ID = {"False": 0, "Neutral": 1, "True": 2}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    def __init__(self, model_name: str = None, device: str = None):
        """Initialize the checker with a pre-trained model."""
        if model_name is None:
            model_name = self.MODEL_NAME

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=self.ID2LABEL,
            label2id=self.LABEL2ID
        )

        self.model = self.model.to(device=self.device, dtype=torch.float32)
        self.model.float()

        # Verify FP32
        param_dtypes = {str(p.dtype) for p in self.model.parameters()}
        assert param_dtypes == {"torch.float32"}, f"Unexpected parameter dtypes: {param_dtypes}"

        # Verify all parameters are finite
        bad_weights = [
            name for name, p in self.model.named_parameters()
            if not torch.isfinite(p).all()
        ]
        assert not bad_weights, f"Non-finite pretrained parameters: {bad_weights[:5]}"

    def prepare_dataset(self, df: pd.DataFrame) -> DatasetDict:
        """
        Prepare tokenization and HuggingFace datasets from dataframe.

        Args:
            df: DataFrame with 'evidence', 'claim', 'gold_label', 'split' columns

        Returns:
            DatasetDict with train, validation, test splits
        """
        # Map labels
        df = df.copy()
        df["labels"] = df["gold_label"].map(self.LABEL2ID)
        assert df["labels"].notna().all(), "One or more gold labels were not mapped."
        df["labels"] = df["labels"].astype("int64")

        # Split frames
        split_frames = {
            split: df.loc[df["split"] == split].copy().reset_index(drop=True)
            for split in ["train", "validation", "test"]
        }

        # Create HuggingFace datasets
        raw_datasets = DatasetDict({
            split: Dataset.from_pandas(
                frame[["evidence", "claim", "labels"]],
                preserve_index=False
            )
            for split, frame in split_frames.items()
        })

        # Tokenize
        def tokenize_batch(batch):
            return self.tokenizer(
                batch["evidence"],
                batch["claim"],
                truncation=True,
                max_length=self.MAX_LENGTH
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_batch,
            batched=True,
            remove_columns=["evidence", "claim"]
        )

        # Verify tokenization
        for split in ["train", "validation", "test"]:
            assert len(tokenized_datasets[split]) > 0
            assert {"input_ids", "attention_mask", "labels"}.issubset(
                set(tokenized_datasets[split].column_names)
            )

        return tokenized_datasets, split_frames

    def test_forward_pass(self, tokenized_dataset: DatasetDict) -> np.ndarray:
        """Run a pre-training numerical health check."""
        self.model.eval()
        sample = tokenized_dataset["test"][0]

        sample_inputs = {
            "input_ids": torch.tensor([sample["input_ids"]], device=self.device),
            "attention_mask": torch.tensor([sample["attention_mask"]], device=self.device)
        }

        with torch.no_grad():
            sample_logits = self.model(**sample_inputs).logits

        assert sample_logits.dtype == torch.float32
        assert torch.isfinite(sample_logits).all()

        print("Pre-training numerical health check PASSED.")
        return sample_logits

    def train(
        self,
        tokenized_datasets: DatasetDict,
        train_size: int,
        batch_size: int = 8,
        epochs: int = 3,
        output_dir: str = "/content/stepcheck_fp32_results",
        **kwargs
    ) -> Trainer:
        """
        Train the factuality checker.

        Args:
            tokenized_datasets: DatasetDict with tokenized data
            train_size: Number of training examples
            batch_size: Batch size for training
            epochs: Number of training epochs
            output_dir: Directory to save checkpoints
            **kwargs: Additional TrainingArguments

        Returns:
            Trained Trainer instance
        """
        # Calculate warmup steps
        steps_per_epoch = math.ceil(train_size / batch_size)
        total_training_steps = steps_per_epoch * epochs
        warmup_steps = int(total_training_steps * 0.10)

        # Base training arguments
        training_kwargs = {
            "output_dir": output_dir,
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 25,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": epochs,
            "weight_decay": 0.01,
            "warmup_steps": warmup_steps,
            "max_grad_norm": 1.0,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_macro_f1",
            "greater_is_better": True,
            "fp16": False,
            "bf16": False,
            "report_to": "none",
            "seed": 20261112,
            "data_seed": 20261112,
        }

        # Check for eval_strategy vs evaluation_strategy
        ta_params = inspect.signature(TrainingArguments.__init__).parameters

        # Choose the evaluation argument supported by this Transformers version
        if "eval_strategy" in ta_params:
            training_kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in ta_params:
            training_kwargs["evaluation_strategy"] = "epoch"

        # Filter to only supported arguments
        training_kwargs = {
            key: value for key, value in training_kwargs.items()
            if key in ta_params
        }

        # Merge with any additional kwargs
        training_kwargs.update(kwargs)

        # Create training arguments
        training_args = TrainingArguments(**training_kwargs)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Define metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            from sklearn.metrics import (
                accuracy_score, precision_recall_fscore_support
            )

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0
            )
            return {
                "accuracy": accuracy_score(labels, predictions),
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1
            }

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        print("Trainer initialized successfully.")
        print("Training examples:", len(trainer.train_dataset))
        print("Validation examples:", len(trainer.eval_dataset))

        return trainer

    def predict(self, evidence: str, claim: str) -> Dict[str, float]:
        """
        Check a claim against evidence.

        Args:
            evidence: Evidence passage
            claim: Claim to verify

        Returns:
            Dictionary with predicted label and probabilities
        """
        self.model.eval()

        batch = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            logits = self.model(**batch).logits

        if not torch.isfinite(logits).all():
            raise ValueError("Model returned non-finite logits.")

        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()

        if not np.isfinite(probs).all():
            raise ValueError("Model returned non-finite probabilities.")

        return {
            "predicted_label": self.ID2LABEL[int(np.argmax(probs))],
            "p_false": float(probs[self.LABEL2ID["False"]]),
            "p_neutral": float(probs[self.LABEL2ID["Neutral"]]),
            "p_true": float(probs[self.LABEL2ID["True"]]),
            "soft_fact_reward": float(
                probs[self.LABEL2ID["True"]] - probs[self.LABEL2ID["False"]]
            )
        }

    def evaluate(self, tokenized_dataset: DatasetDict) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./temp_eval",
                report_to="none",
            ),
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        test_output = trainer.predict(tokenized_dataset["test"])
        test_logits = np.asarray(test_output.predictions)
        y_true = np.asarray(test_output.label_ids)

        y_pred = np.argmax(test_logits, axis=-1)

        from sklearn.metrics import (
            classification_report, confusion_matrix
        )

        print(classification_report(
            y_true, y_pred,
            labels=[0, 1, 2],
            target_names=["False", "Neutral", "True"],
            digits=4,
            zero_division=0
        ))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        print("Confusion matrix:")
        print(cm)

        return {
            "accuracy": float((y_true == y_pred).mean()),
            "predictions": y_pred,
            "labels": y_true
        }


__all__ = ["StepCheckFactualityChecker"]