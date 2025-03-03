import os
import logging
import torch
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)

def train_model(model, train_dataset, eval_dataset, output_dir, training_args):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=training_args.weight_decay,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=training_args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

def evaluate_model(model, eval_dataset):
    trainer = Trainer(model=model)
    eval_results = trainer.evaluate(eval_dataset)
    logger.info(f"Evaluation results: {eval_results}")
    return eval_results

def save_model(model, output_dir):
    model.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")