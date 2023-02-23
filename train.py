from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import evaluate

dataset = load_dataset("amcoff/skolmat")["train"].train_test_split(test_size=0.1)

id2label = {k: v for k, v in enumerate(dataset["train"].features["label"].names)}
label2id = {v: k for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained("KBLab/bert-base-swedish-cased")

max_length = 128


def tokenize_function(examples):
    return tokenizer(
        examples["meal"], padding="max_length", truncation=True, max_length=max_length
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["train"]

model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/bert-base-swedish-cased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.save_model("model")
