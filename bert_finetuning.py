# BERT Fine-Tuning for Sentiment Classification
# Checkpoint Submission

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("🚀 Starting BERT Sentiment Fine-Tuning Checkpoint...\n")

# 1. Load Dataset
dataset = load_dataset("imdb")

# Use subset for faster training (you can increase for better results)
train_dataset = dataset["train"].shuffle(seed=42).select(range(4000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 2. Load Tokenizer & Model

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 4. Metrics & Training Args
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    logging_steps=50,
)

# 5. Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("Starting training... (This may take 15-40 minutes with GPU)")
trainer.train()

# 6. Final Evaluation
results = trainer.evaluate()
print("\n✅ Final Evaluation Results:")
print(results)

# Confusion Matrix
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)

cm = confusion_matrix(predictions.label_ids, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Fine-tuned BERT")
plt.show()

# 7. Test with Custom Examples
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

print("\n🧪 Testing Model on New Examples:")
examples = [
    "This movie was absolutely fantastic! I loved every moment.",
    "Terrible film. Waste of time and money.",
    "The acting was okay but the plot was confusing.",
    "One of the best movies I have seen this year!"
]

for ex in examples:
    print(f"Text: {ex}")
    print(f"Prediction: {predict_sentiment(ex)}\n")

print("🎉 Training and Evaluation Completed!")
