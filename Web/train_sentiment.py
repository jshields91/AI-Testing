import tensorflow as tf

# Check for available devices (including GPU)
print(tf.config.list_physical_devices())

# Enable memory growth for the first GPU (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
logical_gpus = tf.config.list_logical_devices('GPU')

if logical_gpus:
  print("GPU is available!")
  device = tf.device('/device:GPU:0')  # Set device to first GPU
else:
  print("GPU is not available. Training on CPU.")
  device = tf.device('/device:CPU:0')  # Fallback to CPU

# %%
from datasets import load_dataset

dataset_name = "vamossyd/finance_emotions"

# %%
train_dataset = load_dataset(dataset_name, split="train[:90%]")
test_dataset = load_dataset(dataset_name, split="train[90%:]")

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# %%
str_to_int = {
    "neutral": 0,
    "sad": 1,
    "anger": 2,
    "disgust": 3,
    "surprise": 4,
    "fear": 5,
    "happy": 6,
}

# %%
def tokenize_batch(batch):
    tokenized_batch = tokenizer(
        batch['cleaned_text'],
        padding="max_length",
        truncation=True
    )

    tokenized_batch['label'] = [str_to_int[label] for label in batch['label']]
    
    return tokenized_batch

# %%
tokenized_train_data = train_dataset.map(tokenize_batch, batched=True)
tokenized_test_data = test_dataset.map(tokenize_batch, batched=True)

# %%
print(tokenized_train_data[99])

# %%
from transformers import TFAutoModelForSequenceClassification


# %%
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=7)

# %%
import numpy as np
import evaluate

# %%
metric = evaluate.load("accuracy")

# %%
def compute_accuracy_metric(eval_pred):
    logits, labels = eval_pred
    
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

# %%
from transformers import TFTrainingArguments
training_args = TFTrainingArguments(
   output_dir="reTrained_Model", 
   evaluation_strategy="epoch", 
   per_device_train_batch_size=8,  
   per_device_eval_batch_size=8,
)


# %%


def save_if_above_goal_accuracy(trainer, eval_result):
    # Extract accuracy from eval results
    accuracy = eval_result.get("eval_accuracy")

    if accuracy and accuracy > 0.95:
        trainer.save_model()
        print(f"Model saved to {training_args.output_dir} with accuracy: {accuracy:.4f}")
    else:
        print(f"Failed: Model accuracy: {accuracy:.4f}")


# %%
from transformers.trainer_tf import TFTrainer

# %%
trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    compute_metrics=compute_accuracy_metric,
    callbacks=[save_if_above_goal_accuracy]
)

# %%
trainer.train()


