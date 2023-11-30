import data_loader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch 
import torchaudio
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
import numpy as np




train_df, test_df = train_test_split(data_loader.dataset_df, test_size=0.2, random_state=42)
print(f"Training Set Size: {len(train_df)}")
print(f"Testing Set Size: {len(test_df)}")

def calculate_label_proportions(df):
    label_counts = df['label'].value_counts()
    total_samples = len(df)
    proportions = label_counts / total_samples
    return proportions

# Calculate proportions for training set
train_label_proportions = calculate_label_proportions(train_df)
print("Training Set Label Proportions:")
print(train_label_proportions)

# Calculate proportions for testing set
test_label_proportions = calculate_label_proportions(test_df)
print("\nTesting Set Label Proportions:")
print(test_label_proportions)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["file_path"])
    batch["speech"] = speech_array.squeeze().numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["label"]
    return batch



train_dataset = train_dataset.map(speech_file_to_array_fn)
test_dataset = test_dataset.map(speech_file_to_array_fn)

def prepare_dataset(batch):
    # Process each example in the batch
    input_values= []
    labels = []
    for speech, sampling_rate, target_text in zip(batch["speech"], batch["sampling_rate"], batch["target_text"]):
        # Process the speech
        input_features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_values
        input_values.append(input_features.squeeze().tolist())

        # Encode the labels
        #encoded_labels = processor.tokenizer.batch_encode_plus([target_text], padding=True, add_special_tokens=True)
        #label_ids = encoded_labels.input_ids[0]
        encoded_labels = processor.tokenizer.encode(target_text, add_special_tokens=True)
        labels.append(encoded_labels)

    batch["input_values"] = input_values
    batch["labels"] = labels
    return batch
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batch_size=8, batched=True)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, batch_size=8, batched=True)
# Print the first example from the training set
print(train_dataset[0])
# Print the first example from the testing set
print(test_dataset[0])

def custom_collator(batch):
    # Check if 'input_values' is in the batch
    if 'input_values' not in batch[0]:
        raise KeyError("input_values key not found in batch. Check dataset preparation.")

    # Pad the input_values
    input_values = pad_sequence([torch.tensor(d["input_values"]) for d in batch], batch_first=True, padding_value=processor.feature_extractor.padding_value)

    # Pad the labels
    labels = pad_sequence([torch.tensor(d["labels"]) for d in batch], batch_first=True, padding_value=-100)

    return {
        "input_values": input_values, 
        "labels": labels
    }

training_args = TrainingArguments(
  output_dir="./wav2vec2-base-960h",
  group_by_length=False,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=1,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  warmup_steps=500,
  save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=custom_collator,
)

trainer.train()

trainer.evaluate()

def evaluate_model(model, processor, test_dataset):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in test_dataset:
            input_values = torch.tensor(batch["input_values"]).unsqueeze(0)
            labels = torch.tensor(batch["labels"]).unsqueeze(0)
            
            outputs = model(input_values)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

            pred_label = processor.batch_decode(pred_ids)[0]
            true_label = processor.decode(labels[0], skip_special_tokens=True)

            predictions.append(pred_label)
            references.append(true_label)

    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='weighted')
    return accuracy, f1

# Evaluate the model
accuracy, f1 = evaluate_model(model, processor, test_dataset)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")


model.save_pretrained("./wav2vec2-base-960h")
