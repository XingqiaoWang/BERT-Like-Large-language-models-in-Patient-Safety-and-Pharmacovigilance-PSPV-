import transformers
import math
from datasets import load_dataset, ClassLabel
import random
import pandas as pd
import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

datasets = load_dataset("text", data_files={"train": '/home/xwang1/livertox_extension/FAERS_data/train_Tramadol.txt', "validation": '/home/xwang1//livertox_extension/FAERS_data/train_Tramadol.txt'})
datasets2 = load_dataset("text", data_files={"train": '/home/xwang1/livertox_extension/FAERS_data/train_Analgesics.txt', "validation": '/home/xwang1//livertox_extension/FAERS_data/train_Analgesics.txt'})
PATH1 = '/home/xwang1/livertox_extension/finetuned_model/Tramadol_finetune_BERT'
isExist = os.path.exists(PATH1)
if not isExist:
    os.makedirs(PATH1)

PATH2 = '/home/xwang1/livertox_extension/finetuned_model/Tramadol_finetune_BERT'
isExist = os.path.exists(PATH2)
if not isExist:
    os.makedirs(PATH2)

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df)

model = transformers.AutoModelForMaskedLM.from_pretrained('bert-base-cased')

tokenizer = transformers.AutoTokenizer.from_pretrained(
    'bert-base-cased',
)

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128,padding="max_length", truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# block_size = tokenizer.model_max_length
block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

model_name = 'bert'
training_args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
model.save_pretrained(PATH1)

model = transformers.AutoModelForMaskedLM.from_pretrained('bert-base-cased')
tokenized_datasets = datasets2.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
model.save_pretrained(PATH2)
