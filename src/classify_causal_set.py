import io
import os
import torch
import numpy as np
import pandas as pd
import transformers
# from tqdm.notebook import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer, AdamW,
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )

# Set seed for reproducibility,
set_seed(123)

# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batches = 64

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels_ids = {'negative': 0, 'postive': 1}
model_name_or_path='dmis-lab/biobert-base-cased-v1.2'
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

def test(dataloader, device_):
    r"""Validation function to evaluate model performance on a
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model's performance.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

      device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """

    # Use global variable for model.
    global model

    # Tracking variables
    predictions_labels = []

    true_labels = []
    # total loss for this epoch.
    total_loss = 0
    logit_list = np.ones((10))
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print(logit_list)
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            print(logits.shape)
            if logit_list.shape == np.ones((10)).shape:
                logit_list = logits

                print('if')
            else:
                logit_list = np.vstack((logit_list, logits))
                print('else')

    # Return all logit_list
    print(logit_list.shape)
    return logit_list

class Dataset(Dataset):
    r"""PyTorch Dataset class for loading data.

    This is where the data parsing happens and where the text gets encoded using
    loaded tokenizer.

    This class is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

      path (:obj:`str`):
          Path to the data partition.

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, path, use_tokenizer, labels_ids, max_sequence_len=None):

        # Check max sequence length.
        max_sequence_len = use_tokenizer.max_len if max_sequence_len is None else max_sequence_len
        texts = []
        labels = []
        print('Reading partitions...')
        # Since the labels are defined by folders with data we loop
        # through each label.
        df = pd.read_csv(path, sep='\t')
        sentence_list=df['sentence']
        for content in sentence_list:
            content = fix_text(content)
            texts.append(content)
        labels=df['label']
        # Number of exmaples.
        self.n_examples = len(labels)
        print(self.n_examples)
        # Use tokenizer on texts. This can take a while.
        print('Using tokenizer on all texts. This can take a while...')
        self.inputs = use_tokenizer(texts, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',
                                    max_length=max_sequence_len)
        # Get maximum sequence length.
        self.sequence_len = self.inputs['input_ids'].shape[-1]
        print('Texts padded or truncated to %d length!' % self.sequence_len)
        # Add labels.
        self.inputs.update({'labels': torch.tensor(labels)})
        print('Finished!\n')

        return

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

          item (:obj:`int`):
              Index position to pick an example to return.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.

        """

        return {key: self.inputs[key][item] for key in self.inputs.keys()}

datapath = '/home/xwang1/code4/dataset/'
test_dataset = Dataset(path=datapath+'test.tsv',
                                    use_tokenizer=tokenizer,
                                    labels_ids=labels_ids,
                                    max_sequence_len=max_length)


test_dataloader = DataLoader(test_dataset, batch_size=batches, shuffle=False)
model_list=['albert-base-v2','bert-base-cased','bert-large-cased','bert-base-casedfine_tuned','roberta-base','emilyalsentzer/Bio_ClinicalBERT'
            ,'emilyalsentzer/Bio_Discharge_Summary_BERT','allenai/scibert_scivocab_uncased','dmis-lab/biobert-base-cased-v1.2']
PATH = '/home/xwang1/livertox_extension/finetuned_model/'
PATH2 = '/home/xwang1/livertox_extension/causal_inference_set/'
isExist = os.path.exists(PATH2)
if not isExist:
    os.makedirs(PATH2)

for model_name in model_list:
    model_name_or_path = PATH+model_name+'/'+'model.h5'
    model = torch.load(model_name_or_path)
    model.eval()
    logits_list=test(test_dataloader, device)
    causal_set_address = PATH2+model_name
    isExist = os.path.exists(causal_set_address)
    if not isExist:
        os.makedirs(causal_set_address)
    np.savetxt(causal_set_address+'np.csv', logits_list, delimiter=',', header=" #1,  #2")
# Plot confusion matrix.
# plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
#                       classes=list(labels_ids.keys()), normalize=True,
#                       magnify=3,
#                       );