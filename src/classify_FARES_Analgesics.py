import io
import os
import torch
import transformers
import pandas as pd
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,balanced_accuracy_score

from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer, AdamW,
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )

# Set seed for reproducibility,
# set_seed(123)

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'negative': 0, 'postive': 1}
# labels_ids = {'neg': 0, 'pos': 1}
# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)
max_length = 256
batch_size = 128
batches = 128
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


def train(dataloader, optimizer, scheduler, model, device_):
    r"""
    Train pytorch model on a single pass through the data loader.

    It will use the global variable `model` which is the transformer model
    loaded on `_device` that we want to train on.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

        optimizer_ (:obj:`transformers.optimization.AdamW`):
            Optimizer used for training.

        scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            PyTorch scheduler.

        device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

        :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss].
    """

    # Use global variable for model.
    # global model

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, model, device_):
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
    # global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

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

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def test(dataloader, model, device_):
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
    # global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # model.eval()

    # Evaluate data for one epoch
    predictions_labels = []
    logit_list = np.ones((10))
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
            if logit_list.shape == np.ones((10)).shape:
                logit_list = logits
            else:
                logit_list = np.vstack((logit_list, logits))
            # Convert these logits to list of predicted labels values.
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    return logit_list, true_labels, predictions_labels


# Get model configuration.
print('Loading configuraiton...')
def train_model(datapath,model_name,save_model_path, epochs):
    # used for finetune pretrain model
    # pretrain_model_name = 'bert-base-cased'
    pretrain_model_name = 'bert-base-cased'

    if 'finetune' in model_name:
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrain_model_name,
                                                  num_labels=n_labels)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrain_model_name)

        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrain_model_name,
                                                                   config=model_config)
    else:
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name,
                                                  num_labels=n_labels)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                                   config=model_config)

    # model_config = transformers.AutoConfig.from_pretrained('dmis-lab/biobert-base-cased-v1.2',
    #                                           num_labels=n_labels)

    # Get the actual model.

    # model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`' % device)

    print('Dealing with Train...')

    # Create pytorch dataset.
    train_dataset = Dataset(path=datapath + 'train.tsv',
                            use_tokenizer=tokenizer,
                            labels_ids=labels_ids,
                            max_sequence_len=max_length)
    # train_dataset = Dataset(path='/home/xwang1/code5/aclImdb/train',
    #                                     use_tokenizer=tokenizer,
    #                                     labels_ids=labels_ids,
    #                                     max_sequence_len=max_length)
    print('Created `train_dataset` with %d examples!' % len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Created `train_dataloader` with %d batches!' % len(train_dataloader))

    print('Dealing with ...')
    # Create pytorch dataset.
    valid_dataset = Dataset(path=datapath + 'test.tsv',
                            use_tokenizer=tokenizer,
                            labels_ids=labels_ids,
                            max_sequence_len=max_length)
    # valid_dataset = Dataset(path='/home/xwang1/code5/aclImdb/test',
    #                                     use_tokenizer=tokenizer,
    #                                     labels_ids=labels_ids,
    #                                     max_sequence_len=max_length)
    print('Created `valid_dataset` with %d examples!' % len(valid_dataset))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print('Created `eval_dataloader` with %d batches!' % len(valid_dataloader))

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}

    # Loop through each epoch.
    print('Epoch')

    for epoch in tqdm(range(epochs)):
        print(epoch)
        print()
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, model, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(valid_dataloader, model, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
            train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, model, device)



    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()),
                                              target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)
    PATH = save_model_path + model_name
    isExist = os.path.exists(PATH)
    if not isExist:
        os.makedirs(PATH)

    torch.save(model, PATH + '/model.h5')
    print('save model #############')
    print(PATH)


def test_model(datapath,model_name, PATH, PATH2, test_file='all.tsv'):
    # used for finetuned pretrain model
    # Create the evaluation report.
    def measurements(y_test, y_pred, y_pred_prob):
        acc = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        sensitivity = metrics.recall_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        specificity = TN / (TN + FP)
        npv = TN / (TN + FN)
        return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy]

    pretrain_model_name = 'bert-base-cased'
    print('Loading tokenizer...')
    if 'finetune' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrain_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    test_dataset = Dataset(path=datapath+test_file,
                                        use_tokenizer=tokenizer,
                                        labels_ids=labels_ids,
                                        max_sequence_len=max_length)


    test_dataloader = DataLoader(test_dataset, batch_size=batches, shuffle=False)


    isExist = os.path.exists(PATH2)
    if not isExist:
        os.makedirs(PATH2)

    model_name_or_path = PATH+model_name+'/'+'model.h5'
    model = torch.load(model_name_or_path)
    model.eval()
    logits_list,y_test,y_pred=test(test_dataloader, model, device)



    causal_set_address = PATH2+model_name+'/'
    isExist = os.path.exists(causal_set_address)
    if not isExist:
        os.makedirs(causal_set_address)
    if test_file == 'all.tsv':

        np.savetxt(causal_set_address+'np.csv', logits_list, delimiter=',', header=" #1,  #2")
    elif test_file == 'test.tsv':
        y_pred_prob = logits_list[:,1]
        metrics_list = measurements(y_test, y_pred, y_pred_prob)
        np.savetxt(causal_set_address + 'metrics.csv', metrics_list, delimiter=',', header=" TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy")
    else:
        y_pred_prob = logits_list[:, 1]
        metrics_list = measurements(y_test, y_pred, y_pred_prob)
        np.savetxt(causal_set_address + 'metrics2.csv', metrics_list, delimiter=',',
                   header=" TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy")
model_list=['albert-base-v1']
# model_list=['albert-base-v2','bert-base-cased','bert-large-cased','roberta-base','emilyalsentzer/Bio_ClinicalBERT'
#                 ,'emilyalsentzer/Bio_Discharge_Summary_BERT','allenai/scibert_scivocab_uncased','dmis-lab/biobert-base-cased-v1.2','/home/xwang1/livertox_extension/finetuned_model/Analgesics_finetune_BERT']
# model_list = ['albert-base-v2']
# model_list=['/home/xwang1/livertox_extension_codine/finetuned_model/codine_finetune_BERT_pretrain']
PATH = '/home/xwang1/livertox_extension/FAERS/Analgesics-induced_acute_liver_failure/finetuned_model_batch_128_2/'
PATH2 = '/home/xwang1/livertox_extension/FAERS/Analgesics-induced_acute_liver_failure/causal_inference_set__batch_128_2/'
datapath = '/home/xwang1/livertox_extension/FAERS/DeepCausalPV-master/dat/Analgesics-induced_acute_liver_failure/proc/'
epochs = 40
for model_name in model_list:
    train_model(datapath,model_name,PATH,epochs)
    test_model(datapath,model_name, PATH, PATH2)
    test_model(datapath, model_name, PATH, PATH2,'test.tsv')

# model_list=['albert-base-v2','bert-base-cased','bert-large-cased','roberta-base','emilyalsentzer/Bio_ClinicalBERT'
#                 ,'emilyalsentzer/Bio_Discharge_Summary_BERT','allenai/scibert_scivocab_uncased','dmis-lab/biobert-base-cased-v1.2','/home/xwang1/livertox_extension/finetuned_model/Tramadol_finetune_BERT']
# PATH = '/home/xwang1/livertox_extension/FAERS/Tramadol-related_mortalities/finetuned_model/'
# PATH2 = '/home/xwang1/livertox_extension/FAERS/Tramadol-related_mortalities/causal_inference_set/'
# datapath = '/home/xwang1/livertox_extension/FAERS/DeepCausalPV-master/dat/Tramadol-related_mortalities/proc/'
# epochs = 20
# for model_name in model_list:
#     train_model(datapath,model_name,PATH,epochs)
#     test_model(datapath,model_name, PATH, PATH2)
# Model class must be defined somewhere
# model2 = torch.load(PATH+'model.h5')
# model2.eval()

# Plot confusion matrix.
# plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
#                       classes=list(labels_ids.keys()), normalize=True,
#                       magnify=3,
#                       );