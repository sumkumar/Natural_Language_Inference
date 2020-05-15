import tensorflow as tf
import torch
import os
import json
from tensorflow import keras
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from tqdm import tqdm
import wget
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def run_bert():

    # Get the GPU device name.
    device_name = tf.test.gpu_device_name()

    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')
        



    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")



    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    sen1_test = []
    sen2_test = []
    res_test = []
    sen1_train = []
    sen2_train = []
    res_train = []
    resp = urlopen("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    for line in zipfile.open('snli_1.0/snli_1.0_train.jsonl').readlines():
        x = json.loads(line)
        k = 0
        y = -1
        if x['gold_label'] == "contradiction":
            y = 0
            k = 1
        elif x['gold_label'] == "neutral":
            y = 1
            k = 1
        elif x['gold_label'] == "entailment":
            y = 2
            k = 1
        if k==1:
            sen1_train.append(x['sentence1'].lower())
            sen2_train.append(x['sentence2'].lower())
            res_train.append(y)
    for line in zipfile.open('snli_1.0/snli_1.0_test.jsonl').readlines():
        x = json.loads(line)
        k = 0
        y = -1
        if x['gold_label'] == "contradiction":
            y = 0
            k = 1
        elif x['gold_label'] == "neutral":
            y = 1
            k = 1
        elif x['gold_label'] == "entailment":
            y = 2
            k = 1
        if k==1:
            sen1_test.append(x['sentence1'].lower())
            sen2_test.append(x['sentence2'].lower())
            res_test.append(y)

    
    #file_id = '1czYDQiKQJmMsIKmw4HBcuKxN6vkSDVV2'
    file_id = '14r9IfokNmd-0E8Tgkqt8BbTZ9MJblypj'
    destination = './BERT_test.pt'
    download_file_from_google_drive(file_id, destination)

    model = torch.load('./BERT_test.pt')
    model.eval()
    model.cuda()

    input_ids = []
    attention_masks = []

    # For every sentence...
    for i in range(len(res_test)):
        encoded_dict = tokenizer.encode_plus(
                            sen1_test[i],                      # Sentence to encode.
                            sen2_test[i],
                            max_length = 512,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    res_test = torch.tensor(res_test)
    batch_size = 128  
    prediction_data = TensorDataset(input_ids, attention_masks, res_test)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    model.eval()
    predictions , true_labels = [], []
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      with torch.no_grad():
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)

    print('    DONE.')

    mypredict = []
    for item in predictions:
        for it in item:
            i = np.argmax(it)
            mypredict.append(i)

    sz = len(mypredict)
    acc = 0
    for i in range(sz):
        if mypredict[i] == res_test[i]:
            acc+=1
    print("Accuracy is :")
    print(acc*100/sz)
    with open('./deep_model.txt', "w") as file:
        for item in mypredict:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            elif item == 2:
                file.write("entailment\n")
            else:
                pass