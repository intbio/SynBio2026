import sys
sys.path.append('../src/FluoreModel/')

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer
from torch import nn
import torch

import model
from gfp_datasets import SimpleDataset, collate_fn
import train_utils



BATCH_SIZE = 16
ESM_TYPE = "facebook/esm2_t33_650M_UR50D"


mutations_df = pd.read_csv("../data/data.csv")
mutations_df.Brightness = mutations_df.Brightness.astype('float32')


train_df = pd.read_csv("../data/train.csv")
X_train = train_df['seq']
y_train = train_df['Brightness']

val_df = pd.read_csv("../data/val.csv")
X_val = val_df['seq']
y_val = val_df['Brightness']


model = model.GFPRegressionModel([1280, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 1], device='cuda', freeze_esm=False)
if torch.cuda.device_count() > 1:
    print(f"Используем {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model) 


tokenizer = AutoTokenizer.from_pretrained(ESM_TYPE)
my_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=1024)    

train_dataset = SimpleDataset(X_train.to_numpy(), y_train.to_numpy())
val_dataset = SimpleDataset(X_val.to_numpy(), y_val.to_numpy())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn)

train_utils.train_model(model, train_loader, val_loader, save_path='../data/best_models/gfp2_best.pt')
