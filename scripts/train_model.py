import sys
sys.path.append('../src/FluoreModel/')

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from functools import partial
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch

import model
from gfp_datasets import SimpleDataset, collate_fn
import train_utils



BATCH_SIZE = 550
ESM_TYPE = "facebook/esm2_t36_3B_UR50D"



train_df = pd.read_csv("../data/rel_brightness_top/train.csv")

X_train = train_df['seq']
y_train = train_df['rel_brightness']

val_df = pd.read_csv("../data/rel_brightness_top/val.csv")
X_val = val_df['seq']
y_val = val_df['rel_brightness']


tokenizer = AutoTokenizer.from_pretrained(ESM_TYPE, return_tensors='pt', padding=True, truncation=True)
my_collate_fn = partial(collate_fn, tokenizer=tokenizer)    

train_dataset = SimpleDataset(X_train.to_numpy(), y_train.to_numpy())
val_dataset = SimpleDataset(X_val.to_numpy(), y_val.to_numpy())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=my_collate_fn, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn, num_workers=4)

regression_model = model.GFPRegressionModel(2560, device='cuda', freeze_esm=True, esm_model_name=ESM_TYPE)

if torch.cuda.device_count() > 1:
    print(f"Используем {torch.cuda.device_count()} GPU!")
    regression_model = nn.DataParallel(regression_model) 


regression_model = regression_model.train()


train_utils.train_model(
    regression_model, train_loader, val_loader, save_path="../data/best_models/gfp4_best.pt", epochs=10
)
