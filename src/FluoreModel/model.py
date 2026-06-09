import torch.nn as nn
import torch.nn.init as init
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple, List
import warnings


import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualBlock1D(nn.Module):
    """
    Остаточный блок с:
    - Pre-activation (Order: Norm -> ReLU -> Linear) для лучшей сходимости
    - LayerNorm вместо BatchNorm (лучше для векторов фиксированной длины)
    - Dropout для регуляризации
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        
        self.skipconnection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # Pre-activation блок 1 (input_dim -> input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim)
        
        # Pre-activation блок 2 (input_dim -> output_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.linear1, self.linear2]:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        
        # Инициализация skip connection, если это Linear
        if isinstance(self.skipconnection, nn.Linear):
            init.xavier_uniform_(self.skipconnection.weight)
            if self.skipconnection.bias is not None:
                init.constant_(self.skipconnection.bias, 0)
    
    def forward(self, x):
        # Сохраняем skip connection
        skip = self.skipconnection(x)
        
        # Pre-activation путь
        out = self.norm1(x)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = self.relu(out)
        out = self.linear2(out)
        
        # Сложение с skip connection
        out = out + skip
        
        return out


class ESM2Block(nn.Module):
    
    def __init__(self,
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 freeze: bool = True,
                 max_length: int = 1024,
                 device: str = None):
        """
        Args:
            model_name: имя модели ESM-2 на Hugging Face
            freeze: заморозить веса ESM-2
            pooling_strategy: стратегия пулинга ('mean', 'max', 'cls')
            max_length: максималtruncation=True,
            padding=True, ьная длина последовательности
            device: устройство ('cuda' или 'cpu')
        """
        super().__init__()
        self.device = device
        
        # Устройство
        self.max_length = max_length
        
        # Загрузка токенизатора и модели
        self.esm = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Заморозка весов
        if freeze:
            print("Режим: ESM-2 заморожен")
            for param in self.esm.parameters():
                param.requires_grad = False
        else:
            print("Режим: ESM-2 дообучается")
            # Можно разморозить только последние слои
            self._unfreeze_last_layers(num_layers=6)
        
        # Перенос на устройство
        self.esm.eval() if freeze else self.esm.train()
    
    def _unfreeze_last_layers(self, num_layers: int = 6):
        """Размораживает последние слои ESM-2"""
        # Замораживаем всё
        for param in self.esm.parameters():
            param.requires_grad = False
        
        # Размораживаем последние num_layers слоёв
        total_layers = len(self.esm.encoder.layer)
        for layer in self.esm.encoder.layer[total_layers - num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Размораживаем embedding слой
        for param in self.esm.embeddings.parameters():
            param.requires_grad = True
    
    def forward(self, inputs) -> torch.Tensor:
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        outputs = self.esm(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled


class GFPRegressionModel(nn.Module):
    def __init__(self,
                 residual_dim: list,
                 esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
                 freeze_esm: bool = True,
                 dropout_rate: float = 0.2,
                 max_length: int = 1024,
                 device: str = None):
        super().__init__()
        
        self.esm_block = ESM2Block(esm_model_name, freeze_esm, max_length, device)
        self.regressor = nn.Sequential(*[ResidualBlock1D(residual_dim[i], residual_dim[i + 1]) for i in range(len(residual_dim) - 1)]).to(device)

    def forward(self, x):
        x = self.esm_block(x)
        x = self.regressor(x)
        return x
    