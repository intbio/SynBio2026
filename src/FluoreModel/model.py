import torch.nn as nn
import torch.nn.init as init
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple, List
import warnings


import torch
import torch.nn as nn
import torch.nn.init as init


import torch.nn as nn


import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        
        # Конфигурация слоев для удобства настройки
        self.hidden_sizes = [int(input_size * 2), input_size * 4]
        self.final_hidden = input_size // 4
        self.output_size = 1
        
        # Слой 1: Входной слой с BatchNorm и Dropout
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(0.5)
        self.skip_layer1 = nn.Linear(input_size, self.hidden_sizes[-1])
        self.fc1 = self._make_linear_layer(input_size, self.hidden_sizes)
        
        # Слой 2: Сжатие до исходного размера
        self.layer_norm2 = nn.LayerNorm(self.hidden_sizes[-1])
        self.skip_layer2 = nn.Linear(self.hidden_sizes[-1], input_size)
        self.fc2 = self._make_linear_layer(self.hidden_sizes[-1], [int(input_size * 2), input_size])
        
        # Слой 3: Дополнительная обработка с skip-соединением
        self.layer_norm2_1 = nn.LayerNorm(input_size)
        self.skip_layer_2_1 = nn.Linear(input_size, input_size)
        self.layer2_1 = self._make_linear_layer(input_size, [input_size * 4, input_size * 2, input_size])
        
        # Слой 4: Постепенное уменьшение размерности
        self.layer_norm3 = nn.LayerNorm(input_size)
        self.skip_layer3 = nn.Linear(input_size, self.final_hidden)
        self.fc3 = self._make_linear_layer(input_size, [input_size // 2, self.final_hidden])
        self.dropout2 = nn.Dropout(0.3)
        
        # Выходной слой
        self.fc4 = self._make_linear_layer(
            self.final_hidden, 
            [input_size // 8, input_size // 16, input_size // 32, self.output_size]
        )
    
    def _make_linear_layer(self, in_size, out_sizes):
        """Создает последовательность линейных слоев с активациями ReLU между ними"""
        if not out_sizes:
            return nn.Identity()
            
        layers = []
        current_size = in_size
        
        for i, out_size in enumerate(out_sizes):
            layers.append(nn.Linear(current_size, out_size))
            if i < len(out_sizes) - 1:  # Добавляем ReLU после всех слоев, кроме последнего
                layers.append(nn.ReLU())
            current_size = out_size
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Первый блок
        identity = x
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.fc1(x) + self.skip_layer1(identity)
        
        # Второй блок
        identity = x
        x = self.layer_norm2(x)
        x = self.dropout(x)
        x = self.fc2(x) + self.skip_layer2(identity)
        
        # Третий блок
        identity = x
        x = self.layer_norm2_1(x)
        x = self.layer2_1(x) + self.skip_layer_2_1(identity)
        
        # Четвертый блок
        identity = x
        x = self.layer_norm3(x)
        x = self.dropout(x)
        x = self.fc3(x) + self.skip_layer3(identity)
        
        # Выходной блок
        x = self.dropout2(x)
        x = self.fc4(x)
        
        return x

    
      





class ResidualBlock1D(nn.Module):
    """
    Pre-activation блок с бутылочным горлышком
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, dropout_rate: float = 0.2):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = int(input_dim * 1.5)
        
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        skip = self.skip(x)
        
        # out = self.norm1(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # out = self.norm2(out)
        out = self.linear2(out)
        out = self.relu(out)
        # out = self.dropout(out)
        
        return out + skip  


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
    
    def _unfreeze_last_layers(self, num_layers: int = 3):
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
        # mask = attention_mask.unsqueeze(-1).float()
        pooled = embeddings[:, 1:-1].mean(dim=1)
        return pooled


class GFPRegressionModel(nn.Module):
    def __init__(self,
                 residual_dim,
                 regressor,
                 esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
                 freeze_esm: bool = True,
                 dropout_rate: float = 0.2,
                 max_length: int = 1024,
                 device: str = None):
        super().__init__()
        
        self.esm_block = ESM2Block(esm_model_name, freeze_esm, max_length, device)
        self.regressor = regressor.to(device)

    def forward(self, x):
        x = self.esm_block(x)
        x = self.regressor(x)
        return x

    def to(self, device, *args, **kwargs):
        self.regressor.to(device)
        self.final_linear.to(device)
        return super().to(device, *args, **kwargs)
    