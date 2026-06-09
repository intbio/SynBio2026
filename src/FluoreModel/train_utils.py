import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import os


def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=1, writer=None, epoch=0):
    """
    Одна эпоха обучения с логированием в TensorBoard
    
    Args:
        model: модель
        dataloader: DataLoader с данными (должен выдавать sequences и targets)
        optimizer: оптимизатор
        criterion: функция потерь
        device: устройство ('cuda' или 'cpu')
        accumulation_steps: шаги накопления градиентов (для больших батчей)
        writer: SummaryWriter для TensorBoard
        epoch: номер эпохи для логирования
    
    Returns:
        avg_loss: средняя потеря за эпоху
        predictions: список предсказаний
        targets: список истинных значений
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Для накопления градиентов
    optimizer.zero_grad()
    
    # Прогресс-бар
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (sequences, targets) in enumerate(pbar):
        # Переносим targets на устройство
        targets = targets.to(device).float()
        
        # Forward pass
        predictions = model(sequences)
        
        # Loss
        loss = criterion(predictions, targets)
        
        # Backward pass с накоплением градиентов
        loss = loss / accumulation_steps
        loss.backward()
        
        # Обновляем веса после accumulation_steps шагов
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Статистика
        total_loss += loss.item() * accumulation_steps
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Логируем loss каждый батч (опционально)
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Batch/TrainLoss', loss.item() * accumulation_steps, global_step)
        
        # Обновляем прогресс-бар
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    # Обработка остаточных градиентов
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    
    # Логируем распределение предсказаний для обучения
    if writer:
        writer.add_histogram('Train/Predictions', np.array(all_predictions), epoch)
        writer.add_histogram('Train/Targets', np.array(all_targets), epoch)
    
    return avg_loss, np.array(all_predictions), np.array(all_targets)


def validate_epoch(model, dataloader, criterion, device, writer=None, epoch=0):
    """
    Одна эпоха валидации с логированием в TensorBoard
    
    Args:
        model: модель
        dataloader: DataLoader с данными
        criterion: функция потерь
        device: устройство
        writer: SummaryWriter для TensorBoard
        epoch: номер эпохи для логирования
    
    Returns:
        avg_loss: средняя потеря
        predictions: список предсказаний
        targets: список истинных значений
        metrics: словарь с метриками (R², RMSE, MAE)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for sequences, targets in pbar:
            targets = targets.to(device).float()
            
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # Вычисляем метрики
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = {
        'r2': r2_score(all_targets, all_predictions),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
        'mae': mean_absolute_error(all_targets, all_predictions),
    }
    
    # Логируем в TensorBoard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/R2', metrics['r2'], epoch)
        writer.add_scalar('Val/RMSE', metrics['rmse'], epoch)
        writer.add_scalar('Val/MAE', metrics['mae'], epoch)
        
        # Логируем распределения
        writer.add_histogram('Val/Predictions', all_predictions, epoch)
        writer.add_histogram('Val/Targets', all_targets, epoch)
        
        # Логируем scatter plot
        fig = create_scatter_plot(all_targets, all_predictions, epoch)
        writer.add_figure('Val/Actual_vs_Predicted', fig, epoch)
        plt.close(fig)
    
    return avg_loss, all_predictions, all_targets, metrics


def create_scatter_plot(targets, predictions, epoch):
    """Создаёт scatter plot для TensorBoard"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Линия идеального предсказания
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal', linewidth=2)
    
    ax.set_xlabel('Actual Brightness', fontsize=12)
    ax.set_ylabel('Predicted Brightness', fontsize=12)
    ax.set_title(f'Predictions vs Actual (Epoch {epoch+1})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def log_model_weights(writer, model, epoch, log_freq=5):
    """Логирует веса и градиенты модели в TensorBoard"""
    if epoch % log_freq == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'Weights/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)


def train_model(model,
                train_loader,
                val_loader,
                epochs=50,
                lr=1e-3,
                weight_decay=1e-4,
                device='cuda',
                patience=15,
                scheduler_patience=10,
                scheduler_factor=0.5,
                accumulation_steps=1,
                save_best=True,
                save_path='best_model.pt',
                verbose=True,
                log_dir='../logs/tensorboard',
                experiment_name=None,
                log_weights_freq=5):
    """
    Полный цикл обучения модели с TensorBoard логированием
    
    Args:
        ... (все предыдущие аргументы)
        log_dir: директория для логов TensorBoard
        experiment_name: имя эксперимента (по умолчанию - дата и время)
        log_weights_freq: частота логирования весов (каждые N эпох)
    
    Returns:
        history: словарь с историей обучения
        best_model: лучшая модель (если save_best=True)
    """
    
    # Создаём имя эксперимента
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаём директорию для логов
    log_path = os.path.join(log_dir, experiment_name)
    writer = SummaryWriter(log_path)
    print(f"\n📊 TensorBoard logs: {log_path}")
    print(f"🚀 Запустите: tensorboard --logdir={log_dir}\n")
    
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler для уменьшения LR при застревании
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=verbose
    )
    
    # Функция потерь
    criterion = nn.MSELoss()
    
    # Логируем гиперпараметры
    writer.add_hparams({
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'patience': patience,
        'accumulation_steps': accumulation_steps,
        'model_type': model.__class__.__name__
    }, {})
    
    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_rmse': [],
        'val_mae': [],
        'lr': []
    }
    
    # Для ранней остановки
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*60}")
    print(f"Устройство: {device}")
    print(f"Эпох: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Patience: {patience}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Обучение
        train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            accumulation_steps, writer, epoch
        )
        
        # Валидация
        val_loss, val_preds, val_targets, val_metrics = validate_epoch(
            model, val_loader, criterion, device, writer, epoch
        )
        
        # Логируем learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # Обновляем scheduler
        scheduler.step(val_loss)
        
        # Логируем веса модели
        log_model_weights(writer, model, epoch, log_weights_freq)
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_metrics['r2'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['lr'].append(current_lr)
        
        # Печать результатов
        if verbose:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val R²:     {val_metrics['r2']:.4f}")
            print(f"Val RMSE:   {val_metrics['rmse']:.4f}")
            print(f"Val MAE:    {val_metrics['mae']:.4f}")
            print(f"LR:         {current_lr:.2e}")
            
            if val_metrics['r2'] < 0:
                print(f"⚠️  Внимание: R² отрицательный! Модель хуже среднего")
        
        # Сохранение лучшей модели
        if save_best and val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_metrics['r2'],
                'history': history
            }, save_path)
            if verbose:
                print(f"✅ Сохранена лучшая модель (R² = {val_metrics['r2']:.4f})")
        else:
            patience_counter += 1
        
        # Ранняя остановка
        if patience_counter >= patience:
            if verbose:
                print(f"\n🛑 Early stopping! Нет улучшения {patience} эпох.")
            break
    
    # Логируем финальные метрики
    writer.add_scalar('Final/Best_Val_R2', best_val_r2, 0)
    writer.add_scalar('Final/Best_Val_Loss', best_val_loss, 0)
    writer.add_text('Final/Summary', f"Best Val R²: {best_val_r2:.4f}, Best Val Loss: {best_val_loss:.4f}", 0)
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Лучший Val R²: {best_val_r2:.4f}")
    print(f"Лучший Val Loss: {best_val_loss:.4f}")
    print(f"\n📊 TensorBoard logs: {log_path}")
    print(f"🚀 Запустите: tensorboard --logdir={log_dir}")
    print(f"{'='*60}")
    
    # Загружаем лучшую модель
    if save_best:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружена лучшая модель из {save_path}")
    
    return history, model


def test_model(model, test_loader, device, load_path=None, writer=None, experiment_name=None):
    """
    Тестирование модели на отложенной выборке с логированием в TensorBoard
    
    Args:
        model: модель
        test_loader: DataLoader для теста
        device: устройство
        load_path: путь к сохранённой модели (опционально)
        writer: SummaryWriter для TensorBoard
        experiment_name: имя эксперимента для создания writer
    
    Returns:
        predictions, targets, metrics
    """
    
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружена модель из {load_path}")
    
    # Если нет writer, но есть experiment_name, создаём новый
    if writer is None and experiment_name is not None:
        writer = SummaryWriter(os.path.join('../logs/tensorboard', f'test_{experiment_name}'))
    
    model.eval()
    
    all_predictions = []
    all_targets = []

    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for sequences, targets in pbar:
            targets = targets.to(device).float()
            predictions = model(sequences)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = {
        'r2': r2_score(all_targets, all_predictions),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
        'mae': mean_absolute_error(all_targets, all_predictions),
    }
    
    # Логируем результаты теста в TensorBoard
    if writer:
        writer.add_scalar('Test/R2', metrics['r2'], 0)
        writer.add_scalar('Test/RMSE', metrics['rmse'], 0)
        writer.add_scalar('Test/MAE', metrics['mae'], 0)
        
        # Логируем scatter plot
        fig = create_scatter_plot(all_targets, all_predictions, 0)
        fig.suptitle('Test Set: Actual vs Predicted', fontsize=14)
        writer.add_figure('Test/Actual_vs_Predicted', fig, 0)
        plt.close(fig)
        
        # Логируем распределения
        writer.add_histogram('Test/Predictions', all_predictions, 0)
        writer.add_histogram('Test/Targets', all_targets, 0)
        
        writer.close()
    
    print(f"\n{'='*50}")
    print(f"ТЕСТОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*50}")
    print(f"R²:     {metrics['r2']:.4f}")
    print(f"RMSE:   {metrics['rmse']:.4f}")
    print(f"MAE:    {metrics['mae']:.4f}")
    print(f"{'='*50}")
    
    return all_predictions, all_targets, metrics