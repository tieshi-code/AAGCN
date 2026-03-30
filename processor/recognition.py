#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        # 默认 loss（如需类别权重，会在 load_data() 里根据训练集标签自动重建）
        self.loss = nn.CrossEntropyLoss()

    def load_data(self):
        super().load_data()

        # 自动使用类别权重（处理类别不平衡）
        if getattr(self.arg, 'use_class_weight', False) and self.arg.phase == 'train':
            if 'train' not in self.data_loader:
                return

            dataset = self.data_loader['train'].dataset
            if not hasattr(dataset, 'label'):
                self.io.print_log('⚠️  use_class_weight=true 但 dataset 没有 label 属性，跳过类别权重。')
                return

            labels = np.array(dataset.label, dtype=np.int64)
            if labels.size == 0:
                self.io.print_log('⚠️  训练集 labels 为空，跳过类别权重。')
                return

            # 优先使用配置里的 num_class
            num_class = None
            try:
                num_class = int(self.arg.model_args.get('num_class'))
            except Exception:
                num_class = int(labels.max() + 1)

            from collections import Counter
            counts = Counter(labels.tolist())
            total = int(labels.shape[0])

            # class-balanced 权重: w_c = total / (num_class * count_c)
            weights = np.zeros((num_class,), dtype=np.float32)
            for c in range(num_class):
                cnt = counts.get(c, 0)
                weights[c] = (total / (num_class * cnt)) if cnt > 0 else 0.0

            # 归一化到均值为1（避免整体 loss 尺度漂移）
            nonzero = weights > 0
            if nonzero.any():
                weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.dev)
            self.loss = nn.CrossEntropyLoss(weight=weight_tensor)

            self.io.print_log('启用 CrossEntropyLoss 类别权重 (use_class_weight=true)')
            self.io.print_log(f'  训练集样本数: {total}')
            self.io.print_log(f'  训练集类别计数: {dict(sorted(counts.items()))}')
            self.io.print_log(f'  类别权重(归一化): {np.round(weights, 4).tolist()}')
        
    def load_optimizer(self):
        # 支持AdamW优化器
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            # 支持自定义优化器参数
            optimizer_kwargs = {
                'lr': self.arg.base_lr,
                'weight_decay': self.arg.weight_decay
            }
            # 添加optimizer_args中的参数
            if hasattr(self.arg, 'optimizer_args') and self.arg.optimizer_args:
                optimizer_kwargs.update(self.arg.optimizer_args)

            self.optimizer = optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs)
        else:
            raise ValueError("Unsupported optimizer: {}".format(self.arg.optimizer))
            
        # 支持多种学习率调度器
        if self.arg.scheduler == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.arg.T_max, 
                eta_min=self.arg.eta_min)
        elif self.arg.scheduler == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.arg.step_size, 
                gamma=self.arg.gamma)
        elif self.arg.scheduler == 'ReduceLROnPlateau':
            # ReduceLROnPlateau 需要从验证损失来调整
            # 注意：PyTorch 的 ReduceLROnPlateau 不支持 verbose 参数
            scheduler_kwargs = {
                'mode': getattr(self.arg, 'scheduler_mode', 'min'),
                'factor': getattr(self.arg, 'scheduler_factor', 0.3),
                'patience': getattr(self.arg, 'scheduler_patience', 7),
                'threshold': getattr(self.arg, 'scheduler_threshold', 1e-4),
                'min_lr': getattr(self.arg, 'scheduler_min_lr', 1e-7)
            }
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_kwargs
            )
            # 如果需要打印信息，手动记录
            if getattr(self.arg, 'scheduler_verbose', True):
                self.io.print_log('ReduceLROnPlateau scheduler initialized: {}'.format(scheduler_kwargs))
            self.use_plateau_scheduler = True
        else:
            self.use_plateau_scheduler = False
        # 如果没有指定调度器，则使用默认的调整学习率方法

    def adjust_lr(self, metric=None):
        # ReduceLROnPlateau 需要验证指标（如 val_loss）
        if hasattr(self, 'use_plateau_scheduler') and self.use_plateau_scheduler:
            if metric is not None:
                self.scheduler.step(metric)
            # ReduceLROnPlateau 没有 get_last_lr()，需要从 optimizer 获取
            self.lr = self.optimizer.param_groups[0]['lr']
        # 如果使用了其他调度器，则调用调度器的step方法
        elif hasattr(self, 'scheduler'):
            self.scheduler.step()
            self.lr = self.scheduler.get_last_lr()[0]
        # 否则使用原有的学习率调整方法
        elif self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        if self.result is None:
            return
        # result 转为 numpy
        result = np.array(self.result)
        rank = result.argsort()
        if self.label is None:
            return
        # label 转为 numpy
        label = np.array(self.label)
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
    
    def show_acceptable_accuracy(self, margin=1):
        """
        Calculate and display acceptable accuracy allowing an error margin of ±margin.
        For example, if margin=1, predictions within ±1 of the true label are considered correct.
        """
        if self.result is None or self.label is None:
            return
        # result 和 label 转为 numpy
        result = np.array(self.result)
        label = np.array(self.label)
        
        # 获取预测值（最高概率的类别）
        pred = result.argmax(axis=1)
        
        # 计算在误差范围内的预测
        hit_acceptable = np.abs(pred - label) <= margin
        accuracy = hit_acceptable.sum() * 1.0 / len(hit_acceptable)
        
        self.io.print_log('\tAcceptable accuracy (margin=±{}): {:.2f}%'.format(margin, 100 * accuracy))

    def train(self):
        self.model.train()
        # warmup learning rate adjustment
        if hasattr(self.arg, 'warmup') and self.arg.warmup and self.meta_info['epoch'] < self.arg.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.meta_info['epoch'] + 1) / self.arg.warmup_epochs
            lr = self.arg.warmup_start_lr + (self.arg.base_lr - self.arg.warmup_start_lr) * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            # Normal learning rate adjustment (for ReduceLROnPlateau, metric is passed in test phase)
            self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for batch_data in loader:
            # 支持截肢标签（如果数据加载器返回3个值）
            if len(batch_data) == 3:
                data, label, amputee_type = batch_data
            else:
                data, label = batch_data
                amputee_type = None

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            # 如果模型支持截肢标签，传递它
            if amputee_type is not None:
                # 检查模型是否支持amputee_type参数
                import inspect
                model_forward_sig = inspect.signature(self.model.forward)
                if 'amputee_type' in model_forward_sig.parameters:
                    # 直接传递列表/元组，让模型内部处理batch级别
                    output = self.model(data, amputee_type=amputee_type)
                else:
                    output = self.model(data)
            else:
                output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for batch_data in loader:
            # 支持截肢标签（如果数据加载器返回3个值）
            if len(batch_data) == 3:
                data, label, amputee_type = batch_data
            else:
                data, label = batch_data
                amputee_type = None
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                # 如果模型支持截肢标签，传递它
                if amputee_type is not None:
                    # 检查模型是否支持amputee_type参数
                    import inspect
                    model_forward_sig = inspect.signature(self.model.forward)
                    if 'amputee_type' in model_forward_sig.parameters:
                        # 直接传递列表/元组，让模型内部处理batch级别
                        output = self.model(data, amputee_type=amputee_type)
                    else:
                        output = self.model(data)
                else:
                    output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
            
            # show acceptable accuracy with ±1 error margin
            self.show_acceptable_accuracy(margin=1)
            
            # Store validation loss for ReduceLROnPlateau scheduler adjustment
            self.last_val_loss = self.epoch_info['mean_loss']

    def save_checkpoint(self, epoch):
        """只保存模型权重（不保存 optimizer / epoch info），用于减少磁盘占用"""
        # Save model
        model_filename = 'epoch{}_model.pt'.format(epoch + 1)
        self.io.save_model(self.model, model_filename)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # scheduler
        parser.add_argument('--scheduler', default=None, help='type of learning rate scheduler')
        parser.add_argument('--T_max', type=int, default=50, help='T_max for CosineAnnealingLR')
        parser.add_argument('--eta_min', type=float, default=0, help='eta_min for CosineAnnealingLR')
        parser.add_argument('--step_size', type=int, default=10, help='step_size for StepLR')
        parser.add_argument('--gamma', type=float, default=0.1, help='gamma for StepLR')
        # ReduceLROnPlateau scheduler args
        parser.add_argument('--scheduler_mode', default='min', help='mode for ReduceLROnPlateau (min/max)')
        parser.add_argument('--scheduler_factor', type=float, default=0.3, help='factor for ReduceLROnPlateau')
        parser.add_argument('--scheduler_patience', type=int, default=7, help='patience for ReduceLROnPlateau')
        parser.add_argument('--scheduler_threshold', type=float, default=1e-4, help='threshold for ReduceLROnPlateau')
        parser.add_argument('--scheduler_min_lr', type=float, default=1e-7, help='min_lr for ReduceLROnPlateau')
        parser.add_argument('--scheduler_verbose', type=str2bool, default=True, help='verbose for ReduceLROnPlateau')
        # warmup
        parser.add_argument('--warmup', type=str2bool, default=False, help='use warmup learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=8, help='number of warmup epochs')
        parser.add_argument('--warmup_start_lr', type=float, default=1e-6, help='starting learning rate for warmup')
        # early stopping
        parser.add_argument('--early_stop_patience', type=int, default=None, help='patience for early stopping')
        # class weight
        parser.add_argument('--use_class_weight', type=str2bool, default=False, help='use class-balanced weights for CrossEntropyLoss')
        # endregion yapf: enable

        return parser