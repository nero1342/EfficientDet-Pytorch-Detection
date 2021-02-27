import torch
from torch import nn, optim 
from torch.utils import data 
from torchnet import meter 
from tqdm import tqdm 
import numpy as np 
import os 
from datetime import datetime 
import traceback 

from loggers import TensorboardLogger

class Trainer():
    def __init__(self, 
                device, 
                config,
                model, 
                criterion,
                optimizer,
                scheduler,
                metric
                ):
        super(Trainer, self).__init__() 

        self.config = config 
        self.device = device
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric 

        self.train_ID = self.config.get('id', None)
        self.train_ID += '-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S') 

        self.nepochs = self.config['trainer']['nepochs']
        self.log_step = self.config['trainer']['log_step']
        self.val_step = self.config['trainer']['val_step']
        
        self.best_loss = np.inf 
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.val_loss = list() 
        self.val_metric = {k: list() for k in self.metric.keys() }

        self.save_dir = os.path.join(config['trainer']['save_dir'], self.train_ID)
        self.tsboard = TensorboardLogger(path=self.save_dir) 

    def save_checkpoint(self, epoch, val_loss, val_metric):
        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        if val_loss < self.best_loss:
            print(
                f'Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights...')
            torch.save(data, os.path.join(self.save_dir, 'best_loss.pth'))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print(f'Loss is not improved from {self.best_loss:.6f}.')

        # for k in self.metric.keys():
        #    if val_metric[k] > self.best_metric[k]:
        #        print(
        #            f'{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights...')
        #        torch.save(data, os.path.join(
        #            self.save_dir, f'best_metric_{k}.pth'))
        #        self.best_metric[k] = val_metric[k]
        #    else:
        #        print(
        #            f'{k} is not improved from {self.best_metric[k]:.6f}.')

        # print('Saving current model...')
        # torch.save(data, os.path.join(self.save_dir, 'current.pth'))    

    def train_epoch(self, epoch, dataloader):
        total_loss = meter.AverageValueMeter() 

        self.model.train()
        print("Training..........")
        progress_bar = tqdm(dataloader)
        max_iter = len(dataloader)
        for i, data in enumerate(progress_bar):
            # progress_bar.update() 
            
            try:
                imgs = data['img']
                ann = data['annot']
                
                imgs = imgs.cuda() 
                ann = ann.cuda() 

                self.optimizer.zero_grad() 
                _, regression, classification, anchors = self.model(imgs)
                cls_loss, reg_loss = self.criterion(classification, regression, anchors, ann)

                cls_loss = cls_loss.mean() 
                reg_loss = reg_loss.mean() 

                loss = cls_loss + reg_loss

		if loss == 0 or not torch.isfinite(loss):
			continue
		total_loss.add(loss.item()) 
                loss.backward() 

                self.optimizer.step() 

                with torch.no_grad():
                    progress_bar.set_description(
                        'Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            i + 1, max_iter, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    
                    self.tsboard.update_scalar(
                        'Loss - train', loss, epoch * len(dataloader) + i 
                    )
                    self.tsboard.update_scalar(
                        'Regression Loss - train', reg_loss, epoch * len(dataloader) + i 
                    )
                    self.tsboard.update_scalar(
                        'Classification Loss - train', cls_loss, epoch * len(dataloader) + i 
                    )

                
            except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
               
                
        print("+ Train result")
        avg_loss = total_loss.value()[0]
        print("Loss:", avg_loss)
        # for m in self.metric.values():
        #     m.summary() 

    @torch.no_grad() 
    def val_epoch(self, epoch, dataloader):
        cls_loss_lst = meter.AverageValueMeter() 
        reg_loss_lst = meter.AverageValueMeter() 
        # for ,
        self.model.eval() 
        print("Evaluating.....")
        progress_bar = tqdm(dataloader)
        # cls_loss
        for i, data in enumerate(progress_bar):
            # progress_bar.update() 
            
            try:
                imgs = data['img']
                ann = data['annot']

                imgs = imgs.cuda() 
                ann = ann.cuda() 

                self.optimizer.zero_grad() 
                _, regression, classification, anchors = self.model(imgs)
                cls_loss, reg_loss = self.criterion(classification, regression, anchors, ann)

                cls_loss = cls_loss.mean() 
                reg_loss = reg_loss.mean() 

                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue 
                
                cls_loss_lst.add(cls_loss.item())
                reg_loss_lst.add(reg_loss.item()) 
            except:
                pass 
        print("+ Evaluation result")
        avg_cls_loss = cls_loss_lst.value()[0]
        avg_reg_loss = reg_loss_lst.value()[0]
        avg_loss = avg_cls_loss + avg_reg_loss
        self.val_loss.append(avg_loss)
        
        self.tsboard.update_scalar(
            'Loss - val', avg_loss, epoch
        )
        self.tsboard.update_scalar(
            'Regression Loss - val', avg_reg_loss, epoch 
        )
        self.tsboard.update_scalar(
            'Classification Loss - val', avg_cls_loss, epoch 
        )
                        
        # Calculate metric here

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.nepochs):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            # Note learning rate
            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group['lr'], epoch)

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)

            print()

            # 2: Evalutation phase
            if (epoch + 1) % self.val_step == 0:
                # 2: Evaluating model
                self.val_epoch(epoch, dataloader=val_dataloader)
                print('-----------------------------------')

                # 3: Learning rate scheduling
                self.scheduler.step(self.val_loss[-1])

                # 4: Saving checkpoints
                if True:
                    # Get latest val loss here
                    val_loss = self.val_loss[-1]
                    val_metric = None # {k: m[-1] for k, m in self.val_metric.items()}
                    self.save_checkpoint(epoch, val_loss, val_metric)
