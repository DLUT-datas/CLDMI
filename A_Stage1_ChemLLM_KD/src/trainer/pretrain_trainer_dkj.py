import torch
import numpy as np
from sklearn.metrics import f1_score
import time, datetime
class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, reg_evaluator, clf_evaluator, result_tracker, summary_writer, device, ddp=False, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.sl_loss_fn = sl_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator = clf_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0
    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        sl_predictions, fp_predictions, md_predictions = model(batched_graph, disturbed_fps, disturbed_mds)
        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask']>=1].cpu().numpy()
        return mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds = self._forward_epoch(model, batched_data)
                sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
                fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
                md_loss = self.reg_loss_fn(md_predictions, mds).mean()
                loss = (sl_loss + fp_loss + md_loss)/3
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.lr_scheduler.step()
                # if self.summary_writer is not None:
                #     loss_mask = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==1],sl_labels.detach().cpu()[mask_replace_keep==1]).mean()
                #     loss_replace = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==2],sl_labels.detach().cpu()[mask_replace_keep==2]).mean()
                #     loss_keep = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==3],sl_labels.detach().cpu()[mask_replace_keep==3]).mean()
                #     preds = np.argmax(sl_predictions.detach().cpu().numpy(),axis=-1)
                #     labels = sl_labels.detach().cpu().numpy()
                #     self.summary_writer.add_scalar('Loss/loss_tot', loss.item(), self.n_updates)
                #     self.summary_writer.add_scalar('Loss/loss_bert', sl_loss.item(), self.n_updates)
                #     self.summary_writer.add_scalar('Loss/loss_mask', loss_mask.item(), self.n_updates)
                #     self.summary_writer.add_scalar('Loss/loss_replace', loss_replace.item(), self.n_updates)
                #     self.summary_writer.add_scalar('Loss/loss_keep', loss_keep.item(), self.n_updates)
                #     self.summary_writer.add_scalar('Loss/loss_clf', fp_loss.item(), self.n_updates)
                #     self.summary_writer.add_scalar('Loss/loss_reg', md_loss.item(), self.n_updates)
                    
                #     self.summary_writer.add_scalar('F1_micro/all', f1_score(preds, labels, average='micro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_macro/all', f1_score(preds, labels, average='macro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_micro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='micro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_macro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='macro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_micro/replace', f1_score(preds[mask_replace_keep==2], labels[mask_replace_keep==2], average='micro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_macro/replace', f1_score(preds[mask_replace_keep==2], labels[mask_replace_keep==2], average='macro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_micro/keep', f1_score(preds[mask_replace_keep==3], labels[mask_replace_keep==3], average='micro'), self.n_updates)
                #     self.summary_writer.add_scalar('F1_macro/keep', f1_score(preds[mask_replace_keep==3], labels[mask_replace_keep==3], average='macro'), self.n_updates)
                #     self.summary_writer.add_scalar(f'Clf/{self.clf_evaluator.eval_metric}_all', np.mean(self.clf_evaluator.eval(fps, fp_predictions)), self.n_updates)
                if self.n_updates == self.args.n_steps:
                    if self.local_rank == 0:
                        self.save_model(model)
                    break

                # # 记录每个epoch下预训练模型在验证数据上的loss
                results_dir = '../pretrain_loss'
                os.makedirs(results_dir, exist_ok=True) 

                # 获取当前日期
                current_date = datetime.date.today()
                current_time = time.strftime("%H:%M:%S")
                combined = f"{current_date} {current_time}"
                
                results.append(['epochs', epoch_idx, 'date', combined, 'train_step', batch_idx, \
                                'train_loss', "%.16f" %loss, 'sl_loss', "%.16f" % sl_loss, 'fp_loss', "%.16f" % fp_loss, 'md_loss', "%.16f" % md_loss])
                                # 'train_loss', loss, 'sl_loss', sl_loss, 'fp_loss', fp_loss, 'md_loss', md_loss])
                                # 'val_step_num', val_step_num, 
                
                if batch_idx % 100 == 0 or batch_idx == len(train_loader)-1:
                    df = pd.DataFrame(results)
                    df.to_csv(
                        results_dir + '/{}_{}.csv'.format('pretrain_train_loss', '20251027_rdkf_fp16'),
                        mode='a', index=False, header=False
                    )
                    results = []

            except Exception as e:
                print(e)
            else:
                continue

    def validation_epoch(self, model, validation_loader, epoch_idx):

        #-----------------------------------------------
        model.eval()
        # 验证预训练模型  不计算梯度，不更新参数
        val_epoch_total_loss = 0.0
        val_single_result=[]
        results_dir1 = '../pretrain_loss'
        os.makedirs(results_dir1, exist_ok=True) 

        with torch.no_grad():
            for batch_idx, batched_data in enumerate(validation_loader):
                mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds = self._forward_epoch(model, batched_data)              
                sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
                fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
                md_loss = self.reg_loss_fn(md_predictions, mds).mean()
                # loss = (sl_loss + fp_loss + md_loss)/3
                loss = sl_loss + fp_loss + md_loss
                val_epoch_total_loss += loss.item()
                    
            self.save_model(model, epoch_idx)
            #-----------------------------------------------
            val_single_result.append(['epochs', epoch_idx, 'val_loss', "%.16f" % val_epoch_total_loss, 'sl_loss', "%.16f" % sl_loss, 'fp_loss', "%.16f" % fp_loss, 'md_loss', "%.16f" % md_loss])
            df = pd.DataFrame(val_single_result)
            df.to_csv(
                results_dir1 + '/{}_{}.csv'.format('pretrain_val_loss', '20251027_rdkf_fp16'),
                mode='a', index=False, header=False
            )
            #-----------------------------------------------
        #-----------------------------------------------

    def fit(self, model, train_loader):
        for epoch in range(1, 1001):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.n_updates >= self.args.n_steps:
                break
            model.eval()
            self.validation_epoch(model, validation_loader, epoch)

    # def save_model(self, model):
    #     torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}.pth")

    def save_model(self, model, epoch=None):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        if epoch==None:
            torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}.pth")
        else:
            torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}_{epoch}.pth")