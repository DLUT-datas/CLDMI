import torch
from torch import nn
import numpy as np
import pdb
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy
from src.utils import set_random_seed
from src.model_config import config_dict

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.config = config_dict[args.lgt_config]
            
    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)
        predictions, emb = model.forward_tune(g, ecfp, md)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        if self.args.eval_kelgt == False: # 从头训练或微调
            for epoch in range(1, self.args.lgt_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(model, train_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(model, test_loader)
                    train_result, train_predictions, train_labels = self.eval(model, train_loader)
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result))

                        # save best model with best_test_result
                        torch.save(model.state_dict(), self.args.model_save_path + '/' + 'kelgt_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())

                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)

                        df_predictions.to_csv(self.args.model_save_path + '/' + 'kelgt_prediction4test.csv', index=False) 
                        df_labels.to_csv(self.args.model_save_path + '/' + 'kelgt_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break

            model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'kelgt_model.pth'))

        else:
            # # 加载独立微调好的kelgt模型
            model.load_state_dict(torch.load(self.args.kelgt_model_save_path + '/' + 'kelgt_model.pth'))

        val_result, val_predictions, val_labels = self.eval(model, val_loader)
        test_result, test_predictions, test_labels = self.eval(model, test_loader)
        test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())
        print("val_result:", val_result, "test_result:", test_result)

        return np.mean(best_train_result), np.mean(val_result), np.mean(test_result)

    
    def fit_f(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        
        if self.args.eval_kelgt == False: # 从头训练或微调
            for epoch in range(1, self.args.lgt_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(model, train_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(model, test_loader)
                    train_result, train_predictions, train_labels = self.eval(model, train_loader)
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result))

                        # save best model with best_test_result
                        torch.save(model.state_dict(), self.args.model_save_path + '/' + 'kelgt_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())

                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)

                        df_predictions.to_csv(self.args.model_save_path + '/' + 'kelgt_prediction4test.csv', index=False) 
                        df_labels.to_csv(self.args.model_save_path + '/' + 'kelgt_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break

            model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'kelgt_model.pth'))

        else:
            # # 加载独立微调好的kelgt模型
            self.args.lgt_model_has_saved_path ='./result_finetuned_kelgt_HSSL/' \
                + self.args.dataset +'_'+ str(self.config['predict_drop']) + '/' + str(self.args.seed) +'/kelgt_model.pth'
            model.load_state_dict(torch.load(self.args.lgt_model_has_saved_path))

        val_result, val_predictions, val_labels = self.eval(model, val_loader)
        test_result, test_predictions, test_labels = self.eval(model, test_loader)


        if self.args.dataset_type == 'regression':
            l_std = self.label_std.detach().cpu()
            l_mean = self.label_mean.detach().cpu()
            val_predictions = torch.cat(val_predictions) * l_std + l_mean
            test_predictions = torch.cat(test_predictions) * l_std + l_mean
            print("val_predictions:", type(val_predictions))

        else:
            val_predictions = torch.cat(val_predictions) 
            test_predictions = torch.cat(test_predictions) 

        # test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())
        print("val_result:", val_result, "test_result:", test_result)


        # return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result)
        return val_predictions, np.mean(best_val_result), np.mean(test_result), test_predictions, test_labels

        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result), test_predictions, test_labels
    
    def eval(self, model, dataloader):
        model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())

        # print("11111:", type(torch.cat(labels_all)), type(torch.cat(predictions_all)))
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        # return result
        return result, predictions_all, labels_all
    
    def final_eval(self, grap_model, kelgrap_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()

        predictions_1, predictions_2, predictions_12avg = [], [], []
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            (smiles, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(self.device)
            md = md.to(self.device)
            g = g.to(self.device)
            labels = labels.to(self.device)
            # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


            p_graph, graph_emb = grap_model.forward(smiles)
            p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)

            predictions_1.append(p_graph)
            predictions_2.append(p_kelgraph)
            predictions_12avg.append((p_graph+p_kelgraph)/2.0)

            labels_all.append(labels)

            # predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            # predictions_all.append(predictions.detach().cpu())
            # labels_all.append(labels.detach().cpu())
        p_graph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_1))
        p_kelgraph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_2))
        f_12_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_12avg))


        # return result
        return p_graph_score, p_kelgraph_score, f_12_score
    



class Trainer_joint_mmpp():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        self.config = config_dict[args.lgt_config]

        self.optimizer_graph_module = optimizer['graph_module']
        self.optimizer_linegraph_module = optimizer['linegraph_module']
        self.optimizer_fusion_module = optimizer['fusion_module']

        self.lr_scheduler_graph_module = lr_scheduler['graph_module']
        self.lr_scheduler_linegraph_module = lr_scheduler['linegraph_module']
        self.lr_scheduler_fusion_module = lr_scheduler['fusion_module']
            
    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        # print("model:", model)
        predictions, emb, _, _ = model(smiles, g, ecfp, md)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step()
                self.lr_scheduler_graph_module.step()
                self.lr_scheduler_linegraph_module.step()
                # self.lr_scheduler_fusion_module.step()

            self.optimizer_graph_module.zero_grad()
            self.optimizer_linegraph_module.zero_grad()
            self.optimizer_fusion_module.zero_grad()

            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.graph_module.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(model.linegraph_module.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(model.fusion_module.parameters(), 5)

            self.optimizer_graph_module.step()
            self.optimizer_linegraph_module.step()
            self.optimizer_fusion_module.step()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)

        self.lr_scheduler_fusion_module.step()


    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        # torch.save(model.state_dict(), '../random_inti/kelgt_model.pth')
        
        flag_lgt = False # 加载微调好的模型文件
        # flag_lgt = True # 从头训练或微调
        if flag_lgt:
            for epoch in range(1, self.args.lgt_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(model, train_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(model, test_loader)
                    train_result, train_predictions, train_labels = self.eval(model, train_loader)
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result))

                        # save best model with best_test_result
                        torch.save(model.state_dict(), self.args.model_save_path + '/' + 'kelgt_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())


                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)
                        # if self.args.task_names is not None:
                        #     df_predictions.columns = self.args.task_names
                        #     df_labels.columns = self.args.task_names

                        df_predictions.to_csv(self.args.model_save_path + '/' + 'kelgt_prediction4test.csv', index=False) 
                        df_labels.to_csv(self.args.model_save_path + '/' + 'kelgt_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break

            model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'kelgt_model.pth'))

        else:
            # # 加载单独的kelgt目录
            model.load_state_dict(torch.load(self.args.kelgt_model_save_path + '/' + 'kelgt_model.pth'))
        val_result, val_predictions, val_labels = self.eval(model, val_loader)
        test_result, test_predictions, test_labels = self.eval(model, test_loader)
        test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())
        print("val_result:", val_result, "test_result:", test_result)

        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result)
        # return val_predictions, np.mean(best_val_result), np.mean(test_result), test_predictions, test_labels
    
    def fit_f(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        # torch.save(model.state_dict(), '../random_inti/kelgt_model.pth')
        
        # flag = False
        flag = True
        if flag:
            for epoch in range(1, self.args.lgt_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(model, train_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(model, test_loader)
                    train_result, train_predictions, train_labels = self.eval(model, train_loader)
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result))

                        # save best model with best_test_result
                        # torch.save(model.state_dict(), self.args.model_save_path + '/' + 'kelgt_model.pth')
                        torch.save({
                            "graph_module": model.graph_module.state_dict(),
                            "linegraph_module": model.linegraph_module.state_dict(),
                            "fusion_module": model.fusion_module.state_dict(),
                            "epoch": epoch,
                        }, self.args.model_save_path + '/' + 'joint_finetune_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())


                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)
                        # if self.args.task_names is not None:
                        #     df_predictions.columns = self.args.task_names
                        #     df_labels.columns = self.args.task_names

                        df_predictions.to_csv(self.args.model_save_path + '/' + 'joint_finetune_prediction4test.csv', index=False) 
                        df_labels.to_csv(self.args.model_save_path + '/' + 'kelgt_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break

            # model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'kelgt_model.pth'))

        else:
            # # # 加载独立微调好的kelgt模型
            # self.args.lgt_model_has_saved_path ='./results_finetune_kelgt_concate_fusion_dkj_Pret_0130_SBCE_only_w_sa_b256_e20/' \
            #     + self.args.dataset +'_'+ str(self.config['predict_drop']) + '/' + str(self.args.seed) +'/kelgt_model.pth'
            # model.load_state_dict(torch.load(self.args.lgt_model_has_saved_path))

            checkpoint_here = torch.load(self.args.model_save_path + '/' + 'joint_finetune_model.pth', map_location=self.device)

            model.graph_module.load_state_dict(
                checkpoint_here["graph_module"]
            )

            model.linegraph_module.load_state_dict(
                checkpoint_here["linegraph_module"]
            )

            model.fusion_module.load_state_dict(
                checkpoint_here["fusion_module"]
            )


        val_result, val_predictions, val_labels = self.eval(model, val_loader)
        test_result, test_predictions, test_labels = self.eval(model, test_loader)


        if self.args.dataset_type == 'regression':
            l_std = self.label_std.detach().cpu()
            l_mean = self.label_mean.detach().cpu()
            val_predictions = torch.cat(val_predictions) * l_std + l_mean
            test_predictions = torch.cat(test_predictions) * l_std + l_mean
            print("val_predictions:", type(val_predictions))

        else:
            val_predictions = torch.cat(val_predictions) 
            test_predictions = torch.cat(test_predictions) 

        # test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())
        print("val_result:", val_result, "test_result:", test_result)


        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result)
        return val_predictions, np.mean(best_val_result), np.mean(test_result), test_predictions, test_labels

        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result), test_predictions, test_labels
    
    def eval(self, model, dataloader):
        model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())

        # print("11111:", type(torch.cat(labels_all)), type(torch.cat(predictions_all)))
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        # return result
        return result, predictions_all, labels_all
    
    def final_eval(self, grap_model, kelgrap_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()

        predictions_1, predictions_2, predictions_12avg = [], [], []
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            (smiles, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(self.device)
            md = md.to(self.device)
            g = g.to(self.device)
            labels = labels.to(self.device)
            # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


            p_graph, graph_emb = grap_model.forward(smiles)
            p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)

            predictions_1.append(p_graph)
            predictions_2.append(p_kelgraph)
            predictions_12avg.append((p_graph+p_kelgraph)/2.0)

            labels_all.append(labels)

            # predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            # predictions_all.append(predictions.detach().cpu())
            # labels_all.append(labels.detach().cpu())
        p_graph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_1))
        p_kelgraph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_2))
        f_12_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_12avg))


        # return result
        return p_graph_score, p_kelgraph_score, f_12_score
    



class Trainer4fusion():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, grap_model, kelgrap_model, bidir_fusion_model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


        p_graph, graph_emb = grap_model.forward(smiles)
        p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)
        p_fusion, emb = bidir_fusion_model(graph_emb, kelgrap_emb)
        return p_fusion, labels

    def train_epoch(self, grap_model, kelgrap_model, bidir_fusion_model, train_loader, epoch_idx):
        
        grap_model.eval()
        kelgrap_model.eval()
        # grap_model.train()
        # kelgrap_model.train()
        bidir_fusion_model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bidir_fusion_model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)


    def fit(self, grap_model, kelgrap_model, bidir_fusion_model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        # torch.save(model.state_dict(), '../random_inti/kelgt_model.pth')

        for param in grap_model.parameters():
            param.requires_grad = False
        for param in kelgrap_model.parameters():
            param.requires_grad = False

        # flag_lgt = False
        flag_lgt = True
        if flag_lgt:
            for epoch in range(1, self.args.fusion_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(grap_model, kelgrap_model, bidir_fusion_model, train_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, test_loader)
                    train_result, train_predictions, train_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, train_loader)
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result), "best_test_score:", np.mean(best_test_result))

                        # save best model with best_test_result
                        torch.save(bidir_fusion_model.state_dict(), self.args.model_save_path + '/' + 'bidir_fusion_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())


                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)
                        if self.args.task_names is not None:
                            df_predictions.columns = self.args.task_names
                            df_labels.columns = self.args.task_names

                        df_predictions.to_csv(self.args.model_save_path + '/' + 'bidir_fusion_prediction4test.csv', index=False) 
                        df_labels.to_csv(self.args.model_save_path + '/' + 'bidir_fusion_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break
                
        bidir_fusion_model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'bidir_fusion_model.pth'))
        test_result, test_predictions, test_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, test_loader)
        test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())

        # return np.mean(best_train_result), np.mean(best_val_result), np.mean(test_result), test_predictions, test_labels

        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result), test_predictions, test_labels
    
    def eval(self, grap_model, kelgrap_model, bidir_fusion_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        # return result
        return result, predictions_all, labels_all
    

    def final_eval(self, grap_model, kelgrap_model, bidir_fusion_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()

        predictions_1, predictions_2, predictions_3, predictions_12avg, predictions_123avg = [], [], [], [], []
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            (smiles, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(self.device)
            md = md.to(self.device)
            g = g.to(self.device)
            labels = labels.to(self.device)
            # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


            p_graph, graph_emb = grap_model.forward(smiles)
            p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)
            p_fusion, emb = bidir_fusion_model(graph_emb, kelgrap_emb)

            predictions_1.append(p_graph)
            predictions_2.append(p_kelgraph)
            predictions_3.append(p_fusion)
            predictions_12avg.append((p_graph+p_kelgraph)/2.0)
            predictions_123avg.append((p_graph+p_kelgraph+p_fusion)/3.0)
            labels_all.append(labels)

            # predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            # predictions_all.append(predictions.detach().cpu())
            # labels_all.append(labels.detach().cpu())
        p_graph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_1))
        p_kelgraph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_2))
        p_gfusion_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_3))
        f_12_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_12avg))
        f_123_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_123avg))

        # return result
        return p_graph_score, p_kelgraph_score, p_gfusion_score, f_12_score, f_123_score
    

class Trainer4MOE():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, grap_model, kelgrap_model, bidir_fusion_model, moe_model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


        p_graph, graph_emb = grap_model.forward(smiles)
        p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)
        p_fusion, emb = bidir_fusion_model(graph_emb, kelgrap_emb)

        p_out = torch.cat([p_graph, p_kelgraph], dim=1)
        print("p_out:", p_out.size())
        weights = moe_model(p_out)
        print("weights:", weights)

        # p = p_graph*weights[:,0:1] + p_kelgraph*weights[:,1:2] + p_fusion*weights[:,2:3]
        p = p_graph*weights + p_kelgraph*(1-weights)
        return p, labels

    def train_epoch(self, grap_model, kelgrap_model, bidir_fusion_model, moe_model, val_loader, epoch_idx):
        
        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()
        moe_model.train()
        for batch_idx, batched_data in enumerate(val_loader):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, moe_model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bidir_fusion_model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(val_loader)+batch_idx+1)


    def fit(self, grap_model, kelgrap_model, bidir_fusion_model, moe_model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        # torch.save(model.state_dict(), '../random_inti/kelgt_model.pth')

        for param in grap_model.parameters():
            param.requires_grad = False
        for param in kelgrap_model.parameters():
            param.requires_grad = False
        for param in bidir_fusion_model.parameters():
            param.requires_grad = False

        # flag_lgt = False
        flag_lgt = True
        if flag_lgt:
            for epoch in range(1, self.args.fusion_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(grap_model, kelgrap_model, bidir_fusion_model, moe_model, val_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, moe_model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, moe_model, test_loader)
    
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result), "best_test_score:", np.mean(best_test_result))

                        # save best model with best_test_result
                        torch.save(moe_model.state_dict(), self.args.model_save_path + '/' + 'moe_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())


                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)
                        if self.args.task_names is not None:
                            df_predictions.columns = self.args.task_names
                            df_labels.columns = self.args.task_names

                        # df_predictions.to_csv(self.args.model_save_path + '/' + 'moe_prediction4test.csv', index=False) 
                        df_labels.to_csv(self.args.model_save_path + '/' + 'moe_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break
                
        moe_model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'moe_model.pth'))
        test_result, test_predictions, test_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, moe_model, test_loader)
        test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())

        # return np.mean(best_train_result), np.mean(best_val_result), np.mean(test_result), test_predictions, test_labels

        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result), test_predictions, test_labels
    
    def eval(self, grap_model, kelgrap_model, bidir_fusion_model, moe_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()
        moe_model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, moe_model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        # return result
        return result, predictions_all, labels_all
    

    def final_eval(self, grap_model, kelgrap_model, bidir_fusion_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()

        predictions_1, predictions_2, predictions_3, predictions_12avg, predictions_123avg = [], [], [], [], []
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            (smiles, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(self.device)
            md = md.to(self.device)
            g = g.to(self.device)
            labels = labels.to(self.device)
            # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


            p_graph, graph_emb = grap_model.forward(smiles)
            p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)
            p_fusion, emb = bidir_fusion_model(graph_emb, kelgrap_emb)

            predictions_1.append(p_graph)
            predictions_2.append(p_kelgraph)
            predictions_3.append(p_fusion)
            predictions_12avg.append((p_graph+p_kelgraph)/2.0)
            predictions_123avg.append((p_graph+p_kelgraph+p_fusion)/3.0)
            labels_all.append(labels)

            # predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            # predictions_all.append(predictions.detach().cpu())
            # labels_all.append(labels.detach().cpu())
        p_graph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_1))
        p_kelgraph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_2))
        p_gfusion_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_3))
        f_12_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_12avg))
        f_123_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_123avg))

        # return result
        return p_graph_score, p_kelgraph_score, p_gfusion_score, f_12_score, f_123_score
    


class Trainer4fusion1234():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, grap_model, kelgrap_model, bidir_fusion_model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


        p_graph, graph_emb = grap_model.forward(smiles)
        p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)
        p_fusion, emb = bidir_fusion_model(graph_emb, kelgrap_emb)
        return p_fusion, labels

    def train_epoch(self, grap_model, kelgrap_model, bidir_fusion_model, train_loader, epoch_idx):
        
        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bidir_fusion_model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)


    def fit(self, grap_model, kelgrap_model, bidir_fusion_model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        # torch.save(model.state_dict(), '../random_inti/kelgt_model.pth')

        for param in grap_model.parameters():
            param.requires_grad = False
        for param in kelgrap_model.parameters():
            param.requires_grad = False

        # flag_lgt = False
        flag_lgt = True
        if flag_lgt:
            for epoch in range(1, self.args.fusion_epochs+1):
                if self.ddp:
                    train_loader.sampler.set_epoch(epoch)
                self.train_epoch(grap_model, kelgrap_model, bidir_fusion_model, train_loader, epoch)
                if self.local_rank == 0:
                    # val_result = self.eval(model, val_loader)
                    # test_result = self.eval(model, test_loader)
                    # train_result = self.eval(model, train_loader)
                    val_result, val_predictions, val_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, val_loader)
                    test_result, test_predictions, test_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, test_loader)
                    train_result, train_predictions, train_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, train_loader)
                    if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                        best_val_result = val_result
                        best_test_result = test_result
                        best_train_result = train_result
                        best_epoch = epoch
                        best_test_predictions = test_predictions

                        print("current_epoch:", epoch, "best_val_score:", np.mean(best_val_result), "best_test_score:", np.mean(best_test_result))

                        # save best model with best_test_result
                        torch.save(bidir_fusion_model.state_dict(), self.args.model_save_path + '/' + 'bidir_fusion_model.pth')

                        test_predictions = np.array(torch.cat(best_test_predictions).detach().cpu())
                        test_labels = np.array(torch.cat(test_labels).detach().cpu())


                        df_predictions = pd.DataFrame(test_predictions)
                        df_labels = pd.DataFrame(test_labels)
                        if self.args.task_names is not None:
                            df_predictions.columns = self.args.task_names
                            df_labels.columns = self.args.task_names

                        df_predictions.to_csv(self.args.model_save_path + '/' + 'bidir_fusion_prediction4test.csv', index=False) 
                        # df_labels.to_csv(self.args.model_save_path + '/' + 'kelgt_label4test.csv', index=False) 

                    # print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                    if epoch - best_epoch >= 20:
                        break
                
        bidir_fusion_model.load_state_dict(torch.load(self.args.model_save_path + '/' + 'bidir_fusion_model.pth'))
        test_result, test_predictions, test_labels = self.eval(grap_model, kelgrap_model, bidir_fusion_model, test_loader)
        test_predictions = np.array(torch.cat(test_predictions).detach().cpu())
        test_labels = np.array(torch.cat(test_labels).detach().cpu())

        # return np.mean(best_train_result), np.mean(best_val_result), np.mean(test_result), test_predictions, test_labels

        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result), test_predictions, test_labels
    
    def eval(self, grap_model, kelgrap_model, bidir_fusion_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        # return result
        return result, predictions_all, labels_all
    

    def final_eval(self, grap_model, kelgrap_model, bidir_fusion_model, dataloader):

        grap_model.eval()
        kelgrap_model.eval()
        bidir_fusion_model.eval()

        predictions_1, predictions_2, predictions_3, predictions_12avg, predictions_123avg = [], [], [], [], []
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            (smiles, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(self.device)
            md = md.to(self.device)
            g = g.to(self.device)
            labels = labels.to(self.device)
            # predictions = model.forward_tune(g, ecfp, md, self.args.dataset_type)


            p_graph, graph_emb = grap_model.forward(smiles)
            p_kelgraph, kelgrap_emb = kelgrap_model.forward_tune(g, ecfp, md)
            p_fusion, emb = bidir_fusion_model(graph_emb, kelgrap_emb)

            predictions_1.append(p_graph)
            predictions_2.append(p_kelgraph)
            predictions_3.append(p_fusion)
            predictions_12avg.append((p_graph+p_kelgraph)/2.0)
            predictions_123avg.append((p_graph+p_kelgraph+p_fusion)/3.0)
            labels_all.append(labels)

            # predictions, labels = self._forward_epoch(grap_model, kelgrap_model, bidir_fusion_model, batched_data)
            # predictions_all.append(predictions.detach().cpu())
            # labels_all.append(labels.detach().cpu())
        p_graph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_1))
        p_kelgraph_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_2))
        p_gfusion_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_3))
        f_12_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_12avg))
        f_123_score = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_123avg))

        # return result
        return p_graph_score, p_kelgraph_score, p_gfusion_score, f_12_score, f_123_score