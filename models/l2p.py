import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import copy 

torch.multiprocessing.set_sharing_strategy('file_system')

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = PromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["arc_batch_size"], shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._network = self._network.module
        self._train(self.train_loader, self.test_loader)
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)
        

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()
            
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

                with torch.no_grad():
                    model.prompt.prompt.grad.zero_()
                    model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()
                
        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.prompt.prompt_key.grad.zero_()
                model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
    
    def task_based_softmax(self, logits, increment, task_ids):
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        # Create a mask for selecting the relevant classes per sample
        class_indices = torch.arange(num_classes, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        start_indices = (task_ids * increment).unsqueeze(1)
        end_indices = ((task_ids + 1) * increment).unsqueeze(1)
        task_mask = (class_indices >= start_indices) & (class_indices < end_indices)

        masked_logits = logits.clone()
        masked_logits[~task_mask] = float('-inf')

        max_logits, _ = torch.max(masked_logits, dim=1)
        sum_of_exps = torch.exp(logits).sum(dim=1)
        prob = torch.exp(max_logits) / sum_of_exps

        return prob
    
    def soften_multiclass_labels(self, labels, num_classes, soften_factor=0.1):
        soft_labels = torch.full((labels.size(0), num_classes), soften_factor / (num_classes - 1))
        hard_indices = torch.arange(labels.size(0)), labels
        soft_labels[hard_indices] = 1 - soften_factor
        return soft_labels

    def entropy_loss(self, predictions):
        p_softmax = F.softmax(predictions, dim=1)
        log_p_softmax = F.log_softmax(predictions, dim=1)
        batch_entropy = -torch.sum(p_softmax * log_p_softmax, dim=1)
        return batch_entropy.mean()  # Average entropy across the batch

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        increment = self.args["increment"]
        iterative_temperature = self.args["iterative_temperature"]

        if self.args["adaptive_retention"]:
            for p in self._network.backbone.head.parameters():
                p.requires_grad=True

            for p in self._network.backbone.prompt.parameters():
                p.requires_grad=True

            criterion = nn.CrossEntropyLoss()
             
            if self.args["train_prompt"]:
                optimizer = optim.Adam([
                    {'params': self._network.backbone.head.parameters(), 'lr': self.args["adjust_head_lr"]}, 
                    {'params': self._network.backbone.prompt.parameters(), 'lr': self.args["adjust_prompt_lr"]} 
                ], weight_decay=self.weight_decay)
            else:
                optimizer = optim.Adam([
                    {'params': self._network.backbone.head.parameters(), 'lr': self.args["adjust_head_lr"]}, 
                ], weight_decay=self.weight_decay)
            

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            
            outputs = self._network(inputs)['logits'][:, :self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

            if self.args["adaptive_retention"]:
                predicts_task_id = predicts[:, 0] // increment
                confidence, _ = torch.max(F.softmax(outputs, dim=1), dim=1)

                # Create a mask for samples exceeding the confidence threshold
                mask = confidence > self.args["confidence_threshold"]

                if mask.any():
                    outputs_masked = outputs[mask]
                    pseudo_labels = predicts[mask][:, 0]
                    if self.args["soften_labels"]:
                        pseudo_labels = self.soften_multiclass_labels(pseudo_labels, (self._cur_task+1) * increment)
                        pseudo_labels = pseudo_labels.to(self._device)
                    
                    optimizer.zero_grad()

                    if not self.args["entropy_loss"]:
                        loss = criterion(outputs_masked, pseudo_labels)
                    elif not self.args["cross_entropy_loss"]:
                        loss = self.entropy_loss(outputs_masked)
                    else:
                        loss = criterion(outputs_masked, pseudo_labels) + self.entropy_loss(outputs_masked)

                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                if self.args["adaptive_correction"] and self._cur_task > 0:
                    top_predictions = predicts[:, 0]
                    predicted_task_ids = top_predictions // increment
                    candidates_mask = (predicted_task_ids == self._cur_task)

                    if candidates_mask.any():
                        candidate_logits = outputs[candidates_mask]
                        current_task_ids = torch.full((candidate_logits.size(0),), self._cur_task, dtype=torch.long, device=outputs.device)

                        prob_in_current_task = self.task_based_softmax(candidate_logits, increment, current_task_ids)

                        prev_tasks_logits = candidate_logits[:, :self._cur_task * increment]
                        max_logits_prev, _ = torch.max(prev_tasks_logits, dim=1)
                        sum_of_exps_prev = torch.exp(prev_tasks_logits).sum(dim=1)
                        prob_in_prev_tasks = torch.exp(max_logits_prev) / sum_of_exps_prev

                        correction_needed_mask = (prob_in_current_task / prob_in_prev_tasks) < self.args["ood_threshold"]

                        if correction_needed_mask.any():
                            logits_to_correct = candidate_logits[correction_needed_mask]
                            task_ids_to_correct = current_task_ids[correction_needed_mask]

                            max_task_id = task_ids_to_correct
                            temp_task_id = task_ids_to_correct
                            max_prob = self.task_based_softmax(logits_to_correct, increment, max_task_id)

                            logits_loop = logits_to_correct.clone()
                            
                            while logits_loop.size(1) > increment:
                                temp_task_id = temp_task_id - 1
                                logits_loop = logits_loop[:, :-increment] / iterative_temperature
                                
                                cur_max_prob = self.task_based_softmax(logits_loop, increment, temp_task_id)

                                update_mask = cur_max_prob > max_prob
                                
                                max_task_id = torch.where(update_mask, temp_task_id, max_task_id)
                                max_prob = torch.where(update_mask, cur_max_prob, max_prob)

                            max_classes = (max_task_id + 1) * increment
                            class_indices = torch.arange(logits_to_correct.size(1), device=logits_to_correct.device).unsqueeze(0)
                    
                            class_mask = class_indices < max_classes.unsqueeze(1)
                            outputs_mod = torch.where(class_mask, logits_to_correct, float('-inf'))
                            corrected_predictions = torch.topk(outputs_mod, k=self.topk, dim=1, largest=True, sorted=True)[1]

                            original_batch_indices = candidates_mask.nonzero(as_tuple=True)[0]
                            indices_to_update_in_batch = original_batch_indices[correction_needed_mask]
                            predicts[indices_to_update_in_batch] = corrected_predictions

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)