import glob
import logging
import os
import random
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.nn as nn
import random

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig ,BertForSequenceClassification, BertTokenizer
from model import bert
from tqdm import trange, tqdm

from utils import (convert_examples_to_features, processors)



def load_and_cache_examples(task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = args['output_mode']

    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(
        args['data_dir'],
        "cached_" + mode + "_" +args['model_name']+"_" + task)

    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(
            args['data_dir'])

        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
                                                cls_token_at_end=bool(args['model_type'] in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                                                pad_on_left=bool(args['model_type'] in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def create_dataset(features):
    logger.info("loading replay features")
    all_input_ids = torch.stack([f.input_ids for f in features])
    all_input_mask = torch.stack([f.input_mask for f in features])
    all_segment_ids = torch.stack([f.segment_ids for f in features])
    all_label_ids = torch.stack([f.label_id for f in features])
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def global_eval(task_list,global_model,fc_layers,tokenizer,output_rep=False,global_step=None):
    result = {}
    tmp = []
    class_cnt = []
    for name in task_list:
        tmp.append(fc_layers[name])
        class_cnt.append(fc_layers[name].weight.shape[0])
    fc_layers = tmp
    eval_output_dir = args['output_dir']
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    false_cnt = torch.zeros(sum(class_cnt))
    pre = 0
    for task_id,task_name in enumerate(task_list):
        result[task_name] = 0
        result['in_single_'+task_name] = 0;
        eval_dataset = load_and_cache_examples(task_name, tokenizer, evaluate=True)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])
        rep_df = pd.DataFrame()
        label_df = pd.DataFrame()
        get_loss = nn.CrossEntropyLoss()
        # Eval!
        logger.info("***** Running evaluation {} *****".format(task_name))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args['eval_batch_size'])
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            global_model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                          }
                labels = batch[3] + pre
                # print(labels)
                global_rep = global_model.bert(**inputs,return_dict = False)
                global_predict = []
                for fc_id,fc in enumerate(fc_layers):
                    logits = fc(global_rep[1])
                    logits /= torch.norm(fc.weight,dim=1)
                    global_predict.append(logits)
                    if(fc_id == task_id):result['in_single_'+task_name] += (torch.argmax(logits,1) == batch[3]).sum().item()
                global_predict = torch.cat(global_predict,1)
                global_predict = torch.argmax(global_predict,1)
                result[task_name] += (global_predict == labels).sum().item()
                false_predict = labels[global_predict != labels]
                
        pre += class_cnt[task_id]
        
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(task_name))
            result[task_name] = result[task_name] / len(eval_dataset)
            result[task_name+'_acc'] = result[task_name]
            result[task_name+'_single_acc'] = result['in_single_'+task_name] / len(eval_dataset)
            writer.write(task_name + "_acc = %s " % str(result[task_name+'_acc']))
            writer.write(task_name + "_single_acc = %s " % str(result[task_name+'_single_acc']))
            
    result['avg_acc'] = np.mean([
        v for k,v in result.items() if '_acc' in k and '_single_acc' not in k
    ])
    
    result['avg_single_acc'] = np.mean([
        v for k,v in result.items() if '_single_acc' in k
    ])
    
    return result



def train_global(task_name,global_model,origin_model,snap_shots,classifiers,tokenizer,global_step=0):

    train_dataset = load_and_cache_examples(task_name, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in origin_model.named_parameters() if (task_name in n)] + [p for p in classifiers[task_name].parameters()],
         'weight_decay': args['weight_decay'],
         'lr': args['learning_rate_snapshot']}
    ]
    
    global_model_parameters = [
        {'params': [p for n,p in global_model.named_parameters()],
         'weight_decay': args['weight_decay'],
         'lr': args['learning_rate_global']}
    ]
    global_optimizer = AdamW(
        global_model_parameters + optimizer_grouped_parameters, 
        lr=args['learning_rate_global'], eps=args['adam_epsilon'])# gloabl model opt
    
    scheduler = get_linear_schedule_with_warmup(
        global_optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total) # snap_shot opt
    scheduler.step()
    
    classifiers[task_name].weight.data /= torch.norm(classifiers[task_name].weight,dim=1,keepdim=True)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)
    
    
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    best = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            global_model.train()
            origin_model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert'] else None,
                    }
            labels = batch[3]
            snap_shot_rep_output = {}
            snap_shot_pseudo_label = {}
            global_predict_label = {}
            
            #train current snap-shot
            origin_model.bert.set_active_adapters(task_name)
            output = origin_model.bert(**inputs,return_dict = False)
            output = classifiers[task_name](output[1])
            output /= torch.norm(classifiers[task_name].weight,dim=1)
            cur_task_snap_shot_loss = ce_loss(output,labels)
            vari_classifier_loss = 0
            cur_classifier = classifiers[task_name]
            for snap_shot in snap_shots:
                if snap_shot == task_name: continue
                snap_shot_classifier = classifiers[snap_shot]
                vari_classifier_loss += torch.abs(cur_classifier.weight.mm(snap_shot_classifier.weight.T)).sum()
            if args['recalibrate']:
                cur_task_snap_shot_loss += vari_classifier_loss    
            
            cur_task_loss = 0.0
            reg_loss = 0.0
            global_rep = global_model.bert(**inputs,return_dict = False)
            for snap_shot in snap_shots:                    #calculate every snap_shot for 
                origin_model.bert.set_active_adapters(snap_shot)
                if snap_shot != task_name:
                    snap_shot_rep_output[snap_shot] = origin_model.bert(**inputs,return_dict = False)[1].detach() # detach to save memory
                else:
                    snap_shot_rep_output[snap_shot] = origin_model.bert(**inputs,return_dict = False)[1] # current snapshot needs updating
                snap_shot_pseudo_label[snap_shot] = torch.exp(classifiers[snap_shot](snap_shot_rep_output[snap_shot]) / args['T'])
                global_predict_label[snap_shot] = torch.exp(classifiers[snap_shot](global_rep[1]) / args['T'])
                if snap_shot == task_name:
                    cur_task_loss = mse_loss(global_rep[1],snap_shot_rep_output[snap_shot])
                    global_predict = classifiers[snap_shot](global_rep[1])
                    global_predict /= torch.norm(classifiers[snap_shot].weight,dim=1)
                    cur_task_loss += ce_loss(global_predict,labels)
                else:
                    reg_loss += mse_loss(global_predict_label[snap_shot],snap_shot_pseudo_label[snap_shot])
            global_model_loss = cur_task_loss + args['lambda'] * reg_loss \
                + cur_task_snap_shot_loss     # train global model
            global_model_loss.backward()
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), args['max_grad_norm'])
            torch.nn.utils.clip_grad_norm_(origin_model.parameters(), args['max_grad_norm'])
            torch.nn.utils.clip_grad_norm_(classifiers[task_name].parameters(), args['max_grad_norm'])
            global_optimizer.step()
            scheduler.step()
            global_model.zero_grad()
            classifiers[task_name].weight.data /= torch.norm(classifiers[task_name].weight,dim=1,keepdim=True)
            print("\r vari_classifier_loss = %f  cur_task_snap_shot_loss = %f cur_task_loss = %f reg_loss = %f" %(vari_classifier_loss,cur_task_snap_shot_loss,cur_task_loss,reg_loss), end='')

            tr_loss += global_model_loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                global_step += 1
                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = global_eval(
                            snap_shots,global_model,classifiers,tokenizer,
                            global_step=global_step,
                        )
                        if not os.path.exists(
                            os.path.join(args['output_dir'], 'best')
                        ):
                            os.makedirs(os.path.join(args['output_dir'], 'best'))
                        tmp = results['avg_acc']
                        if tmp > best:
                            origin_model.bert.save_all_adapters(os.path.join(args['output_dir'], 'best'))
                            torch.save(classifiers[task_name],os.path.join(args['output_dir'], 'best/fc.bin'))
                            global_model.bert.save_pretrained(os.path.join(args['output_dir'], 'best'))
                            for snap_shot in snap_shots:
                                results['dataframe_'+snap_shot].to_csv(
                                    os.path.join(args['output_dir'], 'best', snap_shot+'_rep.csv'), index=0
                                )
                            best = tmp
                    logging_loss = tr_loss
                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
        origin_model.bert.save_all_adapters(args['output_dir'])
        tokenizer.save_pretrained(args['output_dir'])
        torch.save(classifiers[task_name],os.path.join(args['output_dir'], 'fc.bin'))
        global_model.bert.save_pretrained(args['output_dir'])
        return global_step, tr_loss / global_step
    
    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = {
        'data_dir': 'data/',
        'reprocess_input_data': False,
        'model_type':  'bert',
        'model_name': 'bert-base-multi',
        'task_name': 'thu1',
        'prev_task':'',
        'output_base_dir': 'ckpts_reproduce',
        'output_dir': '',
        'lambda': 1,
        'T': 3,
        'recalibrate': True,
        'max_seq_length': 128,
        'output_mode': 'classification',
        'train_batch_size': 32,
        'eval_batch_size': 64,

        'num_class': 3,
        'gradient_accumulation_steps': 1,
        'num_train_epochs': 1,
        'weight_decay': 0,
        'learning_rate_snapshot': 1e-4,
        'learning_rate_global': 2e-5,
        'adam_epsilon': 1e-8,
        'warmup_steps': 0, 
        'max_grad_norm': 1.0,

        'logging_steps': 500,
        'evaluate_during_training': True,
        'save_steps': 5000,
        'ckpt_path': 'pre-trained/multi_lang/',
        'original_ckpt_path': 'pre-trained/multi_lang/',
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
    }

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
    config_class.from_json_file(
        os.path.join(args['ckpt_path'], 'config.json')
    )

    tokenizer = tokenizer_class.from_pretrained(args['ckpt_path'])
    args['config'] = config_class
    
    global_step = 0
    prev_task = ''
    next_tasks = ['thu1', 'thu2', 'thu3', 'thu4']
    
    for task in next_tasks:
        # update args
        args['task_name'] = task
        args['prev_task'] = prev_task

        args['output_dir'] = os.path.join(args['output_base_dir'], f'multi_lang_{task}')
        processor = processors[task]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
        global_model = bert(args['ckpt_path']).to('cuda')
        classifiers = {}
        init_fc = nn.Linear(768,num_labels,bias=False).to(device)
        classifiers[task] = init_fc

        origin_model_args = args.copy()
        origin_model = bert(origin_model_args['original_ckpt_path']).to('cuda')
        origin_model.bert.add_adapter(task)
        if len(args['prev_task']) != 0:
            prev_task_names = args['prev_task'].split(',')
            for prev_task_name in prev_task_names:
                fc_layer = torch.load(
                    os.path.join(args['output_base_dir'],
                                 'multi_lang_'+prev_task_name, 
                                 'best', 'fc.bin'),
                    map_location=torch.device('cuda'))
                origin_model.bert.load_adapter(
                    os.path.join(args['ckpt_path'], prev_task_name))
                classifiers[prev_task_name] = fc_layer

        origin_model = origin_model.to('cuda')
        print(processors)
        # train_dataset = load_and_cache_examples(task, tokenizer)
        snap_shots = list(classifiers.keys())
        print(snap_shots)
        # results = global_eval(snap_shots,global_model,classifiers,tokenizer)
        global_step, tr_loss = train_global(task,global_model,origin_model,snap_shots,classifiers,tokenizer,global_step=global_step)
        #fine_tune_fc(task,global_model,snap_shots,classifiers,tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # update args
        prev_task = f'{task},' + prev_task
        prev_task = prev_task.strip(',')
        args['ckpt_path'] = os.path.join(args['output_base_dir'],
                                         'multi_lang_'+task, 'best')
