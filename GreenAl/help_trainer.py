import logging
import glob
from tqdm import tqdm, trange
import numpy as np
import torch
import os

import logging
import pickle
import random
import re
import shutil

from transformers import TextDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("transformer.log")
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

        
def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset


from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

def train(args, train_dataset, valid_dataset, test_dataset, model, tokenizer, gpu):
    
    #print(torch.cuda.memory_allocated())
    #print(torch.cuda.memory_reserved())
    #print("memory summary1", torch.cuda.memory_summary(device=model.device), flush = True)
    
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset) 
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    print ("total step", t_total)
    print ("len(train_dataloader)", len(train_dataloader))
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    print ("num_epochs", int(args.num_train_epochs))
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        print ("epoch ", _, "\n")
        losses = []
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_tokens1(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            
            outputs = model(inputs) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            losses.append(tr_loss)
            #print(torch.cuda.memory_allocated())
            #print(torch.cuda.memory_reserved())
            #print("memory summary1", torch.cuda.memory_summary(device=model.device), flush = True)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
               
                
                if gpu == 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print ('total step, logging step', t_total, global_step)
                    print ('lr', scheduler.get_lr()[0], global_step, flush = True)
                    print ('loss', (tr_loss - logging_loss)/args.logging_steps, global_step, flush = True)
                    logging_loss = tr_loss

                if gpu == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, 'model_tt.pth'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if gpu == 0 and args.evaluate_during_training: 
            results = evaluate(args, model, valid_dataset, tokenizer)
        #wandb.log(results)
        print ("evaluation ", results, flush = True)
        print ("losses", np.array(losses).mean())
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    #print(torch.cuda.memory_allocated())
    #print(torch.cuda.memory_reserved())
    #print("memory summary", torch.cuda.memory_summary(device=model.device), flush = True)
    results = evaluate(args, model, test_dataset, tokenizer)
    print ("test ", results, flush = True)
    

    return global_step, tr_loss / global_step

def evaluate(args, model, dataset_valid, tokenizer, prefix=""):
    """ Evaluate the trained model
    
    Parameters:
    - args - object argument class which stores the training parameter (learning rate, batch_size)
    - dataset_valid - dataset for validation (class transformers.TextDataset by default)
    - model - model to valid (DistributedDataParallel by default)
    - prefix - identifier to output file (optional)
    
    Returns:
    - dict with keys "perplexity" and "loss"
    
    """
    
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = dataset_valid
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=True, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    perplexity = 0.0
    nb_eval_steps = 0
    model.eval()
    losses = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
            perplexity += torch.exp(torch.tensor(eval_loss))
            losses.append(eval_loss)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    #perplexity = perplexity / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity,
        "loss":eval_loss
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result