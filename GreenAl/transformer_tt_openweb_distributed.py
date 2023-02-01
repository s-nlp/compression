from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, default_data_collator
from transformers import default_data_collator
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2Tokenizer

import argparse
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import pathlib
# BASE_DIR will be like '/home/jovyan/DemoExample/'
BASE_DIR = pathlib.Path().absolute()
print(f"Working dir: {BASE_DIR}")

import argparse

import sys
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR)+"/"+"src")

from src.classes.gpt2_tt import GPT2_TT_Model
from src.classes.gpt_med_config import GPT2MedConfig
from help_trainer_distr import train

#tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#mapping datasets
from datasets import load_dataset
from transformers import TextDataset
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

# iterable datasets
from src.data_classes.iterable_dataset_mp import getListOfFiles, FileListDataset

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"

filelist = getListOfFiles(str(BASE_DIR)+'/' +'owt_files/openwebtext/texts')
print ("lf", len(filelist))

#filelist = filelist1 + filelist2 + filelist3 + filelist4 + filelist5

#del filelist4, filelist3, filelist2, filelist1, filelist5

#print (len(filelist))

def train_mp_wrapper(gpu, args):
    
    """Wraps the process of training distributed model on a single gpu.
       Registers the process, creates Data Parralell GPT model (regular or compressed), create test, valid datasets and train dataloders.
    """
    
    print ("wr", gpu, flush = True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=gpu, world_size=args.n_gpu)
    print ("gpu", gpu, flush = True)
    # Initializing a GPT2 configuration
    configuration = GPT2MedConfig()
   
    # Initializing a model from the configuration
    if (args.rank > 0):
        model = GPT2_TT_Model(configuration, rank = args.rank)
    else:
        model = GPT2LMHeadModel(configuration)
    model.to(gpu)
    
    device = torch.device(f'cuda:{gpu}') 
    
    ddp_model = DDP(model, device_ids=[gpu], output_device = gpu)
    ddp_model.to(gpu)
    
    # loading from checkpoint
    if (args.from_chkpt):
        dictt1 = torch.load("/notebook/GreenAl/out_transformer_0_v4/checkpoint-22000/model_tt.pth", map_location=device)
        ddp_model.load_state_dict(dictt1)  
        
    print ("model loaded", flush = True)
    
    torch.manual_seed(0)
    dataset_valid = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/GreenAl/wikitext-103/wiki.valid.tokens", 
                                block_size=1024)
    
    dataset_test = TextDataset(tokenizer=tokenizer, 
                                file_path="/notebook/GreenAl/wikitext-103/wiki.valid.tokens", block_size=1024)
    print ("loaded test valid datsets", flush = True)
    
    dataset_train = FileListDataset.from_filelist(filelist=filelist, tokenizer=tokenizer, seq_len=1024, current_proc=gpu, n_proc=args.n_gpu)
    train_dataloader = DataLoader(dataset_train, batch_size=args.per_gpu_train_batch_size, collate_fn=FileListDataset.collate_fn, drop_last=True)
    
    print ("wr5", gpu)
    
    train(args, train_dataloader, dataset_valid, dataset_test, ddp_model, gpu)
    dist.destroy_process_group()

def main():
    
    """
    The main function of an experiment module.
    Processes arguments from the command line and provides data-parallel training of GPT-based models on several GPUs.
    Every training process corresponds to a certain GPU, the process started is provided by the spawn method - creating a fresh python process with its separate own interpreter.
    The spawn point should be as close to the start of the main module process as possible. It is not recommended to add working functionality in the main process before the spawn point.
    
    The following experiment attributes are set:
    
        - training attributes:
           - max_steps - maximum number of steps in a training process
           - per_gpu_train_batch_size - batch size processed into single GPU card while training. In the corresponding experiment, settings is equivalent to train_batch_size
           - per_gpu_eval_batch_size - batch size processed into single GPU card while evaluating. In the corresponding experiment settings is equivalent to eval_batch_size
           - n_gpu - number of GPU cards involved in training. In the corresponding experiment, settings is equivalent to the number of the separate training process (i.e. "world size").
           - num_train_epochs - number of training epoch (40 -100 for relatively small datasets, 4-6 for a big one)
           - weight_decay - weight weights regularizer is added with
           - learning_rate - the size of step in gradient descent
           - adam_epsilon - epsilon parameter in case of default optimizer (AdamW)
           - warmup_steps - warmup_steps in case of scheduler
           - seed - random seed to further reproduce the experiments
           - device - cuda device
           - fp16 - is low precision is used while training (False for every experiment)
           - max_grad_norm - gradient norm
           - logging_steps - how often model will be evaluated and write results to log
           - save_steps - how often model will be saved
           - evaluate_during_training - whether the model will be evaluated during the training process (on a validation dataset) or only at the end of training process (on a test dataset)
           - output_dir - path to the directory where checkpoints (model weights, optimizer, and scheduler states)
           - eval_batch_size - batch size processed while evaluating
           - save_total_limit - number of checkpoints stored 
          
        - loading attributes:
           - from_chkpt - should the model be loaded from the checkpoint or not
           - chkpt_path - path to the checkpoint for loading (if from_chkpt is True)
           
        - model attributes:
          - rank - set therank of TT layer. If rank is 0, model is a regular GPT
              
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, default=200)

    args = parser.parse_args()

    #args.rank = args_out.rank
    
    args.local_rank = 0
    args.max_steps = -1
    args.per_gpu_train_batch_size = 3
    args.per_gpu_eval_batch_size = 2
    args.n_gpu = 2
    args.gradient_accumulation_steps = 22
    args.num_train_epochs = 5
    args.weight_decay = 0.005
    args.learning_rate = 5.85e-6#2.95e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 1500
    args.seed = 15
    args.mlm = False
    args.device = torch.device('cuda')
    args.fp16 = False
    args.max_grad_norm = 1.0
    args.logging_steps = 100
    args.save_steps = 1000
    args.evaluate_during_training = True
    args.output_dir = '/notebook/GreenAl/out_transformer_0_v4'
    args.eval_batch_size = 16
    args.save_total_limit = 2
    args.from_chkpt = False
    args.chkpt_path = "/notebook/GreenAl/out_transformer_0_v4/checkpoint-22000/"

    mp.spawn(train_mp_wrapper, nprocs=args.n_gpu, args=(args,))
    
if __name__ == "__main__":
    main()
    
    
#AdamW, β1=0.9, β2=0.95, eps=1e−8
#learning rate:
#peak=6e-5
#warmup over 183_105 samples (375M tokens)
#cosine decay for learning rate down to 10% of its value, over 410B tokens (after 410B tokens, training continues at 10% of the original learning rate, that is fixed --min-lr)
#clipping by global norm of 1 (as in GPT-3)
#weight decay of 0.1 (same as in GPT3 and 530B trainings)