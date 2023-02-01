import torch
from torch.nn import functional as F
import numpy as np
# from transformers import BertModel, BertConfig, BertTokenizer, BertLMHeadModel, BertTokenizerFast
# from transformers import *
from datasets import load_dataset

import os
import sys
from tqdm import tqdm
import random
import math

import matplotlib.pyplot as plt


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

filelist = getListOfFiles('/notebook/GreenAl/owt_files/openwebtext')
len(filelist)

class FileListIteratorMultiproc:
    def __init__(self, filelist, current_proc, n_proc, from_checkpoint = False):
        # current_file_idx - defines index of file on which previous iteration is stopped
        # only if files are not shuffled
        self.filelist = filelist
        self.nfiles = len(self.filelist)
        self.n_proc = n_proc
        self.current_proc = current_proc
        self.from_chekpoint = from_checkpoint
        self.current_file_idx = 0
        
    def __iter__(self):
        #worker_info = torch.utils.data.get_worker_info()
        #print ("worker_info it", worker_info)
        # self.lines = 0
        self.fileidx = 2700000#self.current_file_idx if self.from_chekpoint else 0
        
        self.shift = 1
        if self.n_proc > 1:
            #per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = self.current_proc
            self.fileidx = self.fileidx + worker_id
            self.shift = self.n_proc
            print ("multiproc")
            print ("worker id", worker_id)
            print ("self.fileidx ", self.filelist[self.fileidx])
            print ("shift", self.shift)
            print ("next", self.filelist[self.fileidx + self.shift])
            print ("\n\n")
            
        self.fin = open(self.filelist[self.fileidx], "r")
        # single-process data loading, return the full iterator
        while True:
            line = self.fin.readline()
            # self.lines += 1
            if line == "":
                # reached EOF
                # print('reached eof of file', self.fileidx, self.nfiles, self.lines)
                # self.lines = 0
                self.fin.close()
                self.fileidx += self.shift
                self.current_file_idx = self.fileidx
                if self.fileidx > self.nfiles - self.n_proc:
                    # end of filelist
                    # print('reached end of filelist', self.fileidx)
                    break
                else:
                    self.fin = open(self.filelist[self.fileidx], "r")
                    line = self.fin.readline()
                    yield line.strip("\n")
            else:
                yield line.strip("\n")
                
class FileListDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterator, tokenizer, seq_len, filelist = ''):
        self._iterator = iterator
        self._tokenizer =tokenizer
        self.seq_len = seq_len
        self.filelist = filelist
        
    
    @classmethod
    def from_filelist(cls, filelist, tokenizer, seq_len, current_proc=0, n_proc=1):
        worker_info = torch.utils.data.get_worker_info()
        print ("worker_info", worker_info)
        iterator = FileListIteratorMultiproc(filelist=filelist, current_proc=current_proc, n_proc=n_proc)
        return cls(
            iterator=iterator,
            tokenizer=tokenizer,
            seq_len=seq_len,
            filelist = filelist
        )
    
    def __iter__(self):
        """
            Yields (List[int])
        """
        ids = []
        for text in self._iterator:
            ids.extend(self._tokenizer.encode(text))
            while len(ids) >= self.seq_len+1:
                yield {"input_ids": ids[:self.seq_len],
                       "labels": ids[1:self.seq_len+1]}
                ids = ids[self.seq_len:]

    @classmethod
    def collate_fn(cls, item):
        """Collate function for DataLoader
        Args:
            item (List[dict[str, List[int]]])
        Returns:
            (dict[str, torch.Tensor]):
        """
        keys = item[0].keys()
        dic = {
            key: torch.tensor([x[key] for x in item])
            for key in keys
        }
        return dic
    
    def __len__(self):
        return len(self.filelist)