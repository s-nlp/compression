# GreenAl
1) TTM layer for forward and backward passes in language models
2) Training GPT-2 model (small and medium) with TTM layers
3) Results on Language modelling task evaluation and results on GLUE benchmark

**1. TTM layer for forward and backward passes in language models**
Time/memory footprint for a single layer in /notebooks/speed_and_memory_TTM_layer.ipynb

The memory footprint for GPT-2 with TTM layers /notebooks/train_transformers_TT.ipynb

**2.1 Train GPT-2 small on Wikitext-103**

You need to load Wikitext-103 dataset:

curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -o wikitext-103-v1.zip
unzip wikitext-103-v1.zip

To start training regular GPT-2 small:
run **python3 transformer_from_scratch_tt.py --rank 0**

To start gpt-small with custom TTM-R layers:

run **python3 transformer_from_scratch_tt.py --rank R**

**2.2 Train GPT-2 medium on Openwebtext**

You need to load Openwebtext texts(~52G) to /BASE_DIR/owt_files/openwebtext/texts

Training can be run on multiple GPUs. We use a custom dataloader based on torch.utils.data.IterableDataset, which parallelizes the process of reading and processing texts from the data folder. Dataloader launch examples are shown in the file /notebooks/multiproc_iterable_dataset.ipynb

Than run CUDA_VISIBLE_DEVICES="0,1" transformer_tt_openweb_distributed.py --rank 72 --n_gpu 2

