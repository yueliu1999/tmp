**This is a pytorch implementation of the CodePanGu2.6B model. It can be inferred, trained, and finetune on the pytorch framework.**

Starting point: Mindspore is a new deep learning framework that many people have not used, so converting mindspore models to pytorch models will allow more people to use our models and allow users to not only experience our large models, but also finetune our models.

Megatron is a large, powerful transformer algorithm library developed by NVIDIA's deep learning applications research team. This port is based on Megatron, and the main work includes converting model files, adding query layer, and modifying model slicing strategy.

# Environments

Supports python >= 3.6, pytorch >= 1.5, cuda >= 10, and nccl >= 2.6 versions.

The official NVIDIA docker image `docker pull nvcr.io/nvidia/pytorch:20.03-py3` is recommended. You need to install [NLTK](https://www.nltk.org/install.html).

You can also download the paired image directly at

```bash
docker pull yands/pangu-alpha-megatron-lm-nvidia-pytorch:20.03.2
```
Using`/opt/conda/bin/python`。

# Model File Download

| Model File                                                                                          |  Size | Parameter Configuration                                               |
|-----------------------------------------------------------------------------------------------------|------|-----------------------------------------------------------------------|
| [PanGu2.6B_fp16_mgt.zip](https://git.openi.org.cn/attachments/72aec03d-6bdb-4652-ac2a-8099db4b0bed) |  4.6G | num-layers : 32<br />hidden-size : 2560<br />num-attention-heads : 32 |
| CodePangu2.6B_fp16 (Coming soon)                                                                    |  -    | num-layers : 32<br />hidden-size : 2560<br />num-attention-heads : 32 |


Model file directory structure.
```txt
PanGu2.6B_fp16_mgt #model directory, --load parameter needs to fill in the path
    -- iter_0001000 # iteration number directory
        --mp_rank_00 # directory for each GPU when the model is parallel
            --model_optim_rng.pt #model file
    --latest_checkpointed_iteration.txt #file of iterations of ckpt
```

# Finetune
Currently only finetune is provided without changing the model structure and data format, i.e. continue pre-training.
##### 1. Preparing training data

Refer to [data](#data) section

##### 2. Model cutting

The model downloaded above is a single machine inference model, so you need to cut the model first when finetune is performed, and cut it into model parallel models.

Parameters.

`-model-parallel-size`: the number of slices of the original model, here is 1

`--num-mp-model`: the number of models after slicing

`--mp-model-save`: the path to save the model after slicing

```bash
python tools/split_full_model_into_mp_model.py \
--model-parallel-size 1 \
--num-mp-model 2 \
--num-layers 31 \
--hidden-size 2560 \
--load /**ful model path**/ \
--mp-model-save /**mp model save path**/ \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--model-type Pangu \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--finetune
```
##### 3. Training

Run the script:

```examples/finetune_pangu_distributed.sh```

##### 4. Model merging

The finished model of finetune is fragmented, so if you want to do single card inference, you need to merge the model first.

Merge script.

`--mp-model-parallel-size`: the number of model slices

`--load`: model save directory

```bash
python tool/merge_mp_partitions.py \
--model-parallel-size 1 \
--mp-model-parallel-size 2 \
--num-layers 31 \
--hidden-size 2560 \
--load /full model ckpt dir/  \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--model-type Pangu \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--reset-attention-mask \
--finetune \
```

# Training

Reference Script

```bash
examples/pretrain_pangu_distributed_2.6B.sh
```

# Data

##### Generate training data

Reference script: `/tools/preprocess_data_pangu.py`

Store multiple `xxx.txt` files in the train_dataset directory, if there are more training data, it is better to have a uniform file size for each `txt` and separate multiple `txt`s, the size can be 10M a file. If there is traditional text that needs to be converted to simplified, you can use `zhconv`.

The format of each `txt` text is (need blank lines to split different samples)
```txt
sample 1 ***
***
***

sample 2 ***
***
***

sample 2 ***
***
***
```
```bash
python /tools/preprocess_data_pangu.py \
--input /train_dataset/*.txt \
--output-prefix /megatron/dataset/ \
--vocab-file /megatron/tokenizer/bpe_4w_pcl/vocab \
--dataset-impl mmap \
--append-eod
```

The files /path/to/dataset/xxx.idx and /path/to/dataset/xxx.bin will be generated.

Finetune and pre-training require the parameter: `-data-path=/path/to/dataset/xxx`





