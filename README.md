Long-Span Summarization
=====================================================
Requirements
--------------------------------------
- python 3.7
- torch 1.2.0
- transformers (HuggingFace) 2.11.0

Overview
--------------------------------------
1. train/ = training scripts for BART, LoBART, HierarchicalModel (MCS)
2. decode/ = running decoding (inference) for BART, LoBART, MCS-extractive, MCS-attention, MCS-combined
3. data/ = data modules, pre-processing, and sub-directories containing train/dev/test data
4. models/ = defined LoBART & HierarchicalModel
5. traintime_select/ = scripts for processing data for trainining (aka ORACLE methods, pad-rand, pad-lead, no-pad)
6. conf/ = configuration files for training

Pipeline (before training starts)
--------------------------------------
- Download data, e.g. Spotify Podcast, arXiv, PubMed & put in data/
- Basic pre-processing (train/dev/test) & put in data/
- ORACLE processing (train/dev/) & put in data/
- Train HierModel (aka MCS) using data with basic pre-processing
- MCS processing & put in data/
- Train BART or LoBART using data above
- Decode BART or LoBART (note that if MCS is applied, run MCS first i.e. save your data from MCS somewhere and load it)

Training BART & LoBART
--------------------------------------
**Training**:

    python train/train_abssum.py conf.txt

**Configurations**:

Setting in conf.txt, e.g. conf/bart_podcast_v0.txt
- **bart_weights** - pre-trained BART weights, e.g. facebook/bart-large-cnn
- **bart_tokenizer** - pre-trained tokenizer, e.g. facebook/bart-large
- **model_name** - model name to be saved
- **selfattn** - full | local
- **multiple_input_span** - maximum input span (multiple of 1024)
- **window_width** - local self-attention width
- **save_dir** - directory to save checkpoints
- **dataset** - podcast
- **data_dir** -  path to data
- **optimizer** - optimzer (currently only adam supported)
- **max_target_len** - maximum target length
- **lr0**  - lr0
- **warmup** - warmup
- **batch_size** - batch_size
- **gradient_accum** - gradient_accum
- **valid_step** - save a checkpoint every ...
- **total_step** - maximum training steps
- **early_stop** - stop training if validaation loss stops improving for ... times
- **random_seed** - random_seed
- **use_gpu** - True | False

Decoding (Inference) BART & LoBART
--------------------------------------
**decoding**:

    python decode/decode_abssum.py \
        --load model_checkpoint
        --selfattn [full|local]
        --multiple_input_span INT
        --window_width INT
        --decode_dir output_dir
        --dataset [podcast|arxiv|pubmed]
        --datapath path_to_dataset
        --start_id 0
        --end_id 1000
        [--num_beams NUM_BEAMS]
        [--max_length MAX_LENGTH]
        [--min_length MIN_LENGTH]
        [--no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE]
        [--length_penalty LENGTH_PENALTY]
        [--random_order [RANDOM_ORDER]]
        [--use_gpu [True|False]]
        
Training Hierarchical Model
--------------------------------------
    python train/train_hiermodel.py conf.txt

 see conf/hiermodel_v1.txt for an example of config file
 
Training-time Content Selection
--------------------------------------
    cd traintime_select
    python oracle_select_{pad|nopad}_{dataset}.py
    
 the configurations are defined manually in the scripts (see VARIABLES in captital)
 
Test-time Content Selection (Running MCS)
--------------------------------------
 **step1**: running decoding for get attention & extractive labelling predictions 
 
    python decode/decode_hiermodel_attn.py
    python decode/decode_hiermodel_ext.py
    
**step2**: combine the two results

    python decode/mcs_inference.py
    
Results using this repository
-----------------------------------------
- Simple fine-tuning vanilla BART(1k) on truncated data

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| Podcast |  26.43  |   9.22  |   18.35 |
|  arXiv  |  44.96  |  17.25  |  39.76  |
|  PubMed |  45.06  |  18.27  |  40.84  |

- Our best results using LoBART(N=4096,W=1024) + MCS (gamma=0.2)

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| Podcast |  27.81  |  10.30  |   19.61 |
|  arXiv  |  48.79  |  20.55  |  43.31  |
|  PubMed |  48.06  |  20.96  |  43.56  |

Trained Weights
-----------------------------------------
To be released, stay tuned!!
