#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=8         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml 
train_config=conf/train.yaml

lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
lang_model=rnnlm.model.best # set a language model to be used for decoding

# data
LDC_dir=/home/psridhar/LDC/unprocessed
swbd1_dir="${LDC_dir}/LDC97S62"
eval2000_dir="${LDC_dir}/LDC2002S09/hub5e_00 ${LDC_dir}/LDC2002T43"
fisher_dir="${LDC_dir}/LDC2004T19 ${LDC_dir}/LDC2005T19 ${LDC_dir}/LDC2004S13 ${LDC_dir}/LDC2005S13"
#fisher_dir=

# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

# exp tag
tag="fisher" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
train_dev=train_dev
recog_set="train_dev eval2000"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/swbd1_data_download.sh ${swbd1_dir}
    local/swbd1_prepare_dict.sh
    local/swbd1_data_prep.sh ${swbd1_dir}
    local/eval2000_data_prep.sh ${eval2000_dir}
    if [ -n "${fisher_dir}" ]; then
	local/fisher_data_prep.sh ${fisher_dir}
    fi
    utils/combine_data.sh data/train_all data/train_fisher data/train

    # upsample audio from 8k to 16k to make a recipe consistent with others
    for x in train_all eval2000; do
	    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 8000 dither | /" data/${x}/wav.scp
    done
    # normalize eval2000 and rt03 texts by
    # 1) convert upper to lower
    # 2) remove tags (%AH) (%HESITATION) (%UH)
    # 3) remove <B_ASIDE> <E_ASIDE>
    # 4) remove "(" or ")"
    for x in eval2000; do
        cp data/${x}/text data/${x}/text.org
        paste -d "" \
            <(cut -f 1 -d" " data/${x}/text.org) \
            <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
            | sed -e 's/\s\+/ /g' > data/${x}/text
        # rm data/${x}/text.org
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 1 data/train_all data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 1 data/eval2000 data/${train_dev}
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 64 --write_utt2num_frames true data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done


    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 64 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 64 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    # map acronym such as p._h._d. to p h d for train_set& dev_set
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/${train_set}/text
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/${train_dev}/text

    echo "make a dictionary"
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt

    # Please make sure sentencepiece is installed
    spm_train --input=data/lang_char/input.txt \
            --model_prefix=${bpemodel} \
            --vocab_size=${nbpe} \
            --character_coverage=1.0 \
            --model_type=${bpemode} \
            --model_prefix=${bpemodel} \
            --input_sentence_size=100000000 \
            --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0 \
            --user_defined_symbols="[laughter],[noise],[vocalized-noise]"

    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    mkdir -p data/local/lm_train ${lmdatadir}
    cut -f 2- -d" " data/train/text | gzip -c > data/local/lm_train/${train_set}_text.gz
    zcat data/local/lm_train/${train_set}_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " data/eval2000/text | \
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/valid.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
	expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
	expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
	--report-cer \
	--report-wer \
	--report-interval-iters 50
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

	# this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
	if [[ "${decode_dir}" =~ "eval2000" ]]; then
            local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
	elif [[ "${decode_dir}" =~ "rt03" ]]; then
	    local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
	fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
