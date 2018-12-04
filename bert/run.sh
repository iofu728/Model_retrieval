#!/usr/bin/env bash
# @Author: gunjianpan
# @Date:   2018-12-02 10:50:24
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-12-03 22:01:42
export BERT_BASE_DIR=/Users/gunjianpan/Desktop/git/bert/chinese_L-12_H-768_A-12 #全局变量 下载的预训练BERT地址
export MY_DATASET=/Users/gunjianpan/Desktop/git/bert/data #全局变量 数据集所在地址

/usr/local/Cellar/python/3.5.6/bin/python3.5 run_classifier.py \
  --task_name=ppap \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/Users/gunjianpan/Desktop/git/bert/data/
