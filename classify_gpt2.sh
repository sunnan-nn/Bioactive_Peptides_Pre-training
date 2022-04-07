kmer=1

CUDA_VISIBLE_DEVICES=2 python3 run_classify.py --pretrained_model_path models/pre-trained_models/uniprot_gpt2_seq256_${kmer}kmer_model.bin \
                                   --vocab_path models/uniprot_${kmer}kmer_vocab.txt \
                                   --config_path models/gpt2_base_config.json \
                                   --train_path dataset/multi-function/train.tsv \
                                   --dev_path dataset/multi-function/dev.tsv \
                                   --learning_rate 7e-5 \
                                   --epochs_num 30 --batch_size 16 \
                                   --report_steps 40 \
                                   --seq_length 256 --kmer ${kmer} \
                                   --output_model_path models/gpt2_finetune_${kmer}kmer_model.bin \
                                   --embedding word_pos --remove_embedding_layernorm \
                                   --encoder transformer --mask causal --layernorm_positioning pre --pooling mean