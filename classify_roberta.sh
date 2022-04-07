kmer=1  

CUDA_VISIBLE_DEVICES=2 python run_classify.py --pretrained_model_path models/pre-trained_models/uniprot_roberta_seq256_${kmer}kmer_model.bin \
                                                            --vocab_path models/uniprot_${kmer}kmer_vocab.txt  \
                                                            --config_path models/roberta_base_config.json \
                                                            --train_path dataset/multi-function/train.tsv \
                                                            --dev_path dataset/multi-function/dev.tsv \
                                                            --epochs_num 35 --batch_size 16 \
                                                            --binary_report \
                                                            --learning_rate 5e-5 \
                                                            --report_steps 40 \
                                                            --seq_length 256 --kmer ${kmer} \
                                                            --output_model_path models/roberta_finetune_${kmer}kmer_model.bin \
                                                            --embedding word_pos_seg --encoder transformer --mask fully_visible