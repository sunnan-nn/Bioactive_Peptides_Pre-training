kmer=1
task=ACP
CUDA_VISIBLE_DEVICES=5 python run_classify_binary.py --pretrained_model_path models/pre-trained_models/uniprot_roberta_seq256_${kmer}kmer_model.bin \
                                                            --vocab_path models/uniprot_${kmer}kmer_vocab.txt  \
                                                            --config_path models/roberta_base_config.json \
                                                            --train_path dataset/single_function/${task}/train.tsv \
                                                            --dev_path dataset/single_function/${task}/dev.tsv \
                                                            --epochs_num 25 --batch_size 16 \
                                                            --learning_rate 2e-5 \
                                                            --report_steps 40 \
                                                            --seq_length 256 --kmer ${kmer} \
                                                            --output_model_path models/roberta_finetune_${kmer}kmer_model_${task}.bin \
                                                            --embedding word_pos_seg --encoder transformer --mask fully_visible