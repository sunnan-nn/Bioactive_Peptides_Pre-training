import sys
import os
import random
import argparse
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
import re
from torch.nn import BCEWithLogitsLoss
from metrics import multi_label_metrics, binary_class_metrics

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts


# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

LABEL_NUMs = 5
LABEL_LIST = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP']
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.criterion = BCEWithLogitsLoss()

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                # loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                output = logits.float()
                target = tgt.float()
                loss = self.criterion(input = output, target = target)

            return loss, logits
        else:
            return None, logits


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            label = line[columns["label"]]
            tgt = [np.float(x) for x in label.split(",")]
            text_a = line[columns["text_a"]]
            
            # split the sentence to kmer
            text_a_list = re.findall(".{" + str(args.kmer) + "}", text_a)
            if len(text_a) % args.kmer != 0: text_a_list.append(text_a[(len(text_a)//args.kmer) * args.kmer : ])
            text_a = " ".join(text_a_list)
            
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    if args.use_adv and args.adv_type == "fgm":
        args.adv_method.attack(epsilon=args.fgm_epsilon)
        loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss_adv = torch.mean(loss_adv)
        loss_adv.backward()
        args.adv_method.restore()

    if args.use_adv and args.adv_type == "pgd":
        K = args.pgd_k
        args.adv_method.backup_grad()
        for t in range(K):
            # apply the perturbation to embedding
            args.adv_method.attack(epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                                   is_first_attack=(t == 0))
            if t != K - 1:
                model.zero_grad()
            else:
                args.adv_method.restore_grad()
            loss_adv, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            if torch.cuda.device_count() > 1:
                loss_adv = torch.mean(loss_adv)
            loss_adv.backward()
        args.adv_method.restore()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    args.model.eval()

    all_prob = []
    all_gold = []

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        # pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        prob = nn.Sigmoid()(logits)

        all_prob.extend(prob.cpu().numpy().tolist())
        all_gold.extend(gold.cpu().numpy().tolist())


    all_pred = all_prob
    for i in range(len(all_prob)):
        for j in range(len(all_prob[i])):
            if all_prob[i][j] < 0.5:
                all_pred[i][j] = 0
            else:
                all_pred[i][j] = 1
    
    results = multi_label_metrics(np.array(all_pred), np.array(all_gold))
    print("Evaluation is done")
    for key, value in results.items():
        print(key + ": ", value)
    
    if args.binary_report: 
        binary_class_metrics(np.array(all_prob), np.array(all_pred), np.array(all_gold), args.id2label)
       
    return results

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument(
        "--early_stop", default=0, type=int, 
        help="set this to a positive integet if you want to perfrom early stop. The model will stop \
                                                    if the auc keep decreasing early_stop times",
    )
    parser.add_argument(
        "--binary_report", action="store_true")

    adv_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    args.labels_num = LABEL_NUMs
    args.id2label = ID2LABEL

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    total_loss, best_acc = 0.0, 0.0
    stop_count = 0

    print("Start training.")
    
    # tb_writer = SummaryWriter()
    global_step = 0
    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            global_step += 1
            if (i + 1) % args.report_steps == 0:
                loss_scalar = total_loss / args.report_steps
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, loss_scalar))
                # tb_writer.add_scalar("tr_loss", loss_scalar, global_step)
                # learning_rate_scalar = scheduler.get_lr()[0]
                # tb_writer.add_scalar("learning_rate", learning_rate_scalar, global_step)
                total_loss = 0.0                     
        
        results = evaluate(args, read_dataset(args, args.dev_path))
        # tb_writer.add_scalar("accuracy", results["accuracy"], epoch)

        if results["accuracy"] > best_acc:
            best_acc = results["accuracy"]
            save_model(model, args.output_model_path)
            print("Saving model checkpoint to ", args.output_model_path)
        
        if args.early_stop != 0:
            if results["accuracy"] < best_acc:
                stop_count += 1
            else:
                stop_count = 0
            # print(stop_count)
            if stop_count == args.early_stop:
                print("Early stop")
                break
  
    # tb_writer.close()

    # Evaluation phase.
    print("Evaluate the following checkpoints: ", args.output_model_path)
    if torch.cuda.device_count() > 1:
        args.model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        args.model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, read_dataset(args, args.dev_path))

    # Testing phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
