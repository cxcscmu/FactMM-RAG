import json
import os
import numpy as np
from tqdm import tqdm,trange
import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data import MedDataset
import torch.nn.functional as F
import wandb
from transformers import get_cosine_schedule_with_warmup
import random
from utils import load_model,get_img_patch_token_size
import pickle
from IPython import embed


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def eval_loss(model, loss_function, valid_reader, device):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    counter = 0.0
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():
            batch_size=batch['query_image_inputs'].size(0)
            
            query_embeddings = model(batch['query_image_inputs'].cuda(),None,device)
            candidate_embeddings = model(batch['pos_neg_image_inputs'].cuda(),batch['pos_neg_report_inputs'],device)
            
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
            logit_scale = model.logit_scale.exp()
            score = torch.matmul(query_embeddings, candidate_embeddings.t())* logit_scale
            target = batch['targets'].cuda()
            
            loss = loss_function(score, target)
            max_score, max_idxs = torch.max(score, 1)

            correct_predictions_count = (max_idxs == target).sum() / batch_size
            total_corr += correct_predictions_count.item()
            total_loss += loss.item()
            counter += 1
            
    if counter == 0:
        return 0.0, 0.0
    return total_loss / counter, total_corr / counter

def train(train_reader, valid_reader, model, device):
    t_total = len(train_reader) // args.gradient_accumulation_steps * args.num_train_epochs
    eval_step = t_total//args.num_train_epochs
    
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_steps*t_total), num_training_steps=t_total
    )             
    loss_function = torch.nn.CrossEntropyLoss()
    tag, global_step, global_loss, best_acc = 0, 0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_reader)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch_size=batch['query_image_inputs'].size(0)
            
            query_embeddings = model(batch['query_image_inputs'].cuda(),None,device)
            candidate_embeddings = model(batch['pos_neg_image_inputs'].cuda(),batch['pos_neg_report_inputs'],device)
            
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
            logit_scale = model.logit_scale.exp()
            score = torch.matmul(query_embeddings, candidate_embeddings.t())* logit_scale
            target = batch['targets'].cuda()
            
            loss = loss_function(score, target)
            max_score, max_idxs = torch.max(score, 1)
            correct_predictions_acc = (max_idxs == target).sum() / batch_size     

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            global_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                wandb.log(
                    {
                        "training_loss":global_loss / global_step,
                        "training_acc":correct_predictions_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"]
                    }
                )
                epoch_iterator.set_description(f"Loss:{global_loss / global_step}")
                
                if global_step % eval_step == 0 and global_step > 0:
                    dev_loss, dev_acc = eval_loss(model, loss_function, valid_reader, device)
                    print(
                        "Evaluation at global step {}, average dev loss: {:.4f},average dev acc: {:.4f}".format(
                            global_step, dev_loss, dev_acc))
                    wandb.log(
                        {
                            "dev_loss":dev_loss,
                            "dev_acc":dev_acc,
                        }
                    )
                    
                    if best_acc <= dev_acc:
                        best_acc = dev_acc
                        torch.save({'epoch': epoch,
                                    'model': model.state_dict()}, args.out_path)
                        print("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))
                        tag = 0
                    else:
                        tag += 1
                    if tag >= args.early_stop:
                        print('*********early stop**********')
                        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--train_pos_path", type=str)
    parser.add_argument("--train_neg_path", type=str)    
    parser.add_argument("--valid_pos_path", type=str)
    parser.add_argument("--valid_neg_path", type=str)    
    
    parser.add_argument("--wandb_name", type=str)
    
    parser.add_argument("--t5_model_name", type=str, default='OpenMatch/t5-ance')
    parser.add_argument("--clip_model_name",type=str,default='openai/clip-vit-base-patch32')
    parser.add_argument("--pretrained_model_path", type=str)
    
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=0.1)
    
    
    args = parser.parse_args()


    set_seed(args)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    wandb.init(
            name=args.wandb_name,
            sync_tensorboard=True)
    
    wandb.config.update(args)  

    
    train_data = json.load(open(args.train_path,"r"))
    valid_data = json.load(open(args.valid_path,"r"))
    train_pos_data = pickle.load(open(args.train_pos_path,'rb'))
    train_neg_data = pickle.load(open(args.train_neg_path,'rb'))
    valid_pos_data = pickle.load(open(args.valid_pos_path,'rb'))
    valid_neg_data = pickle.load(open(args.valid_neg_path,'rb'))    

    tokenizer, model, image_processor = load_model(args,device)
    model.to(device)
 
      
    img_patch_token_size=get_img_patch_token_size(args.clip_model_name)
    
    train_dataset = MedDataset(image_processor, tokenizer, train_data,train_pos_data,train_neg_data,img_patch_token_size)
    valid_dataset = MedDataset(image_processor, tokenizer, train_data,valid_pos_data,valid_neg_data,img_patch_token_size,valid_data)
    


    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset) 
    
    traindata_reader = DataLoader(dataset=train_dataset, sampler=train_sampler, num_workers=args.num_workers,
                                  batch_size=args.train_batch_size, collate_fn=train_dataset.Collector, drop_last=True)
    validdata_reader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size, collate_fn=valid_dataset.Collector, drop_last=False)
    if args.pretrained_model_path != None:
        model.load_state_dict(torch.load(args.pretrained_model_path)['model'],strict=False)
    model.cuda()
    train(traindata_reader, validdata_reader, model, device)
