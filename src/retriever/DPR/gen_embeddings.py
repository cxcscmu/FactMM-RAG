import json

from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import os
from PIL import Image
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

from IPython import embed
from PIL import ImageFile
import pyarrow as pa
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import load_model,get_img_patch_token_size



def gen_img_embeddings(model, valid_reader, outpath):
    model.eval()
    all_img_embeddings = []
    
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():

            embeddings = model(batch['img_inputs'].cuda(), None, device)
            embeddings = F.normalize(embeddings, dim=-1).cpu()
            all_img_embeddings.append(embeddings)
                
    all_img_embeddings = torch.cat(all_img_embeddings, dim=0).numpy()
    with open(outpath, 'wb') as fout:
        pickle.dump(all_img_embeddings, fout)

def gen_txt_embeddings(model, valid_reader, findings_outpath):
    model.eval()
    all_findings_embeddings = []
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():        
            findings_embeddings =  model(None, batch["findings_inputs"], device)
            findings_embeddings = F.normalize(findings_embeddings, dim=-1).cpu()
            all_findings_embeddings.append(findings_embeddings)
            
    all_findings_embeddings = torch.cat(all_findings_embeddings, dim=0).numpy()
    
    with open(findings_outpath, 'wb') as fout:
        pickle.dump(all_findings_embeddings, fout)      


class MimicImgDataset(Dataset):
    def __init__(self, args, img_path, preprocess, tokenizer):

        self.preprocess_fn = preprocess
        self.tokenizer = tokenizer
        self.img_paths = []

        with open(img_path, "r") as fin:
            images = json.load(fin)
            for i in range(len(images)):
                self.img_paths.append(images[i]['image'][0])
                
    def __len__(self):
        return len(self.img_paths)

    def encode_img(self, img, idx):
        img = self.preprocess_fn(images=Image.open(img), return_tensors="pt")["pixel_values"][0]
        return {'img': img}

    def Collector(self, batch):
        img_inputs = []

        for example in batch:
            img_inputs.append(example['img_inputs'])

        processed_batch = {}
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)

        return processed_batch

    def __getitem__(self, index):
        img_inputs = self.encode_img(self.img_paths[index], index)
        instance = {
            'img_inputs': img_inputs['img']
        }


        return instance


class MimicTxtDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.findings = []
        
        with open(data_path, "r") as fin:
            text_data = json.load(fin)
            for instance in text_data:
                self.findings.append(instance["finding"])

    
    def __len__(self):
        return len(self.findings)

    def Collector(self, batch):
        processed_batch = {
            'findings_inputs': self.tokenizer(batch, return_tensors='pt',
                                         padding=True, truncation=True)          
        }
        return processed_batch

    def __getitem__(self, index):
        return self.findings[index]





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")
    parser.add_argument("--t5_model_name", type=str, default='OpenMatch/t5-ance')
    parser.add_argument("--clip_model_name",type=str,default='openai/clip-vit-base-patch32')
    parser.add_argument("--saved_ckpt",type=str,default='/FactMM-RAG/src/retriever/output/dpr.best.pt')
    parser.add_argument("--train_path",type=str,default='/FactMM-RAG/data/mimic/train.json') 
    parser.add_argument("--train_path",type=str,default='/FactMM-RAG/data/mimic/valid.json') 
    parser.add_argument("--test_path",type=str,default='/FactMM-RAG/data/mimic/test.json')

    parser.add_argument("--output_train_image_path",type=str,default='/FactMM-RAG/DPR/embedding/train_embedding_image.pkl')      
    parser.add_argument("--output_train_finding_path",type=str,default='/FactMM-RAG/DPR/embedding/train_embedding_finding.pkl')    
    parser.add_argument("--output_valid_image_path",type=str,default='/FactMM-RAG/DPR/embedding/valid_embedding_image.pkl')     
    parser.add_argument("--output_test_image_path",type=str,default='/FactMM-RAG/DPR/embedding/test_embedding_image.pkl')     
    
    args = parser.parse_args()

    t5_tokenizer, model, image_processor = load_model(args,device)
    model.load_state_dict(torch.load(args.saved_ckpt,map_location='cuda:0')['model'],strict =False)
    model.cuda()
    
    args.img_patch_token_size=get_img_patch_token_size(args.clip_model_name)
    train_path = args.train_path
    test_path = args.test_path
    valid_path = args.valid_path

    txt_data = MimicTxtDataset(train_path, t5_tokenizer)
    sampler = SequentialSampler(txt_data)
    txt_reader = DataLoader(dataset=txt_data, sampler=sampler, num_workers=10,
                            batch_size=32, collate_fn=txt_data.Collector) 
    gen_txt_embeddings(model, txt_reader, args.output_train_finding_path)    
    
    img_data = MimicImgDataset(args, train_path, image_processor, t5_tokenizer)        
    sampler = SequentialSampler(img_data)
    img_reader = DataLoader(dataset=img_data, sampler=sampler, num_workers=10,
                            batch_size=32, collate_fn=img_data.Collector) 
    gen_img_embeddings(model, img_reader, args.output_train_image_path) 
 
    img_data = MimicImgDataset(args, valid_path, image_processor, t5_tokenizer)        
    sampler = SequentialSampler(img_data)
    img_reader = DataLoader(dataset=img_data, sampler=sampler, num_workers=10,
                            batch_size=32, collate_fn=img_data.Collector) 
    gen_img_embeddings(model, img_reader, args.output_valid_image_path) 
 

        
    img_data = MimicImgDataset(args, test_path, image_processor, t5_tokenizer)        
    sampler = SequentialSampler(img_data)
    img_reader = DataLoader(dataset=img_data, sampler=sampler, num_workers=10,
                            batch_size=32, collate_fn=img_data.Collector) 
    gen_img_embeddings(model, img_reader, args.output_test_image_path) 


