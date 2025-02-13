
import torch
from PIL import Image
import io
import numpy as np
import torch
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import random 
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class MedDataset(Dataset):
    def __init__(self, preprocess_fn, tokenizer, data,data_pos,img_special_len=49,valid_data = None):
        self.preprocess_fn = preprocess_fn
        self.tokenizer=tokenizer
        self.img_special_len=img_special_len
        self.data = data
        self.data_pos = data_pos
        self.valid_data = valid_data

        self.splited_data_pos_idx_pair = []

        if self.valid_data != None:
            self.data_pos_idx_filtered = []
            for idx,data_pos_instance in enumerate(self.data_pos):
                if len(data_pos_instance)!=0:
                    self.data_pos_idx_filtered.append(idx)
        else:
            #Splitted training data
            self.splited_data_pos_idx_pair =  [(qid, pos_id) for qid, pos_id_list in enumerate(self.data_pos) for pos_id in pos_id_list]

            
            
        

    def __len__(self):
        if self.valid_data!= None:
            return len( self.data_pos_idx_filtered )
        else:
            return len(self.splited_data_pos_idx_pair )


    def encode_img(self,img,report = None):
        img = self.preprocess_fn(images=Image.open(img), return_tensors="pt")["pixel_values"][0]
        if report != None:
            pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap=pre_token+" "+report
            return {'pos_image': img, 'pos_report':cap}
        else:
            return {'image': img}
            
    def Collector(self, batch):
        query_image_inputs = []
        pos_image_inputs = []
        pos_report_inputs = []

        processed_batch = {}
        for qid, example in enumerate(batch):
            query_image_inputs.append(example['image'])
            pos_image_inputs.append(example['pos_image'])
            pos_report_inputs.append(example['pos_report'])

        processed_batch['query_image_inputs'] = torch.stack(query_image_inputs, dim=0)
        processed_batch['pos_image_inputs'] = torch.stack(pos_image_inputs, dim=0)
        processed_batch['pos_report_inputs'] = self.tokenizer(pos_report_inputs, return_tensors='pt',padding=True,truncation=True)

        return processed_batch

    def __getitem__(self, index):
        if self.valid_data!=None:
            example = self.valid_data[self.data_pos_idx_filtered[index]]
            example_pos_id = random.choice(self.data_pos[self.data_pos_idx_filtered[index]])   
        else:
            example_id,example_pos_id = self.splited_data_pos_idx_pair[index]
            example = self.data[example_id]

        image = example['image'][0]
        instance = self.encode_img(image)
        instance_pos = self.encode_img(self.data[example_pos_id]['image'][0],self.data[example_pos_id]['finding']+" "+self.data[example_pos_id]['impression'])
        instance.update(instance_pos)
        
        return instance



