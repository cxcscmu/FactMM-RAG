from transformers import T5Tokenizer, T5ForConditionalGeneration, CLIPProcessor, T5Model,CLIPVisionModel
from multi_model import MultiModal
import numpy as np
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_model(args,device):
    clip_model_name=args.clip_model_name
    t5_model_name=args.t5_model_name
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5Model.from_pretrained(t5_model_name)
    t5_tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]})
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    image_processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = MultiModal(clip_model_name, t5_model, t5_tokenizer)
    model = model.to(device)
    return t5_tokenizer, model, image_processor

def get_img_patch_token_size(clip_model_name):
    clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
    image_size=clip_model.config.image_size
    patch_size=clip_model.config.patch_size
    img_patch_token_size=int(image_size/patch_size)**2
    return img_patch_token_size

def manhattan_distance(u, v):
    return np.sum(np.abs(u - v))

def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list.values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score

