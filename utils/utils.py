import torch
from itertools import islice, cycle
import os
import random
import numpy as np
import json
import base64
from PIL import Image


def set_random_seed(seed_number):
    # position of setting seeds also matters
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.random.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def truncate_prediction(prediction: str) -> str:
    """Truncate captions at the first newline character, removing leading spaces."""
    prediction = prediction.strip()  # Remove leading and trailing whitespace
    trunc_index = prediction.find('\n')
    if trunc_index != -1:
        prediction = prediction[:trunc_index].strip()
    else:
        # If no newline is found, find the first period and truncate
        trunc_index = prediction.find('.') + 1
        if trunc_index > 0:
            prediction = prediction[:trunc_index].strip()
    return prediction


def load_image(img_ids, root_path):
    if isinstance(img_ids, str):
        img_ids = [img_ids]
    images = []
    image_paths = []
    for img_id in img_ids:
        image_path = os.path.join(root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        image_paths.append(image_path)
        
    return images, image_paths

def coco_id_to_imgname(img_id, prefix='COCO_val2014_'):
    return f'{prefix}{img_id:012}.jpg'

## load data
def load_data(args):
    dataDir = args.dataDir
    query_file = os.path.join(dataDir, args.dataset, 'query.json')
    support_file = os.path.join(dataDir, args.dataset, 'support.json')

    with open(query_file, 'r') as f:
        query_meta = json.load(f)
    with open(support_file, 'r') as f:
        support_meta = json.load(f)

    return query_meta, support_meta
    

def load_text_data(args):
    dataset = args.dataset
    dataDir = args.dataDir
    if dataset == 'agnews':
        from datasets import load_dataset
        data = load_dataset('ag_news')
        support, query = data['train'], data['test']
        label_dict = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        support_meta = []
        for s in support:
            support_meta.append({'question': s['text'], 'answer': label_dict[s['label']]})
        query_meta = []
        for q in query:
            query_meta.append({'question': q['text'], 'answer': label_dict[q['label']]})
    elif dataset == 'imdb':
        from datasets import load_dataset
        data = load_dataset('imdb')
        support, query = data['train'], data['test']
        label_dict = {0: 'Negative', 1: 'Positive'}
        support_meta = []
        for s in support:
            support_meta.append({'question': s['text'], 'answer': label_dict[s['label']]})
        query_meta = []
        for q in query:
            query_meta.append({'question': q['text'], 'answer': label_dict[q['label']]})
        query_meta = query_meta[:1000]
    elif dataset == 'trec':
        from datasets import load_dataset
        data = load_dataset('trec')
        support, query = data['train'], data['test']
        label_dict = {0: 'ABBR', 1: 'ENTY', 2: 'DESC', 3: 'HUM', 4: 'LOC', 5: 'NUM'}
        support_meta = []
        for s in support:
            support_meta.append({'question': s['text'], 'answer': label_dict[s['coarse_label']]})
        query_meta = []
        for q in query:
            query_meta.append({'question': q['text'], 'answer': label_dict[q['coarse_label']]})
    elif dataset == 'mit_movies_director':
        field_name = "Director"
        all_fields = ["Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
        assert field_name in all_fields
        all_fields.remove(field_name)
        filter_tags = [f"B-{field}" for field in all_fields] + [f"I-{field}" for field in all_fields] + ["O"]
        target_tags = [f"B-{field_name}", f"I-{field_name}"]

        with open(f'{dataDir}/{dataset}/train', 'r') as f:
            lines = f.readlines()
            lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
        support_meta = []
        for line in lines:
            answer = ''
            untagged_line = ''
            for word in line.split(' '):
                contains_target = [tag in word for tag in target_tags]
                if np.any(contains_target):
                    for tag in target_tags:
                        word = word.replace(':' + tag, '')
                    answer += word + ' '
                for tag in filter_tags:
                    word = word.replace(':' + tag, '')
                untagged_line += word + ' '

            if answer != '':
                support_meta.append({'question': untagged_line.strip(), 'answer': answer.strip()})

        query_meta = []
        with open(f'{dataDir}/{dataset}/test', 'r') as f:
            lines = f.readlines()
            lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
            
        for line in lines:
            answer = ''
            untagged_line = ''
            for word in line.split(' '):
                contains_target = [tag in word for tag in target_tags]
                if np.any(contains_target):
                    for tag in target_tags:
                        word = word.replace(':' + tag, '')
                    answer += word + ' '
                for tag in filter_tags:
                    word = word.replace(':' + tag, '')
                untagged_line += word + ' '

            if answer != '':
                query_meta.append({'question': untagged_line.strip(), 'answer': answer.strip()})
    elif dataset in ['open_mi_captioned', 'open_fvqa_captioned', 'math_induction_text', 'math_induction_text_interleaved', 'clevr_simple_text', 'cobsat_text', 'open_t2i_mi_text', 'matching_mi_text', 'matching_mi_2_text']:
        with open(f'{dataDir}/{dataset}/query.json', 'r') as f:
            query_meta = json.load(f)
        with open(f'{dataDir}/{dataset}/support.json', 'r') as f:
            support_meta = json.load(f)
    return query_meta, support_meta

def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image, mime_type