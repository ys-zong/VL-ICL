import numpy as np
import re
import torch
import os


def eval_scores(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset in ['textocr', 'operator_induction', 'clevr', 'open_mi',
                    'operator_induction_interleaved']:
        score = exact_match(results, dataset)
    elif dataset == 'matching_mi':
        score = exact_yes_no(results)
    elif dataset == 'open_t2i_mi' or dataset == 'operator_induction_t2i' or dataset == 'fast_attr_t2i' or dataset == 'fast_count_t2i':
        score = llava_judge_t2i(results, model, tokenizer, processor, dataset)
    elif dataset == 'cobsat':
        score = llava_judge_cobsat(results, model, tokenizer, processor)
    return score

def exact_yes_no(results):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if result['answer'].lower() == 'yes' and 'yes' in str(prediction).lower():
            acc.append(1)
        elif result['answer'].lower() == 'no' and 'yes' not in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_in_match(results):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if str(result['answer']).lower() in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_match(results, dataset):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if 'operator_induction' in dataset or 'clevr_simple' in dataset:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                prediction = ''

        if str(prediction).lower() == str(result['answer']).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def llava_judge_open_t2i_mi(results, model, tokenizer, processor):
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token
    from PIL import Image

    acc = []
    for result in results:
        prompt = f"Decide whether the image contains {result['answer']}. Answer with 'yes' or 'no'.\n"
        image = Image.open(result['prediction']).convert('RGB')
        image_tensor =  processor.preprocess([image], return_tensors='pt')['pixel_values'].cuda().half()
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        
        conv_mode = 'llava_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                do_sample=False,
                temperature=1,
                max_new_tokens=5,
                min_new_tokens=1,
                )
            
        input_token_len = input_ids.shape[1]
        predicted_answers = tokenizer.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
        if 'yes' in predicted_answers.lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def llava_judge_cobsat(results, model, tokenizer, processor):
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token
    from PIL import Image

    acc = []
    acc_latent = []
    acc_non_latent = []
    for result in results:
        accs = []
        for answer in [result['answer'][0], result['answer'][1]]:
            prompt = f"Decide whether the image contains the following concept: {answer}. Answer with 'yes' or 'no'.\n"
            image = Image.open(result['prediction']).convert('RGB')
            image_tensor =  processor.preprocess([image], return_tensors='pt')['pixel_values'].cuda().half()
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
            conv_mode = 'llava_v1'
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    do_sample=False,
                    temperature=1,
                    max_new_tokens=5,
                    min_new_tokens=1,
                    )
                
            input_token_len = input_ids.shape[1]
            predicted_answers = tokenizer.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
            if 'yes' in predicted_answers.lower():
                accs.append(1)
            else:
                accs.append(0)
        acc_latent.append(accs[0])
        acc_non_latent.append(accs[1])
        if accs[0] == 1 and accs[1] == 1:
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    avg_acc_latent = np.average(acc_latent)
    avg_acc_non_latent = np.average(acc_non_latent)
    return {'total': avg_acc, 'latent': avg_acc_latent, 'non_latent': avg_acc_non_latent}
