import random
import copy


def select_demonstration(support_meta, n_shot, dataset, query=None):
    if 'operator_induction' in dataset:
        operator_index = {'+': 0, '-': 1, 'x': 2}
        n_shot_support_raw = random.sample(support_meta, n_shot)
        n_shot_support = copy.deepcopy(n_shot_support_raw)
        operator = query['operator']
        operator_idx = operator_index[operator]
        for support in n_shot_support:    
            support['answer'] = support['answer'][operator_idx]
    elif dataset == 'open_mi':
        # use two classes for now
        query_class = query['answer']
        other_class = random.choice([cls for cls in query['classes'] if cls != query_class])
        order_keys = [query_class, other_class] if random.choice([True, False]) else [other_class, query_class]
        answers = {query_class: query_class, other_class: other_class}
        
        n_shot_support = []
        for i in range(n_shot):
            for key in order_keys:
                # For each key, add one shot
                support = {
                    'image': [query['support'][key]['images'][i]], 
                    'answer': answers[key],
                    'question': "This is a"
                }
                n_shot_support.append(support)
    
    elif dataset == 'matching_mi':
        n_shot_support_raw = copy.deepcopy(random.sample(support_meta, n_shot))
        n_shot_support = []
        for i in range(n_shot):
            n_shot_support.append(n_shot_support_raw[i]['same'])
            n_shot_support.append(n_shot_support_raw[i]['diff'])
    
    elif dataset == 'open_t2i_mi':
        query_class = query['task_label']
        other_class = random.choice([cls for cls in query['classes'] if cls != query_class])
        order_keys = [query_class, other_class] if random.choice([True, False]) else [other_class, query_class]
        answers = {query_class: query_class, other_class: other_class}
        n_shot_support = []
        for i in range(n_shot):
            for key in order_keys:
                # For each key, add one shot
                if dataset == 'open_t2i_mi':
                    support = {
                        'image': query['support'][key]['images'][i], 
                        'question': f'Generate a {key}'
                    }
                else:
                    support = {
                        'answer': query['support'][key]['images'][i], 
                        'question': f'Generate a {key}'
                    }
                n_shot_support.append(support)
    elif dataset == 'cobsat':
        latent_var = query['latent']
        latent = query[latent_var]
        task = query['task']
        # get support set with same latents
        n_shot_support = [x for x in support_meta if (x[latent_var] == latent and x['latent'] == latent_var and x['task'] == task)]
        n_shot_support = copy.deepcopy(random.sample(n_shot_support, n_shot))
        
    elif dataset == 'clevr':
        n_shot_support_raw = random.sample(support_meta, n_shot)
        n_shot_support = copy.deepcopy(n_shot_support_raw)
        for i in n_shot_support:
            n_shot_support['question'] = f"Image 1: {n_shot_support['question1']}\nImage 2: {n_shot_support['question2']}"
            n_shot_support['image'] = [n_shot_support['image1'], n_shot_support['image2']]
    
    else:
        n_shot_support = random.sample(support_meta, n_shot)
    return n_shot_support

def get_task_instruction(args):
    dataset = args.dataset
    description = args.task_description
    if description == 'nothing':
        instr = ''
        return instr
    
    if dataset == 'textocr':
        if description == 'concise':
            instr = 'Answer with the text inside the red box.'
        elif description == 'detailed':
            instr = 'An image will be provided where a red box is drawn around the text of interest. Answer with the text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.'
    elif dataset == 'operator_induction':
        if description == 'concise':
            instr = 'Induce the mathematical operator and calculate the result.'
        elif description == 'detailed':
            instr = 'The image contains two digit numbers and a ? representing the mathematical operator. Induce the mathematical operator (addition, multiplication, minus) according to the results of the in-context examples and calculate the result.'
    elif dataset == 'operator_induction_interleaved':
        if description == 'concise':
            instr = 'Induce the mathematical operator between the two images and calculate the result.'
        elif description == 'detailed':
            instr = 'There are two input images, each representing a single digit number. Induce the mathematical operator (addition, multiplication, minus) that is applied between the two images according to the results of the in-context examples. Calculate the result for the new query images.'
    elif dataset == 'open_mi':
        if description == 'concise':
            instr = 'Answer the question with a single word or phase.'
        elif description == 'detailed':
            instr = "Induce the concept from the in-context examples. Answer the question with a single word or phase."
    elif dataset == 'clevr':
        if description == 'concise':
            instr = 'Find objects of the given type, induce what operation to use and calculate the result.'
        elif description == 'detailed':
            instr = 'The image contains objects of different shapes, colors, sizes and materials. The question describes the attribute and its value. You need to find all objects within the image that satisfy the condition. You should induce what operation to use according to the results of the in-context examples and then calculate the result.'
    elif dataset == 'matching_mi':
        if description == 'concise':
            instr = 'Determine the output for the new pair of images.'
        elif description == 'detailed':
            instr = 'According to the few-shot examples, induce what operation to do and determine the output for the two new images. Answer with "yes" or "no".'
    
    # t2i
    elif dataset == 'cobsat':
        if description == 'concise':
            instr = 'Generate the next image based on the latent object or attribute from the few-shot examples.'
        elif description == 'detailed':
            instr = 'Based on the sequence, infer the latent object or attribute. Generate the image of the inferred object or attribute combined with the given text.'
    elif dataset == 'open_t2i_mi':
        if description == 'concise':
            instr = 'Generate the image of the given class.'
        elif description == 'detailed':
            instr = 'Based on the few-shot examples, induce the concept and generate the image of the given class.'
    
    return instr

def format_answer(answer, dataset, query=None):
    if dataset in ['operator_induction', "clevr", 'operator_induction_interleaved']:
        answer = str(answer)
    return answer