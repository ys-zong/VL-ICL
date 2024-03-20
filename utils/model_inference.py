import torch
try:
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
except:
    pass

import os
import time
from PIL import Image
from .ICL_utils import get_task_instruction, format_answer
from .utils import load_image, encode_image


def ICL_I2T_inference(args, engine, dataset, model, tokenizer, query, 
                      n_shot_support, data_path, processor, max_new_tokens):
    task_instruction = get_task_instruction(args)
    img_id = query['image']
    query_images, query_image_paths = load_image(img_id, data_path)
    query_text = query['question']
    if 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                inputs.append({'image': os.path.join(data_path, image_path)})
            inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
                            '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        
        for query_image_path in query_image_paths:
            inputs.append({'image': query_image_path})
        inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'llava' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"
        image_tensor = torch.stack(
                [
                    processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                    for image_file in images
                ]
            )
        image_tensor = image_tensor.half().cuda()
        conv_mode = 'llava_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                )
        input_token_len = input_ids.shape[1]
        predicted_answers = tokenizer.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]

    elif 'flamingo' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
        for query_image in query_images:
            images.append(query_image)
            input_text += "<image>"
            
        vision_x = [processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        input_text += f"{query_text}\nAnswer:"
        
        lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(torch.bfloat16).cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_token_len = lang_x['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'otter' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"User: {n_shot_support[i]['question']}\nGPT:<answer> {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
        for query_image in query_images:
            images.append(query_image)
            input_text += "<image>"
        input_text += f"User: {query_text}\nGPT:<answer>"

        vision_x = processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = model.text_tokenizer(
            [
                input_text,
            ],
            return_tensors="pt",
        )
        bad_words_id = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x["input_ids"].to(model.device),
                attention_mask=lang_x["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                bad_words_ids=bad_words_id,
            )
        input_token_len = lang_x['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'internlm-x' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                image = Image.open(os.path.join(data_path, image_path)).convert("RGB")
                image = model.vis_processor(image)
                images.append(image)
                input_text += "<ImageHere>"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        for query_image in query_images:
            images.append(model.vis_processor(query_image))
            input_text += "<ImageHere>"
        input_text += f"{query_text}\nAnswer:"
        image = torch.stack(images).to(torch.bfloat16).cuda()
        predicted_answers, history = model.chat(tokenizer, query=input_text, image=image, history=[], do_sample=False, max_new_tokens=max_new_tokens)
    elif 'emu2-chat' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "[<IMG_PLH>]"
            input_text += f"[{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}]."
        for query_image in query_images:
            images.append(query_image)
            input_text += "[<IMG_PLH>]"
        input_text += f"[{query_text}\nAnswer:"
        inputs = model.build_input_ids(
            text=[input_text],
            tokenizer=tokenizer,
            image=images
        )
        
        with torch.no_grad():
            predicted_answers = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,)
        predicted_answers = tokenizer.decode(predicted_answers[:, :].cpu()[0], skip_special_tokens=True)
        
    elif 'idefics' in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                prompts.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
            prompts.append(f"\nUser: {n_shot_support[i]['question']}")
            #prompts.append("<end_of_utterance>")
            prompts.append(f"\nAssistant: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n")
        for query_image in query_images:
            prompts.append(query_image)
        prompts.append(f"\nUser: {query_text}")
        #prompts.append("<end_of_utterance>")
        prompts.append("\nAssistant:")
        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = model.generate(**inputs, 
                                       eos_token_id=exit_condition, 
                                       bad_words_ids=bad_words_ids, 
                                       max_new_tokens=max_new_tokens,
                                       do_sample=False)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(generated_ids[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'gpt4v' in engine:
        import openai
        from openai import OpenAI
        # configure your openai key by `export OPENAI_API_KEY=""` in command line
        api_key = os.environ['OPENAI_API_KEY']
        client = OpenAI(api_key=api_key)
        task_instruction = get_task_instruction(args)
        img_id = query['image']
        query_images, query_image_paths = load_image(img_id, data_path)
        query_text = query['question']
        
        content = [{
                "type": "text",
                "text": f"{task_instruction}\nEnsure the generated answers only contain the answer to the question and no other information."
            }]
        for item in n_shot_support:
            for image_path in item['image']:
                base64_image, mime_type = encode_image(os.path.join(data_path, image_path))
                content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "low"},
                })
            content.append({
                    "type": "text",
                    "text": item['question']
            })
            content.append({
                    "type": "text",
                    "text": "The answer is " + str(item['answer'])
            })
        for query_image_path in query_image_paths:
            base64_image, mime_type = encode_image(os.path.join(data_path, query_image_path))
            content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}",
                                  "detail": "low"},
                    
            })
        content.append({
                "type": "text",
                "text": query_text + " The answer is"
        })
        messages = [{
            "role": "user",
            "content": content
        }]
        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=max_new_tokens,
                )
                predicted_answers = response.choices[0].message.content
                print(query['id'], '\t', predicted_answers)
                break
            except openai.RateLimitError as e:
                print("Rate limit reached, waiting for 1 hour")
                time.sleep(3600)  # Wait for 1 hour (3600 seconds)
                continue
            except Exception as e:
                print("pausing")
                time.sleep(1)
                continue

    return predicted_answers

def ICL_T2I_inference(args, engine, model, tokenizer, query, n_shot_support, data_path, processor, max_new_tokens):
    task_instruction = get_task_instruction(args)
    query_text = query['question']
    if engine == 'emu2-gen':
        prompt = [task_instruction]
        for i in range(len(n_shot_support)):
            prompt.append(f"{n_shot_support[i]['question']}")
            image = Image.open(os.path.join(data_path, n_shot_support[i]['image'])).convert("RGB")
            prompt.append(image)
        
        prompt.append(query_text)

        outputs = model(prompt)
        predicted_answers = outputs.image
    elif engine == 'emu1-gen':
        prompt = [task_instruction]
        for i in range(len(n_shot_support)):
            prompt.append(f"{n_shot_support[i]['question']}")
            image = Image.open(os.path.join(data_path, n_shot_support[i]['image'])).convert("RGB")
            prompt.append(image)
            
        prompt.append(query_text)

        predicted_answers = model(prompt, height=512, width=512, guidance_scale=10.)
    elif engine == 'gill':
        prompt = [task_instruction]
        for i in range(len(n_shot_support)):
            prompt.append(f"{n_shot_support[i]['question']}")         
            image = Image.open(os.path.join(data_path, n_shot_support[i]['image'])).convert("RGB")
            prompt.append(image)
            
        prompt.append(query_text)

        return_outputs = model.generate_for_images_and_texts(
            prompt, num_words=2, ret_scale_factor=100.0)
        text_output, image_output = return_outputs
        predicted_answers = image_output['gen'][0][0]
    elif 'seed-llama' in engine:
        def generate(tokenizer, input_tokens, model, max_new_tokens):
            input_ids = tokenizer(
                input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
            input_ids = input_ids.to("cuda")
            generate_ids = model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            generate_ids = generate_ids[0][input_ids.shape[1]:]
            return generate_ids
        
        def decode_image(generate_ids, tokenizer):
            eoi_list = torch.where(generate_ids == tokenizer(
                EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
            eoi_index = eoi_list[0]
            image_ids = (generate_ids[:eoi_index] -
                        image_id_shift).reshape(1, -1)
            images = tokenizer.decode_image(image_ids)
            images = images[0]          
            return images

        def preprocess_image(image):
            image_tensor = processor(image).to(torch.bfloat16).cuda()
            img_ids = tokenizer.encode_image(image_torch=image_tensor)
            img_ids = img_ids.view(-1).cpu().numpy()
            img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item)
                                            for item in img_ids]) + EOI_TOKEN
            return img_tokens

        s_token, e_token, sep = "[INST] ", " [/INST]", "\n"

        BOI_TOKEN, EOI_TOKEN, IMG_TOKEN = '<img>', '</img>', '<img_{:05d}>'

        image_id_shift = 32000
        input_tokens = tokenizer.bos_token  + s_token + task_instruction + sep
        for i in range(len(n_shot_support)):
            input_tokens += n_shot_support[i]['question']
            image = Image.open(os.path.join(data_path, n_shot_support[i]['image'])).convert("RGB")
            img_tokens = preprocess_image(image)  
            input_tokens += img_tokens
            
        input_tokens += query_text
        input_tokens = input_tokens + e_token + sep + BOI_TOKEN 

        generated_ids = generate(tokenizer, input_tokens, model, max_new_tokens)
        predicted_answers = decode_image(generated_ids, tokenizer)

    return predicted_answers