import torch
import os
import json
import argparse
import gc
from utils import model_inference, utils, ICL_utils, load_models


def parse_args():
    parser = argparse.ArgumentParser(description='T2I Evaluation')

    parser.add_argument('--dataDir', default='./VL-ICL', type=str, help='Data directory.')
    parser.add_argument('--dataset', default='open_t2i_mi', type=str, choices=['open_t2i_mi', 'cobsat', 'fast_attr_t2i', 'fast_count_t2i'])
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')

    parser.add_argument("--engine", "-e", choices=['emu2-gen', 'emu1-gen', 'gill', 'seed-llama-14b', 'seed-llama-8b'],
                        default=["emu2-gen"], nargs="+")
    parser.add_argument('--max-new-tokens', default=15, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='nothing', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()


def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens
    image_save_path = f"{data_path}/{args.dataset}/prediction/{engine}/{n_shot}-shot/"
    os.makedirs(image_save_path, exist_ok=True)

    for query in query_meta:
        try:
            img_id = query['image']
        except:
            img_id = query['id'] + '.jpg'

        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, query=query)

        predicted_answer = model_inference.ICL_T2I_inference(args, engine, model, tokenizer, query, 
                                                      n_shot_support, data_path, processor, max_new_tokens)
        save_path = f"{image_save_path}/{img_id.split('/')[-1]}"
        predicted_answer.save(save_path)
        query['prediction'] = save_path
        results.append(query)

    return results
    

if __name__ == "__main__":
    args = parse_args()

    query_meta, support_meta = utils.load_data(args)
    
    for engine in args.engine:

        model, tokenizer, processor = load_models.load_t2i_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        
        utils.set_random_seed(args.seed)
        for shot in args.n_shot:
            results_dict = eval_questions(args, query_meta, support_meta, model, 
                                          tokenizer, processor, engine, int(shot))
            os.makedirs(f"results/{args.dataset}", exist_ok=True)
            with open(f"results/{args.dataset}/{engine}_{shot}-shot.json", "w") as f:
                json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()