import torch
import os
import json
import argparse
import gc
from utils import model_inference, utils, ICL_utils, load_models


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Inference')

    parser.add_argument('--dataDir', default='./VL-ICL', type=str, help='Data directory.')
    parser.add_argument('--dataset', default='operator_induction', type=str, choices=['operator_induction', 'textocr', 'open_mi', 
                                                                             'clevr','operator_induction_interleaved', 'matching_mi',])
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-80b-instruct', 'gpt4v'],
                        default=["llava16-7b"], nargs="+")
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=15, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='nothing', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()


def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens

    for query in query_meta:
        
        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, query=query)

        predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                      n_shot_support, data_path, processor, max_new_tokens)
        query['prediction'] = predicted_answer
        results.append(query)

    return results
    

if __name__ == "__main__":
    args = parse_args()

    query_meta, support_meta = utils.load_data(args)
    
    for engine in args.engine:

        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        
        utils.set_random_seed(args.seed)
        for shot in args.n_shot:
            results_dict = eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, int(shot))
            os.makedirs(f"results/{args.dataset}", exist_ok=True)
            with open(f"results/{args.dataset}/{engine}_{shot}-shot.json", "w") as f:
                json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()