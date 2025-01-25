import json
import argparse
from evals import eval


metrics = {
    "open_t2i_mi": "Acc",
    "cobsat": "Acc",
    "fast_attr_t2i": "Acc",
    "fast_count_t2i": "Acc",
}

def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Evaluation')

    parser.add_argument('--dataDir', default='./VL-ICL', type=str, help='Data directory.')
    parser.add_argument('--dataset', default='open_t2i_mi', type=str, choices=['open_t2i_mi', 'cobsat', 'fast_attr_t2i', 'fast_count_t2i'])
    parser.add_argument("--engine", "-e", choices=['gill', 'emu2-gen', 'emu1-gen', 'seed-llama-14b', 'seed-llama-8b'],
                        default=["gill"], nargs="+")
    parser.add_argument('--max-new-tokens', default=40, type=int, help='Max new tokens for generation.')
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.dataset in ['open_t2i_mi', 'cobsat', 'fast_attr_t2i', 'fast_count_t2i']:
        from llava.model.builder import load_pretrained_model as load_llava_model
        import torch
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1.6-vicuna-7b', model_base=None, model_name='llava')
        processor = image_processor
        model = model.to(torch.bfloat16).cuda()
    else:
        tokenizer, model = None, None
    result_files = [f"results/{args.dataset}/{engine}_{shot}-shot.json" for engine in args.engine for shot in args.n_shot]

    for result_file in result_files:
        engine, shot = result_file.split("/")[-1].replace(".json", "").split("_")
        with open(result_file, "r") as f:
            results_dict = json.load(f)
        
        score = eval.eval_scores(results_dict, args.dataset, model, tokenizer, processor)
        if "cobsat" in args.dataset:
            print(f'{args.dataset} {metrics[args.dataset]} of {engine} {shot}: ',
                  f"Total: {score['total'] * 100.0:.2f}, Latent: {score['latent'] * 100.0:.2f}, Non-latent: {score['non_latent'] * 100.0:.2f}", flush=True)
        else:
            print(f'{args.dataset} {metrics[args.dataset]} of {engine} {shot}: ', f"{score * 100.0:.2f}", flush=True)