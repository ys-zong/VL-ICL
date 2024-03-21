# VL-ICL

[[Webpage]](https://ys-zong.github.io/VL-ICL/)[[Paper]](https://arxiv.org/abs/2403.13164) [[Data]](https://huggingface.co/datasets/ys-zong/VL-ICL)

VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning.


## Data Preparation
We host the our dataset at HuggingFace [here](https://huggingface.co/datasets/ys-zong/VL-ICL).
```bash
git lfs install
git clone https://huggingface.co/datasets/ys-zong/VL-ICL
cd VL-ICL
bash unzip.sh
cd ..
```

## Environment
Different conda environments may be needed for different models.

```bash
conda create -n {env_name} python==3.10 -y
pip install -r requirements/{model.txt}
conda activate {env_name}
```
Replace `{model.txt}` with corresponding file.

Most of the models can be automatically downloaded from Huggingface. For Text-to-image models (Emu1, Emu2, GILL, SEED-LLaMA), please see here for detailed instructions.

## I2T

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine {model_name} --n_shot {shots} --dataset {dataset_name} --task_description detailed 
```
For example,
```bash
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 1 2 4 5 --task_description detailed --dataset open_mi
```

### Evaluation
```bash
python I2T_evaluate.py  --dataset {dataset_name} --engine {model_name} --n_shot {shots}
```

## T2I

### Inference
```bash
CUDA_VISIBLE_DEVICES=0 python T2I_inference.py --engine {model_name} --n_shot {shots} --dataset {dataset_name} --task_description detailed 
```

For example,
```bash
CUDA_VISIBLE_DEVICES=0 python T2I_inference.py --engine emu1-gen --n_shot 0 1 2 4 5 --task_description detailed --dataset open_t2i_mi
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python T2I_evaluate.py --dataset open_t2i_mi  --engine seed-llama
```

## Citation
```
@article{zong2024vlicl,
  title={VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning},
  author={Zong, Yongshuo and Bohdal, Ondrej and Hospedales, Timothy},
  journal={arXiv preprint arXiv:2403.13164},
  year={2024}
}
```