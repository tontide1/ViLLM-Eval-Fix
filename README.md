# ViLLM-Eval

We utilize the lm-eval-harness library to conduct evaluations. 
This library allows us to efficiently evaluate language models, ensuring robustness and accuracy in our assessments.
Feel free to explore our project and discover the capabilities of the language models we employ.

## Install

```bash
git clone https://huggingface.co/datasets/vlsp-2023-vllm/ViLLM-Eval
cd ViLLM-Eval
pip install -e .
```

## Basic Usage

```bash
# Add trust_remote_code=True if your model is a custom model
MODEL_ID=pretrained=vinai/PhoGPT-4B-Chat,trust_remote_code=True

# Add load_in_4bit=True or load_in_8bit=True if you want to run in INT4/INT8 mode, note that it will reduce evaluation effectiveness
MODEL_ID=pretrained=vinai/PhoGPT-4B-Chat,load_in_4bit=True
```

### LAMBADA_vi

```bash
MODEL_ID=vlsp-2023-vllm/hoa-1b4 # replace your HF model here

python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_ID \
    --tasks lambada_vi \
    --device cuda:0
```

### Exam_vi

```bash
MODEL_ID=vlsp-2023-vllm/hoa-1b4 # replace your HF model here

python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_ID \
    --tasks exams_dialy_vi,exams_hoahoc_vi,exams_lichsu_vi,exams_sinhhoc_vi,exams_toan_vi,exams_vatly_vi,exams_van_vi \
    --num_fewshot 5 \
    --device cuda:0
```

### GKQA

```bash
MODEL_ID=vlsp-2023-vllm/hoa-1b4 # replace your HF model here

python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_ID \
    --tasks wikipediaqa_vi \
    --num_fewshot 5 \
    --device cuda:0
```

### ComprehensionQA

```bash
MODEL_ID=vlsp-2023-vllm/hoa-1b4 # replace your HF model here

python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_ID \
    --tasks comprehension_vi \
    --device cuda:0
```

## Cite as

```
@misc{nguyen2024villmeval,
      title={ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large Language Models}, 
      author={Trong-Hieu Nguyen and Anh-Cuong Le and Viet-Cuong Nguyen},
      year={2024},
      eprint={2404.11086},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
