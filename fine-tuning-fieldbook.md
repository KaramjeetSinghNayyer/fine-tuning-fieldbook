# Fine-tuning Fieldbook

A practical guide to open-source language model fine-tuning.

By **Karamjeet Singh (IITG CSE)**

Contact: [ksn2939@gmail.com](mailto:ksn2939@gmail.com) · [+91 9123701916](tel:+919123701916)

Teaching code targets the pinned dependencies in the local project. Training has not been executed by the author. The 30-example synthetic dataset is a pipeline demonstration, not evidence of model quality.

## What fine-tuning actually changes

Start with the mental model, before the machinery.

### A model is a learned function

A language model maps a sequence of tokens to probabilities for the next token. Its parameters—often called weights—are the numbers learned during training. Pretraining builds broad language ability from a large corpus. Fine-tuning starts from those existing weights and trains on a narrower set of examples.

### The fine-tuning loop

1. Your examples
2. Tokenize
3. Predict & measure loss
4. Update trainable weights

### Teach a repeatable behavior

Imagine a support inbox. You want each request classified as BILLING, ACCOUNT, or TECHNICAL. A training example pairs a request with the label you want. Fine-tuning nudges the model toward this response pattern. It does not install a database of guaranteed facts, and an adapter can still make mistakes.

### Choose the right tool

| Approach | What changes | Use it when |
| --- | --- | --- |
| Prompting | The input text | Instructions and a few examples already work. |
| Retrieval (RAG) | Context fetched at inference | Answers depend on changing documents or source citations. |
| Fine-tuning | Trainable parameters | You have repeated behavior to teach and representative examples. |

### Full fine-tuning versus adapters

Full fine-tuning updates the base model weights. Parameter-efficient fine-tuning freezes most or all of the base and adds a small trainable component. LoRA is one adapter method; it reduces the number of trained parameters while still influencing the model’s outputs.

### Before training

Write down a measurable goal and evaluate the base model with a good prompt. If the baseline already meets your goal, fine-tuning may add cost without a useful gain.

### A concrete training example

One row of samples.jsonl

```json
{
  "text": "I was billed twice.",
  "label": "BILLING",
  "split": "train"
}
```

**What the code does**

- **text:** The request the model will see.
- **label:** The desired completion; consistency is essential.
- **split:** Our project stores the split explicitly to keep evaluation data separate.

### What you should be able to explain

Fine-tuning changes learned behavior through training. Prompting changes instructions, and retrieval changes available context. These approaches can be combined; choosing one does not exclude the others.

Primary sources: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [LoRA paper](https://arxiv.org/abs/2106.09685).

## Tokens, loss, and learning

Follow a single example through the training loop.

### Tokens are the model’s alphabet

A tokenizer converts text into integer IDs. A token may represent a word, part of a word, punctuation, or another text fragment. The model operates on these IDs. Always use the tokenizer belonging to your base model: an ID has meaning only in that vocabulary.

### Next-token prediction

For the sequence “The sky is blue”, the model learns to predict each next token from the tokens before it. During training the correct previous tokens are supplied; this is called teacher forcing. At inference, the model consumes its own generated tokens instead.

### Cross-entropy, in plain English

L = − mean(log p(correct next token))

If the correct token has probability 0.8, its loss is about 0.223. At probability 0.1, its loss is about 2.303. Lower loss means more probability assigned to the target. This is a training signal, not a complete measure of usefulness.

### Feel what a loss function measures

Standalone example

```python
import torch

probability = torch.tensor([0.8, 0.1])
loss = -torch.log(probability)
print(loss.tolist())
```

**What the code does**

- **torch.tensor:** Creates a numeric array managed by PyTorch.
- **torch.log:** Computes the natural logarithm element by element.
- **The minus sign:** Turns small probabilities into larger penalties.
- **Expected output:** Approximately [0.2231, 2.3026]. This demonstrates the formula, not a full model training step.

### Backpropagation connects errors to parameters

A forward pass produces predictions and a loss. Backpropagation computes gradients: how the loss would change if each trainable parameter moved slightly. An optimizer uses these gradients to adjust parameters. The learning rate controls the size of an update.

### One real PyTorch update

Standalone example

```python
import torch

torch.manual_seed(42)
layer = torch.nn.Linear(2, 1)
optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
x = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[3.0]])

optimizer.zero_grad()
prediction = layer(x)
loss = torch.nn.functional.mse_loss(prediction, target)
loss.backward()
optimizer.step()
print("Loss before update:", loss.item())
```

**What the code does**

- **nn.Linear:** A tiny layer with learned weights and a bias. This is a toy regression model, not an LLM.
- **AdamW:** Tracks gradient statistics and updates the layer parameters.
- **zero_grad:** Clears old gradients so they do not accumulate accidentally.
- **mse_loss:** Measures squared prediction error for this numeric toy example. LLMs typically use cross-entropy instead.
- **backward / step:** Compute gradients, then change parameters. The printed loss belongs to the forward pass before this update.

### The distinction that matters

Training uses gradients to change parameters. Inference uses the current parameters to generate an answer. Merely calling generate() does not fine-tune anything.

Primary sources: [PyTorch optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [Chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating).

## Your local Python workspace

Know what each library contributes and prepare VS Code.

### The course project

We use Qwen2.5-0.5B-Instruct, a small public instruction model, and train a LoRA adapter to output one of three support labels. Download the local project from the top bar and open the extracted folder in VS Code. You need Python 3.11, internet for the first model download, and several GB of free memory and disk space.

### Hardware and verification

The main project uses float32 and no quantization. CPU execution is possible but slow; CUDA or Apple MPS can help when available. Exact memory and runtime depend on your machine. The code has been syntax-checked; model training and package integration have not been executed here.

### Create an isolated environment

Terminal · run in the extracted project folder

```shell
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**What the code does**

- **venv:** Keeps this project’s packages separate from other Python projects.
- **Activate:** Selects the environment in your terminal. In VS Code, also select its Python interpreter.
- **pip:** Installs Python packages. Windows can use python instead of python3.

### Install PyTorch for your computer

Use the official PyTorch installation selector at pytorch.org/get-started/locally/ to choose your operating system and accelerator. Run its pip command inside the environment. Then install the teaching dependencies below. These are intentionally pinned teaching versions rather than a claim about the newest releases.

### Teaching dependencies

requirements.txt · included in the download

```text
# Install PyTorch separately for your platform using pytorch.org/get-started/locally/.
transformers==4.57.1
peft==0.17.1
datasets==4.0.0
accelerate==1.10.1

```

**What the code does**

- **transformers:** Loads the tokenizer and pretrained causal language model; Trainer coordinates training.
- **peft:** Adds LoRA parameters and saves or reloads adapters.
- **datasets:** Turns rows into an indexed dataset and applies preprocessing.
- **accelerate:** Supports device handling used by Trainer. It does not make every workload automatically fast.
- **PyTorch:** Provides tensors, gradients, neural-network operations, and device support. Install it separately for your hardware.

### Install and run

Terminal · after installing PyTorch

```shell
python -m pip install -r requirements.txt
python train.py
python infer.py
```

**What the code does**

- **train.py:** Loads the base model, measures baseline predictions, trains, and saves the adapter and results.
- **infer.py:** Reloads the saved adapter and generates a label. Run after training completes.
- **First run:** Downloads model files. Output and execution time vary with hardware; accuracy is not guaranteed.

### Why this course uses Trainer

We keep tokenization and the loss mask explicit, then let Transformers Trainer handle batching, optimization, checkpointing, and evaluation. TRL’s SFTTrainer is a higher-level alternative for supervised fine-tuning, but it is not required by this project.

Primary sources: [Transformers Trainer · v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [PEFT documentation](https://huggingface.co/docs/peft/index), [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## Build a dataset worth learning from

Quality, splits, and the shape of a useful example.

### Examples define the behavior

Choose one precise task. Our labels are mutually exclusive, uppercase, and unambiguous in the sample requests. In real support traffic, a message may mention several topics. Decide a priority rule or add an escalation category before labelling, and document it for annotators.

### Three splits, three jobs

| Split | Used for | Our tiny demo |
| --- | --- | --- |
| Train | Optimizer updates | 18 examples |
| Validation | Selecting a checkpoint and tuning settings | 6 examples |
| Test | Final baseline versus adapted comparison | 6 examples |

### A pipeline demo, not evidence of quality

Thirty synthetic examples make the flow easy to inspect. They are too few to establish reliability. Replace them with representative, permissioned data before drawing conclusions. There is no universal minimum dataset size: coverage, consistency, and task complexity matter.

### Read and check the JSONL file

Standalone check · requires samples.jsonl

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in
        Path("samples.jsonl").read_text().splitlines() if line.strip()]
assert all(r["label"] in {"BILLING", "ACCOUNT", "TECHNICAL"} for r in rows)
assert len({r["text"] for r in rows}) == len(rows)
print("Examples:", len(rows))
```

**What the code does**

- **JSONL:** One complete JSON object per line, making examples easy to inspect and append.
- **Path.read_text:** Reads the local file into memory.
- **json.loads:** Parses each line into a Python dictionary.
- **assert:** Stops on invalid labels or exact duplicate request strings. This does not catch near duplicates or label errors.

### Avoid leakage

Keep copies and near copies of the same request out of different splits. Group messages from the same conversation, customer, or template before splitting. For time-sensitive tasks, a chronological test can better reflect deployment. Never train on test labels or use the test score repeatedly to choose hyperparameters.

### Audit more than formatting

Check label balance, realistic spelling, long requests, unfamiliar wording, ambiguous cases, and relevant languages. Remove private information unless you have a valid reason and permission to retain it. A beautifully formatted dataset with bad labels teaches bad behavior.

### Keep a record

Save your task definition, source provenance, annotation rules, split method, and dataset version. Record the model and tokenizer revision too. Seeds help repeat an experiment, but different hardware and kernels can still produce different results.

Primary sources: [Transformers Trainer · v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).

## Turn text into training tensors

Understand chat templates, attention masks, and label masks.

### Chat formatting belongs to the model

Instruction models expect conversation markers and role delimiters in a particular format. apply_chat_template uses the tokenizer’s template rather than inventing generic separators. We format a system instruction and user request, then add the assistant-generation prefix. The target completion is the label followed by the end-of-sequence token.

### Format the prompt

Standalone example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "system", "content": "Reply with BILLING, ACCOUNT, or TECHNICAL only."},
    {"role": "user", "content": "I was billed twice."},
]
ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                    add_generation_prompt=True)
print(tokenizer.decode(ids))
```

**What the code does**

- **AutoTokenizer:** Loads the tokenizer configuration for this model.
- **Role dictionaries:** Describe who is speaking; the template supplies the actual special markers.
- **add_generation_prompt:** Ends the prompt where the assistant’s answer should begin.
- **decode:** Turns IDs back into visible text so you can inspect the format.

### The three arrays

| Tensor | Meaning | Example |
| --- | --- | --- |
| input_ids | Prompt, answer, and padding token IDs | [prompt …, answer …, pad …] |
| attention_mask | Which positions are real input tokens | [1 …, 1 …, 0 …] |
| labels | Which tokens should contribute to the loss | [-100 …, answer …, -100 …] |

### Encode a completion-only example

Excerpt from train.py · requires tokenizer and prompt_ids

```python
def encode(row):
    prefix = prompt_ids(row["text"])
    answer = tokenizer(row["label"], add_special_tokens=False)["input_ids"]
    answer += [tokenizer.eos_token_id]
    ids = prefix + answer
    if len(ids) > 256:
        raise ValueError("Example exceeds 256 tokens; shorten it or raise the limit.")
    padding = 256 - len(ids)
    return {
        "input_ids": ids + [tokenizer.pad_token_id] * padding,
        "attention_mask": [1] * len(ids) + [0] * padding,
        "labels": [-100] * len(prefix) + answer + [-100] * padding,
    }

```

**What the code does**

- **prefix:** The prompt IDs are visible to the model but receive no direct loss.
- **answer:** The target label IDs plus an EOS token teach the output and when to stop.
- **-100:** The loss function’s ignore index masks prompt and padding labels. It is not a vocabulary token.
- **256 tokens:** Fixed padding makes ordinary batching simple. Overlong examples raise an error instead of silently dropping the answer.
- **Causal shift:** The causal language model shifts predictions against next-token labels internally. Do not shift these labels again.

### Two masks, different purposes

The attention mask describes real input positions. The label mask decides what is scored by the loss. Do not mask every occurrence of the EOS ID just because padding uses the same ID: the real answer’s EOS must remain a target.

### Why padding is simple here

All samples are padded to 256 tokens. This wastes some computation but makes the first project easy to reason about. A larger project can use dynamic padding with a collator that also pads labels with −100; measure throughput before adding packing or other optimizations.

Primary sources: [Chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating), [Transformers Trainer · v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).

## LoRA: learn a smaller update

The equation, the adapter, and the knobs that matter.

### A small detour around frozen weights

A dense layer transforms input using a weight matrix W. LoRA keeps W frozen and learns a low-rank update through two smaller matrices, A and B. The adapter changes the layer’s output without training every number in W.

### The adapter update

W′ = W + (α / r) BA

For W with shape d_out × d_in, A has shape r × d_in and B has shape d_out × r. The added parameter count is r(d_in + d_out). The rank r is deliberately small.

### A concrete size comparison

For a 4096 × 4096 matrix, full training has 16,777,216 weights. At rank 8, the two adapter matrices have 65,536 weights—about 0.39% of that matrix’s parameter count. This is arithmetic for one layer, not a promise about total GPU memory or quality.

### Attach a LoRA adapter

Excerpt from train.py · requires a loaded model

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

**What the code does**

- **LoraConfig:** Describes how to insert the adapters; it does not train them.
- **r:** The low-rank dimension. Larger ranks add capacity and trainable parameters.
- **lora_alpha:** Sets adapter scaling relative to rank in standard LoRA.
- **lora_dropout:** Randomly drops adapter inputs during training as regularization.
- **q_proj / v_proj:** Query and value projections in this Qwen architecture. Other models may use different names.
- **get_peft_model:** Wraps the base model and marks the intended adapter parameters trainable.

### A starting configuration, not a best configuration

Rank 8 and two attention projections are a modest teaching setup. Real experiments may target more attention projections and feed-forward layers. Compare configurations on validation data, with comparable training budgets; more parameters do not automatically produce better answers.

### What gets saved

An adapter checkpoint contains the learned adapter weights and configuration. It is not the whole base model. At inference, you need the same compatible base model and the tokenizer used during training.

Primary sources: [LoRA paper](https://arxiv.org/abs/2106.09685), [PEFT documentation](https://huggingface.co/docs/peft/index).

## Run the complete experiment

From baseline to checkpoint, with no missing glue.

### Read the execution order

The script fixes a seed, loads the tokenizer and model, checks the dataset, builds masked tensors, measures the base model on held-out requests, inserts LoRA, trains against the training split, chooses the checkpoint using validation loss, and saves the adapter. The final report compares predictions on the same test set.

### Training settings you should understand

| Setting | Our value | What it controls |
| --- | --- | --- |
| Learning rate | 2e−4 | The optimizer’s update scale; a starting point for this adapter. |
| Epochs | 2 | Two passes over the training split. |
| Batch size | 1 | Examples in each forward/backward microbatch. |
| Gradient accumulation | 4 | Four microbatches per update, except a possible final partial group. |
| Evaluation / saving | Every epoch | Validation measurement and checkpoint cadence. |
| Best checkpoint | Lowest validation loss | Chooses among saved checkpoints without using test accuracy. |

### Effective batch size

With one device, batch size 1 and accumulation 4 correspond to roughly four examples per optimizer update. With multiple devices, multiply by the device count. Gradient accumulation allows smaller microbatches, but it does not remove the memory needed for model weights and activations.

### Complete training script

train.py · complete runnable file in the local project

```python
"""Tiny LoRA teaching experiment. Run: python train.py
Not a production dataset or a promise of improved accuracy.
"""
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model

set_seed(42)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Keep revision fixed to a Hub commit for stricter reproducibility.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32
)
# Float32 is a simple CPU/MPS/CUDA baseline, but uses more memory.
# Trainer moves the model to the available device.
model.config.use_cache = False

SYSTEM = "Classify the support request. Reply with BILLING, ACCOUNT, or TECHNICAL only."
def prompt_ids(text):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True,
    )

def encode(row):
    prefix = prompt_ids(row["text"])
    answer = tokenizer(row["label"], add_special_tokens=False)["input_ids"]
    answer += [tokenizer.eos_token_id]
    ids = prefix + answer
    if len(ids) > 256:
        raise ValueError("Example exceeds 256 tokens; shorten it or raise the limit.")
    padding = 256 - len(ids)
    return {
        "input_ids": ids + [tokenizer.pad_token_id] * padding,
        "attention_mask": [1] * len(ids) + [0] * padding,
        "labels": [-100] * len(prefix) + answer + [-100] * padding,
    }

rows = [json.loads(line) for line in Path("samples.jsonl").read_text().splitlines() if line.strip()]
assert all(r["label"] in {"BILLING", "ACCOUNT", "TECHNICAL"} for r in rows)
assert len({r["text"] for r in rows}) == len(rows), "Duplicate request"
splits = {s: [r for r in rows if r["split"] == s]
          for s in ("train", "validation", "test")}
assert all(splits.values()), "Every split must contain examples"
data = {s: Dataset.from_list(rs).map(encode, remove_columns=["text", "label", "split"])
        for s, rs in splits.items()}

@torch.inference_mode()
def predict(text):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids(text)], device=device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=8, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def score(rows):
    predictions = [predict(r["text"]) for r in rows]
    return {"accuracy": sum(p == r["label"] for p, r in zip(predictions, rows)) / len(rows),
            "predictions": predictions}

# Save baseline before adapting weights. Test examples are never used to train.
# Baseline and adapter scores are compared once at the end.
model.to("cuda" if torch.cuda.is_available() else
         "mps" if torch.backends.mps.is_available() else "cpu")
baseline = score(splits["test"])
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=8, lora_alpha=16,
    lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
    bias="none",
))
model.print_trainable_parameters()
args = TrainingArguments(
    output_dir="checkpoints", num_train_epochs=2,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=4, learning_rate=2e-4,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    greater_is_better=False, save_total_limit=2,
    logging_steps=1, report_to="none", optim="adamw_torch",
    seed=42, data_seed=42, label_names=["labels"],
)
trainer = Trainer(model=model, args=args, train_dataset=data["train"],
                  eval_dataset=data["validation"], processing_class=tokenizer)
trainer.train()
model.save_pretrained("adapter")
tokenizer.save_pretrained("adapter")
adapted = score(splits["test"])
report = {"baseline": baseline, "adapted": adapted,
          "test_size": len(splits["test"]),
          "note": "Tiny synthetic demo. Do not infer generalization from this score."}
Path("results.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

```

**What the code does**

- **set_seed:** Sets common random generators. Seeds improve repeatability without guaranteeing identical results everywhere.
- **model.to:** Selects CUDA, MPS, or CPU for baseline generation. Trainer handles the training device.
- **Dataset.map:** Applies encode to each row and keeps just the tensor-ready fields.
- **Trainer:** Runs the forward pass, loss, backpropagation, optimizer, evaluation, and saving.
- **load_best_model_at_end:** Restores the checkpoint with the best validation loss. The save and evaluation schedules match.
- **save_pretrained:** Writes the adapter and tokenizer locally. No Hub upload happens.
- **results.json:** Stores baseline and adapted test predictions so failures remain inspectable.

### Run it locally

The browser is a reading and copying surface. Download the local project, install its dependencies, and run python train.py in VS Code’s terminal. Small-data results may be unchanged or worse; that is a valid experiment outcome.

Primary sources: [Transformers Trainer · v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [PEFT documentation](https://huggingface.co/docs/peft/index).

## Evaluate, then reload your adapter

A lower training loss is only the beginning.

### Compare what users will actually receive

Our classifier’s exact-match accuracy is the proportion of generated strings that equal the correct label. A verbose answer counts as incorrect even if it includes the right label: the task requires a single label. Generation is deterministic with do_sample=False, and the base and adapted model use the same prompt and decoding settings.

### Inspect your saved results

Standalone check · run after train.py

```python
import json
from pathlib import Path

result = json.loads(Path("results.json").read_text())
for name in ("baseline", "adapted"):
    print(name, result[name]["accuracy"])
    print(result[name]["predictions"])
print("Test examples:", result["test_size"])
```

**What the code does**

- **accuracy:** A number between 0 and 1, based on the exact output strings.
- **predictions:** Shows what the model actually generated; inspect invalid labels and errors.
- **test_size:** Only six in this demonstration. A one-example change moves accuracy by about 16.7 percentage points.

### Use more than one score

On a representative dataset, calculate per-class precision and recall, a confusion matrix, invalid-output rate, and accuracy by important slices. Compare latency and memory as well. For open-ended tasks, use a clear rubric and human review; a lower validation loss alone does not prove a better assistant.

### Read learning curves

Training loss falling while validation loss rises can indicate overfitting. Try better examples, fewer epochs, a smaller learning rate, or reduced adapter capacity. Keep the test set untouched while choosing changes. If the baseline is already perfect, use harder representative examples rather than training until the tiny score looks impressive.

### Load and use the saved adapter

infer.py · run after train.py saves adapter/

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(base, "adapter")
tokenizer = AutoTokenizer.from_pretrained("adapter")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device).eval()
messages = [
    {"role": "system", "content": "Classify the support request. Reply with BILLING, ACCOUNT, or TECHNICAL only."},
    {"role": "user", "content": "I was charged twice this month."},
]
ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                    add_generation_prompt=True, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(ids, attention_mask=torch.ones_like(ids),
                            max_new_tokens=8, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))

```

**What the code does**

- **PeftModel.from_pretrained:** Loads the saved LoRA weights onto the base model.
- **model.eval:** Turns off training behaviors such as dropout.
- **torch.inference_mode:** Avoids building a gradient graph during generation.
- **do_sample=False:** Uses greedy decoding for a repeatable comparison.
- **Output slice:** Removes the prompt tokens and decodes only newly generated tokens.

### Share the whole recipe

For reproducible use, preserve the base-model revision, tokenizer files, adapter configuration, dependency versions, prompt, and decoding settings. Test reloading in a fresh environment before depending on the model.

Primary sources: [Transformers Trainer · v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer), [PEFT documentation](https://huggingface.co/docs/peft/index).

## QLoRA, troubleshooting, and next steps

Move beyond the first experiment with a clear map.

### What QLoRA adds

QLoRA combines low-rank adapters with a quantized frozen base model. Quantization stores base weights at lower precision to reduce their memory footprint. Adapters still train using floating-point computation. Quantized weights do not mean the entire training process uses four-bit arithmetic.

### Optional: prepare a four-bit base

Optional illustration · not a drop-in replacement for train.py

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

if not torch.cuda.is_available():
    raise RuntimeError("This example targets a compatible NVIDIA CUDA setup.")
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=quantization, device_map={"": 0},
)
model = prepare_model_for_kbit_training(model)
# Next: attach LoRA using get_peft_model, then configure training.
```

**What the code does**

- **bitsandbytes:** An additional optional package; not part of the baseline requirements. Install a version compatible with your CUDA, PyTorch, and hardware.
- **NF4:** A four-bit format designed for normally distributed weights.
- **Double quantization:** Also quantizes quantization constants to reduce storage overhead.
- **Compute dtype:** Uses BF16 when supported and FP16 otherwise; training precision must match your setup.
- **prepare_model_for_kbit_training:** Prepares the quantized base for adapter training.
- **device_map:** Places the base on GPU 0 for this single-GPU example. Do not add model.to() from the unquantized script.

### Take the ordinary LoRA path first

Four-bit support depends on backend, operating system, and package versions. This optional snippet is targeted at compatible NVIDIA CUDA systems, not a promise of Mac support. Adapt training precision and device placement before combining it with the full project.

### Diagnose common failures

| Symptom | Likely check | Useful action |
| --- | --- | --- |
| Out of memory | Sequence length, activations, batch size | Reduce sequence length or batch size; consider checkpointing or quantization. |
| No answer tokens scored | Labels are all −100 | Inspect a preprocessed example before training. |
| Loss falls, behavior gets worse | Overfitting or task mismatch | Audit data and compare validation predictions. |
| Unknown argument / import error | Version or environment mismatch | Check selected VS Code interpreter and installed versions. |
| Adapter cannot load | Wrong base model or missing files | Reload the matching base and adapter configuration. |
| Model returns extra prose | Prompt, targets, decoding limit | Check target consistency and count format failures in evaluation. |

### A practical next experiment

Replace the tiny dataset with a reviewed set of real, permitted examples. Establish a prompt baseline. Hold out representative tests, run one LoRA configuration, inspect errors, and make one justified change at a time. Record what improved, what regressed, and how much it cost.

### What comes after supervised fine-tuning

Continued pretraining adapts next-token prediction to a domain corpus. Preference methods such as DPO compare preferred and rejected responses. Reinforcement learning methods optimize a reward. They solve different problems and require different data; learn a trustworthy supervised evaluation loop first.

Primary sources: [QLoRA paper](https://arxiv.org/abs/2305.14314), [Bitsandbytes guide](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes).

