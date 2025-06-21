---
typora-root-url: ./image
---

本文将使用Unsloth和一个医疗数据集来微调Deepseek-R1蒸馏模型：DeepSeek-R1-Distill-Qwen-7B。

 ## 1、为什么要微调模型

​	大型预训练模型在通用任务上表现得很好，但在某些特定任务（如医疗问答、法律分析、客户支持等）中，可能因为缺乏领域专业知识而无法达到理想的效果。通过微调，可以让模型在这些特定领域的表现更加精准。例如：在医疗领域，如果你有大量医学文献或病历数据，你可以微调模型使其更擅长理解和回答医学相关的问题，而不是像一个通用的模型那样回答。

## 2、环境模型数据准备 

### 2.1 搭建环境

本实验在Torch:2.4.1+cu121 ；Transformers==4.49.0；Python=3.10；3090（24G)

环境下运行的；接下来我们需要搭建一些必要的库：

![image-20250227172820400](/image-20250227172820400.png)

```
pip install unsloth
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes unsloth_zoo
```

注意：在Unsloth官方文档中（https://docs.unsloth.ai/get-started/installing-+-updating/google-colab），安装Unsloth时有一个colab-new可选依赖项，要求“trl<0.9.0”。

![image-20250227172828212](/image-20250227172828212.png)

否则在后续安装过程中会出现报错：AttributeError: 'PeftModelForCausalLM' object has no attribute '_unwrapped_old_generate'`

### 2.2 下载模型

我们可以去摩达社区下载DeepSeek-R1-Distill-Qwen-7B模型，总大小约15G。（https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/files）

选择 **DeepSeek-R1-Distill-Qwen-7B** 进行微调任务的原因在于其兼顾了性能、效率和实用性。**适中的参数规模**和**蒸馏技术**使其在资源消耗和推理速度上表现优异；**Qwen 系列的语言能力**保证了其在专业领域的适应性；**支持推理过程**则提升了回答的可解释性。这些好处共同使得该模型成为微调任务的理想选择，尤其适用于医疗、教育和资源有限的应用场景。



运行以下命令：

```
pip install modelscope
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

![image-20250227173005334](/image-20250227173005334.png)

### 2.3 准备数据集

模型下载完成后需要下载数据集：medical-o1-reasoning-SFT，该数据集用于微调 HuatuoGPT-o1，这是一个为高级医学推理设计的医学大语言模型。该数据集是通过 GPT-4o 构建的，GPT-4o 会搜索可验证的医学问题的解决方案，并通过医学验证器验证这些方案。

![image-20250227173048734](/image-20250227173048734.png)

该数据集主要由三部分组成：Question，Complex_CoT，Response；

**Question**：医学领域的具体问题，通常需要深入理解和推理。

**Complex_CoT**：复杂的思维链（Chain-of-Thought），提供逐步推理的过程，展示从问题到答案的逻辑推导。

**Response**：最终的答案或解决方案。

数据集中的 Complex_CoT 与模型的推理能力直接对齐。通过微调，模型可以学习如何根据医学问题生成高质量的推理步骤，并最终输出准确的响应。这种结构化的输入-推理-输出的模式，能够充分利用模型的“think”特性，使其在推理任务中表现更优。



我们可以在摩达社区下载该数据集：[medical-o1-reasoning-SFT · 数据集](https://www.modelscope.cn/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/files)

通过以下命令即可下载数据集，一共138MB：

```
modelscope download --dataset FreedomIntelligence/medical-o1-reasoning-SFT
```

![image-20250228091129635](/image-20250228091129635.png)

## 3、微调模型

### 3.1对话基座模型

​	当下载好基座模型后，启动Unsloth，与模型进行对话，查看微调前基座模型的输出：

```
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 
dtype = None 
load_in_4bit = True 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/jiangsiyuan/deepseek/r1-qwen-7b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

**model_name**：模型的路径或名称，用于加载预训练的语言模型。

**max_seq_length**：输入序列的最大长度，控制模型接受的输入文本的最大字数。

**dtype**：数据类型，控制模型的精度以优化性能或资源消耗。

**load_in_4bit**：是否使用 4-bit 量化加载模型，以节省内存。

```
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.
### Question:
{}
### Response:
{}"""
question = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
```

**input_ids**：模型输入的 token ID 序列，表示输入文本经过 tokenizer 转换成的数字序列。

**attention_mask**：用于标识哪些 token 是有效的，哪些是填充 token。值为 1 的位置表示有效的 token，值为 0 的位置表示填充 token。

**max_new_tokens**：指定生成的最大 token 数量。这个参数控制模型生成的文本长度，不包括输入的 token 数量。

**use_cache**：是否启用缓存机制。启用时，模型会缓存中间计算结果，加速生成过程，减少计算量。



运行结果如下图所示：

![image-20250228091715106](/image-20250228091715106.png)

![image-20250228091720572](/image-20250228091720572.png)

### **3.2** **处理数据**

在这里需要加上一个EOS（End of Sequence）token在每个训练数据集的输入中，否则可能会出现无法停止的情况。

```
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'zh', split = "train[0:500]", trust_remote_code=True)
print(dataset.column_names)
dataset = dataset.map(formatting_prompts_func, batched = True)
```

这段代码的主要目的是从 medical-o1-reasoning-SFT 数据集中加载中文训练数据（前 500 个样本），并通过 formatting_prompts_func 函数将原始的 Question、Complex_CoT 和 Response 三部分格式化为一个统一的文本序列，末尾添加 EOS_TOKEN。格式化后的数据以 "text" 列的形式存储，准备用于微调 DeepSeek-R1-Distill-Qwen-7B 模型。

### 3.3 模型训练

**首先配置微调模型的相关参数:**

```
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None,
)
```

**r**：LoRA（Low-Rank Adaptation）矩阵的秩，决定了适配器层中低秩矩阵的维度。较高的 `r` 值可以提供更高的表示能力，但也会增加计算和内存开销。

**target_modules**：指定要应用 LoRA 的模型层（如 `q_proj`, `k_proj`, `v_proj`, `o_proj` 等）。这些模块一般涉及注意力机制的查询、键、值和输出的投影矩阵，LoRA 会对这些模块进行低秩适配。

**lora_alpha**：LoRA 的缩放因子，用于控制低秩适配器对模型权重更新的影响。较高的值会让 LoRA 模块的影响更大。

**lora_dropout**：LoRA 适配器中的 dropout 概率，用于减少过拟合。在训练中，dropout 通过随机丢弃神经网络的部分连接，帮助提高模型的泛化能力。

**bias**：LoRA 适配器中的偏置处理方式。可选值为 `"none"`, `"all"`, 或 `"lora"`。`"none"` 表示不调整偏置，`"all"` 表示所有层的偏置都包含在 LoRA 模块中，`"lora"` 则表示仅对 LoRA 模块内的部分偏置进行调整。

**use_gradient_checkpointing**：是否启用梯度检查点。启用后，可以在前向传播时保存中间激活，减小显存占用，尤其适合处理大模型。这里 `"unsloth"` 可能是一个特定的标识符或配置，表示特定方式的梯度检查点使用。

**random_state**：设置随机种子，用于控制随机过程（如初始化、数据划分等）。确保每次运行时可以得到相同的结果，便于实验复现。

**use_rslora**：是否使用 RSLORA（Recurrent Sparse Low-Rank Adaptation）。如果为 `True`，模型会使用稀疏的低秩适配方法，帮助降低计算复杂度。

**loftq_config**：配置 LoFTQ（Low-rank Fine-tuning with Quantization）的设置。如果为 `None`，则没有启用该功能；否则会根据提供的配置启用量化相关的设置，用于加速模型推理。



**定义训练器，配置训练器的参数:**

```
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
    )
```

**per_device_train_batch_size**：每个设备（GPU）上的训练批次大小，即每次训练时传入模型的数据量。较小的批次大小有助于减少内存使用，但可能导致训练时间延长。

**gradient_accumulation_steps**：梯度累积步数。每隔 `gradient_accumulation_steps` 个批次计算一次梯度并更新模型参数，允许使用更小的批次大小来模拟更大的批次。

**warmup_steps**：学习率预热的步数。训练开始时，学习率从 0 逐步增加，帮助模型稳定训练。一般来说，warmup 步数设置为训练总步数的一个小比例。

**max_steps**：最大训练步数。训练将持续执行，直到达到指定的步数。

**learning_rate**：学习率，控制每次权重更新的幅度。较小的学习率有助于稳定训练，但可能使训练时间更长。

**fp16**，**bf16**:选择模型训练的精度。

**logging_steps**：每隔多少步进行一次日志记录。用于控制训练过程中的日志输出频率。

**optim**：优化器类型。`"adamw_8bit"` 是 AdamW 优化器的 8 位量化版本，能减少内存占用并加速训练。

**weight_decay**：权重衰减系数，用于控制模型的复杂度，防止过拟合。较小的值有助于增加模型的泛化能力。

**lr_scheduler_type**：学习率调度类型。`"linear"` 表示学习率从初始值逐渐线性下降至零。

**seed**：随机种子，确保每次训练时使用相同的随机数生成器，从而使实验结果可复现。

**output_dir**：模型训练结果的保存目录，通常包含训练的检查点、日志等文件。

**report_to**：报告方式，`"none"` 表示不报告训练过程。如果启用，可以指定报告工具，如 WandB（Weights & Biases）等。



在配置完所有参数后，我们就可以使用以下命令开始模型的微调训练：

```
trainer_stats = trainer.train()
```

训练过程如下图所示：

![image-20250228092726321](/image-20250228092726321.png)

为了方便演示，此处只训练了60个epoch，大家在实操的时候可以适当调整epoch和学习率的大小，以求得最好的效果。

在训练结束后，训练参数会保存到我们指定的output_dir路径下：

![image-20250228093027340](/image-20250228093027340.png)

### 3.4 加载训练参数，对话微调后的模型

通过PeftModel.from_pretrained方法加载基座模型mode_path和微调模型权重lora_path。

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
mode_path = '/home/jiangsiyuan/deepseek/r1-qwen-7b'
lora_path="/home/jiangsiyuan/deepseek/SFT_medical/unsloth_trainconfig/checkpoint-60"
  # 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
   # 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    mode_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).eval()
model = PeftModel.from_pretrained(model, model_id=lora_path)
    # 将模型移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

将模型加载好后设置输入格式，使用model.generate与模型进行对话，代码通过循环接收用户输入的问题（医学相关），利用预定义的提示模板（prompt_style）格式化输入，并结合分词器（tokenizer）和预训练模型（model）生成逻辑清晰的回答：

```
# 设置输入格式
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.
### Question:
{}
### Response:
{}"""
# 循环接收输入并生成响应
while True:
    question = input("我：")  
    if question.strip() == "stop":
        break
    # 格式化输入并封装成模型需要的格式
    input_text = prompt_style.format(question, "")
    #print(f"Type of input_text: {type(input_text)}, Value: {input_text}")
    # 转换输入为 token ID 并移动到 GPU
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # 生成回复
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=2048,  # 限制生成长度
        use_cache=True,
    )
    # 解码生成的输出并打印响应
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response_text:
        response_text = response_text.split("assistant")[-1].strip()
    print("Deepseek: ", response_text)
```

**提示模板设置**：通过 prompt_style 定义输入格式，要求模型扮演医学专家角色，在回答前进行逐步推理（Chain-of-Thought, CoT），确保回答的逻辑性和准确性。

**交互式问答**：使用 while True 循环接收用户输入的问题，输入 "stop" 可退出。

**输入处理**：将用户问题嵌入模板，转化为模型可处理的 token ID 格式，并移至 GPU（device）加速计算。

**响应生成**：调用模型的 generate 方法生成回答，限制最大生成长度为 2048 个 token，使用缓存提高效率。

**输出解析**：解码模型生成的 token，去除特殊标记，若包含 "assistant" 标记则提取其后的内容，最终打印响应。



下面是微调后的模型推理结果；通过对比微调前后同一个问题的推理结果，可以发现微调后的模型回答更有逻辑性。

![image-20250320112616369](/image-20250320112616369.png)