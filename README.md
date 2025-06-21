# LLM_Related
分享一些大模型相关的工具和知识，并手把手介绍如何使用

### 大模型性能评估工具
EvalScope：
是魔搭社区倾力打造的模型评测与性能基准测试框架，为您的模型评估需求提供一站式解决方案。无论您在开发什么类型的模型，EvalScope 都能满足您的需求：

🧠 大语言模型

🎨 多模态模型

🔍 Embedding 模型

🏆 Reranker 模型

🖼️ CLIP 模型

🎭 AIGC模型（图生文/视频）

。。。

https://github.com/Enermy/LLM_Related/tree/main/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AF%84%E6%B5%8B-evalscope

deepeval：深度评估（DeepEval） 是一个专为大型语言模型（LLM）输出评测设计的开源框架。它借鉴了Pytest的设计理念，但更加专注于单元测试LLM生成的答案，确保其质量符合预期标准。通过集成最新的研究，如G-Eval等评价方法，DeepEval提供了一系列指标来量化LLM响应的准确性、相关性和其他关键特性。此外，该框架具有高度模块化，便于使用者选择性地运用其内置指标，或是开发自定义评估逻辑，适用于多样化的应用场景。

https://github.com/Enermy/LLM_Related/tree/main/deepeval

### RAG本地知识库

RAGFlow:RAGFlow是一款基于深度文档理解构建的开源RAG（检索增强生成）引擎，它结合了检索（Retrieval）和生成（Generation）技术，并引入了工作流（Workflow）优化概念，旨在提升生成式AI系统的效率和性能。与传统RAG系统相比，RAGFlow通过多阶段处理、智能任务分配、自动化反馈机制和并行处理能力等特性，为企业和个人提供了一套精简而强大的知识处理解决方案。

https://github.com/Enermy/LLM_Related/tree/main/RagFlow%E6%9C%AC%E5%9C%B0%E7%9F%A5%E8%AF%86%E5%BA%93

下面是个人对RAGFLow的源码分析（有错误的地方希望指出谢谢）

https://github.com/Enermy/LLM_Related/tree/main/RAGFlow%E4%BB%A3%E7%A0%81%E8%AF%A6%E8%A7%A3

### LLamafactory的使用

下面两个案例介绍了如何使用LLamafactory，以及如何使用其进行微调；

LLamafactory微调Deepseek-R1模型（蒸馏版）

https://github.com/Enermy/LLM_Related/tree/main/%E4%BD%BF%E7%94%A8LLaMAFactory%E5%BE%AE%E8%B0%83DeepSeek-R1%E8%92%B8%E9%A6%8F%E6%A8%A1%E5%9E%8B

LLamafactory微调Qwen3：

https://github.com/Enermy/LLM_Related/tree/main/LLamafactory%E5%BE%AE%E8%B0%83Qwen3

### Unsloth

Unsloth提供了很多量化版模型，也可以使用其作为一个微调工具使用

https://github.com/Enermy/LLM_Related/tree/main/Unsloth%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B

### 蒸馏模型

介绍了模型蒸馏的方法，以及如何蒸馏Qwen2.5-1.5B-Instruct模型的详细步骤

https://github.com/Enermy/LLM_Related/tree/main/Distill_deepseek
