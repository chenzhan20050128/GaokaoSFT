- coding: utf-8 -*-
"""监督微调（Supervised Fine-Tuning, SFT）的训练脚本。

该模块的核心是 `run_sft_training` 函数，它 orchestrates（编排）了整个
SFT 流程。它利用了 Hugging Face `trl` 库提供的 `SFTTrainer`，这是一个
专门为在类 GPT 模型上进行指令微调而设计的强大工具。

整个流程与项目中的其他模块紧密协作：
- 从 `config` 加载配置。
- 从 `data` 构建数据集。
- 从 `modeling` 加载和准备模型。
- 从 `utils` 设置随机种子。
"""

# from __future__ import annotations: 同样是为了支持延迟解析类型注解。
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# `SFTConfig` 和 `SFTTrainer` 是 `trl` 库中用于 SFT 的核心类。
from trl import SFTConfig, SFTTrainer

from ..config import ProjectConfig
from ..data import build_sft_datasets
from ..modeling import (
    ensure_precision,
    load_base_model,
    merge_and_save,
    prepare_lora_model,
)
from ..utils import set_global_seed, dump_dataclass


def run_sft_training(
    project: ProjectConfig,
    *, # 强制后续参数为关键字参数
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    """执行完整的监督微调流程，并返回训练过程中的指标。

    执行步骤:
    1.  **初始化**: 确保输出目录存在，保存项目配置，并设置全局随机种子以保证可复现性。
    2.  **模型加载**: 使用 `load_base_model` 加载（可能量化的）基础模型和分词器。
    3.  **LoRA 准备**: 使用 `prepare_lora_model` 在基础模型上应用 LoRA 配置，
        得到一个 `PeftModel`，其中只有 LoRA 相关的权重是可训练的。
    4.  **数据准备**: 调用 `build_sft_datasets` 来加载、处理并分割训练和评估数据集。
    5.  **配置训练参数**: 将项目内部的 `TrainingConfig` 中的配置转换为 `SFTConfig`
        （`trl` 库的训练参数类）所期望的格式。这里还处理了不同 `trl` 版本
        的参数名兼容性问题，展示了良好的向后兼容设计。
    6.  **初始化训练器**: 创建 `SFTTrainer` 实例，将模型、分词器、数据集和
        训练参数传入。
    7.  **执行训练**: 调用 `trainer.train()` 方法启动训练过程。此方法会处理所有
        底层的训练循环、梯度更新、日志记录和模型保存。
    8.  **保存结果**:
        - 保存最终的 LoRA 适配器权重。
        - 如果配置了 `save_merged_model`，则调用 `merge_and_save` 将 LoRA
          权重合并到基础模型并保存为完整模型，便于直接部署。
        - 保存训练器的状态和最后一个检查点，以便未来可以恢复训练。
    9.  **返回指标**: 返回一个包含训练损失、步数等信息的字典。
    """
    # 1. 初始化
    project.ensure_directories()
    training_cfg = project.training
    set_global_seed(training_cfg.random_seed)

    # 将最终使用的配置保存为 JSON 文件，便于追溯和复现实验。
    dump_dataclass(project, training_cfg.output_dir / "project_config.json")

    # 2. 模型加载
    model, tokenizer = load_base_model(training_cfg)

    # 3. LoRA 准备
    peft_model = prepare_lora_model(model, training_cfg)

    # 4. 数据准备
    train_dataset, eval_dataset = build_sft_datasets(training_cfg, tokenizer)

    # 检查硬件精度支持
    _, bf16_supported = ensure_precision()

    # 5. 配置训练参数
    # 使用字典来构建参数，便于动态修改和传递。
    args_dict: dict[str, Any] = dict(
        per_device_train_batch_size=training_cfg.micro_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        # **梯度累积（Gradient Accumulation）详解**:
        # - **标准训练**: 每个批次（batch）的数据通过一次前向和反向传播，计算出梯度，然后立即用这个梯度更新模型的参数。
        # - **梯度累积训练**: 连续处理 N 个小批次（micro-batch），每次只计算梯度但不更新参数，而是将这 N 个小批次的梯度在内存中累加起来。
        #   当累积了 N 次之后，用这个累加的总梯度对模型参数进行一次更新，然后清空累加的梯度，开始下一个循环。
        # - **核心原理**: N 个小批次梯度平均值的期望，在数学上等同于一个大小为 N * micro-batch_size 的大批次梯度的期望。
        #   这意味着，梯度累积可以在不增加显存占用的情况下，模拟出使用大批次进行训练的效果。
        # - **主要作用**: 解决显存瓶颈。当GPU显存无法容纳理想的大批次数据时，可以通过增大梯度累积步数，在保持有效批次大小
        #   (effective_batch_size = micro_batch_size * gradient_accumulation_steps) 不变的情况下，显著降低每个小批次对显存的占用。
        #   这使得在消费级显卡上训练大模型成为可能。

        warmup_steps=training_cfg.warmup_steps,
        max_steps=training_cfg.max_steps,
        num_train_epochs=training_cfg.num_train_epochs,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        # **AdamW 优化器与权重衰减 (Weight Decay)**:
        # - AdamW（Adam with Weight Decay）是 Adam 优化器的一个改进版本。
        # - 在传统的 L2 正则化中，权重衰减项是加在损失函数上的，这会影响梯度的计算。
        # - AdamW 将权重衰减项从损失函数中解耦出来，直接在权重更新步骤中从权重中减去一小部分（`weight * weight_decay`）。
        # - 这种做法被证明在 Adam 这类自适应学习率的优化器中效果更好，能带来更稳定的训练和更好的泛化性能。

        fp16=not bf16_supported, # 如果不支持 bf16，则使用 fp16
        bf16=bf16_supported,
        logging_steps=training_cfg.logging_steps,
        # 根据是否使用4位量化选择不同的优化器。`paged_adamw_8bit` 是 unsloth/bitsandbytes 提供的优化版8位 Adam，能进一步节省显存。
        optim="paged_adamw_8bit" if training_cfg.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine", # 使用余弦学习率调度器，它会使学习率随训练步数从初始值平滑下降到0，通常能获得更好的收敛效果。
        seed=training_cfg.random_seed,
        output_dir=str(training_cfg.checkpoints_dir), # 训练检查点保存目录
        eval_strategy="steps", # 按步数进行评估
        eval_steps=training_cfg.eval_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        dataset_num_proc=training_cfg.dataset_num_proc,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        # **梯度检查点（Gradient Checkpointing）详解**:
        # - **核心思想**: 用计算换显存，是一种典型的时间-空间权衡（Time-Space Tradeoff）。
        # - **标准训练**: 在前向传播过程中，所有层的中间激活值（Activation）都会被保存下来，因为反向传播计算梯度时需要它们。这消耗了大量的显存。
        # - **梯度检查点**: 它并非保存所有中间激活值，而是有选择地只保存其中一部分（称为“检查点”）。在反向传播过程中，当需要那些已被丢弃的中间激活值时，
        #   系统会利用最近的一个检查点，重新执行一部分前向计算来临时生成它们。这显著降低了显存占用，但代价是增加了额外的计算量，会导致训练时间变长。
    )

    # 兼容性处理：较新版本的 trl 将 `eval_strategy` 重命名为 `evaluation_strategy`。
    # 通过检查 `SFTConfig` 的 `__init__` 方法的参数名来动态适应，增强了代码的健壮性。
    if "evaluation_strategy" in SFTConfig.__init__.__code__.co_varnames:
        args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")

    # 使用 `**args_dict` 将字典解包为关键字参数来初始化 SFTConfig。
    training_args = SFTConfig(**args_dict)

    # 6. 初始化训练器
    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text", # 告诉训练器数据集中哪个字段包含要训练的文本
        args=training_args,
        max_seq_length=training_cfg.max_seq_length,
    )

    # 7. 执行训练
    # `resume_from_checkpoint` 参数允许从上次中断的地方继续训练。
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = {**train_result.metrics}

    # 8. 保存结果
    # 保存 LoRA 适配器权重
    peft_model.save_pretrained(str(training_cfg.finetuned_model_dir))
    tokenizer.save_pretrained(str(training_cfg.finetuned_model_dir))

    # 合并并保存完整模型
    merge_and_save(peft_model, tokenizer, training_cfg)

    # 保存训练器状态，以便未来可以恢复训练
    trainer.save_state()
    # `save_model` 会保存最后一个检查点
    trainer.save_model(str(Path(training_cfg.checkpoints_dir) / "last"))

    # 9. 返回指标
    return metrics

