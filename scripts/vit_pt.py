import logging
import os
import sys
from argparse import ArgumentParser

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    ViTImageProcessor,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import yaml
from trl import ModelConfig, ScriptArguments, TrlParser
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

logger = logging.getLogger(__name__)
from vit_medical.model.vit_modeling_config import ViTConfig
from vit_medical.model.modeling_vit import ViTModel, ViTForImageClassification

logger = logging.getLogger(__name__)
from transformers import AutoImageProcessor

cache_dir = '/root/autodl-tmp/vit_medical_cache'
os.makedirs(cache_dir, exist_ok=True)

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True, cache_dir=cache_dir)


class ImageCollator:
    """处理图像的数据整理器"""
    def __init__(self, processor):
        self.processor = processor
        # 初始化转换器
        self.size = (480, 480)
        self.normalize = Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        self.transforms = Compose([Resize(self.size), ToTensor(), self.normalize])
        
    def __call__(self, features):
        # 提取图像和标签
        batch = {}
        
        if "images" in features[0]:
            # 处理原始图像
            images = [feature["images"] for feature in features]
            batch["pixel_values"] = torch.stack([
                self.transforms(img.convert("RGB")) for img in images
            ])
            
            if "labels" in features[0]:
                batch["labels"] = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        elif "pixel_values" in features[0]:
            # 如果已经预处理过的特征
            # 修复这里，确保每个feature["pixel_values"]是Tensor而不是list
            batch["pixel_values"] = torch.stack([
                feature["pixel_values"] if isinstance(feature["pixel_values"], torch.Tensor) 
                else torch.tensor(feature["pixel_values"]) 
                for feature in features
            ])
            
            if "labels" in features[0]:
                batch["labels"] = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        
        return batch
    
    def transform_dataset(self, examples):
        """用于dataset.map的转换函数"""
        # 确保返回的是Tensor类型而不是list
        examples["pixel_values"] = [
            self.transforms(img.convert("RGB")) for img in examples["images"]
        ]
        del examples["images"]
        return examples

def main(script_args, training_args, model_args, model_config):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    dataset = load_from_disk(script_args.dataset_name)
    dataset['test'] = dataset['test'].select(range(1000))
    
    ######################
    # Load image processor
    ######################
    image_processor = ViTImageProcessor(
        do_resize=(model_config["image_size"], model_config["image_size"]),
        do_center_crop=model_config["image_size"],
        do_normalize=True,
    )

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    

    ################################
    # Initialize model
    ################################
    logger.info("*** Initializing model ***")
    config = ViTConfig(**model_config)
    model = ViTForImageClassification(
        config=config,
    )
    model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model structure: {model}")
    logger.info(f"Model parameters: {model_num_params}")
    # 

    # 创建 ImageCollator 并用它预处理数据集
    data_collator = ImageCollator(processor=image_processor)
    
    # 使用 data_collator 对数据集进行预处理
    logger.info("*** Preprocessing dataset with ImageCollator ***")
    dataset_transformed = {}
    for split in dataset:
        dataset_transformed[split] = dataset[split].map(
            data_collator.transform_dataset, 
            batched=True,
            batch_size=2,
            desc=f"Processing {split} split",
            num_proc = 1
        )
    
    ################################
    # Initialize the PT trainer
    ################################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_transformed[script_args.dataset_train_split],
        eval_dataset=dataset_transformed[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None, 
        processing_class=image_processor,
        data_collator=data_collator,  # 使用 ImageCollator
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Training loop ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint='/root/vit-medical/data/vit-medical/checkpoint-500')
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)    
    logger.info(f"Model saved to {training_args.output_dir}")
    
   
    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        eval_result = trainer.evaluate()
        metrics = eval_result.metrics
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Evaluation complete ***")  

    ################################
    # Register the model and save
    ################################
    AutoConfig.register("vit-medical", ViTConfig)
    AutoModel.register(ViTConfig, ViTModel)
    AutoModelForImageClassification.register(ViTConfig, ViTForImageClassification)
    ViTConfig.register_for_auto_class("vit-medical")
    ViTModel.register_for_auto_class("AutoModel", "vit-medical")
    ViTForImageClassification.register_for_auto_class("AutoModelForImageClassification", "vit-medical")
    # image_processor = AutoImageProcessor.from_pretrained(training_args.output_dir)
    # image_processor.save_pretrained(training_args.output_dir)
    model = AutoModelForImageClassification.from_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub()
        logger.info(f"Model pushed to the hub")

    logger.info("*** Training complete ***")

if __name__ == "__main__":
    model_config_parser = ArgumentParser()
    model_config_parser.add_argument(
        "--config", type=str, default="./configs/training_config.yaml", help="path to yaml config file of PT"
    )
    parser = TrlParser((ScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()   


    model_config = yaml.load(
        open(model_config_parser.parse_args().config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )["model_config"]
    main(script_args, training_args, model_args, model_config)