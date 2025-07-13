import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
# from Dataset.multi_dataset import multi_dataset
from Dataset.multi_dataset_try import multi_dataset
# from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from Model.RadFM.multimodality_model_copy import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
from datasets import load_metric
# from Dataset.multi_dataset_test_for_close import multi_dataset_close
from Dataset.multi_dataset_test_for_close_try import multi_dataset_close
import numpy as np
import torch
import pdb

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import Conv1D


def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    ACCs = eval_preds.predictions
    # print(ACCs)
    return {"accuracy": np.mean(ACCs,axis=-1)}

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(default="./RadFM/Quick_demo/Language_files")
    tokenizer_path: str = field(default='./RadFM/Quick_demo/Language_files', metadata={"help": "Path to the tokenizer data."})   
    
    
    

@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default = False)
    batch_size_2D: int = field(default = 2)
    batch_size_3D: int = field(default = 1)
    output_dir: Optional[str] = field(default="./RadFM/mdl_fine/nokbit_temp/fine-tune-pth/")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # should_save: bool = field(default = True)
    num_train_epochs: int = field(default = 1)

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        vision_xs, lang_xs, attention_masks, labels,loss_reweight,key_words_query = tuple([instance[key] for instance in instances] for key in ('vision_x','lang_x', 'attention_mask', 'labels', 'loss_reweight','key_words_query'))
        
        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs],dim  = 0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks],dim  = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels],dim  = 0)
        loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight],dim  = 0)
        #print(lang_xs.shape,attention_masks.shape,labels.shape)
        
        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0
           
        D_list = list(range(4,65,4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] >6:
                D_list = list(range(4,33,4))
        
        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D)< abs(target_D - MAX_D):
                target_D = temp_D
                
        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256
            
        vision_xs = [torch.nn.functional.interpolate(s, size = (target_H,target_W,target_D)) for s in vision_xs]
        
        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        print(vision_xs.shape,vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            attention_mask=attention_masks,
            labels = labels,
            loss_reweight = loss_reweight,
            key_words_query = key_words_query
        )


def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        # if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D)):
        if isinstance(module, (torch.nn.Linear)):
        # if isinstance(module, (torch.nn.Linear)) and ('embedding_layer' not in name):
            layer_names.append(name)
    # print(layer_names)
    return layer_names



def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.data_sampler = My_DistributedBatchSampler
    
    print("Setup Data")
    Train_dataset = multi_dataset(text_tokenizer = model_args.tokenizer_path)
    Eval_dataset = multi_dataset_close(text_tokenizer = model_args.tokenizer_path)
    print("Setup Model")

    
    model = MultiLLaMAForCausalLM(
        lang_model_path='./RadFM/Quick_demo/Language_files', ### Build up model based on LLaMa-13B config
    )
    ckpt = torch.load('./RadFM/mdl_fine/pytorch_model.bin', map_location ='cpu') 
    model.load_state_dict(ckpt)
    model = model.to('cuda')
    print("Finish loading model")
    
    lora_config = LoraConfig(
                    r=16,
                    lora_alpha=8,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules=get_specific_layer_names(model),
                )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      args = training_args,
                      data_collator = DataCollator(),
                      compute_metrics= compute_metrics
                      )

    
    trainer.train()
    model.save_pretrained("./RadFM/mdl_fine/nokbit_lora_adapter/lora_adapter", save_adapter=True, save_config=True)
    
if __name__ == "__main__":
    main()
