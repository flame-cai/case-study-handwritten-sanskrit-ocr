from transformers import HfArgumentParser, TensorFlowBenchmark, TensorFlowBenchmarkArguments
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers import EarlyStoppingCallback
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

load_checkpoint_iter = 1000
finetune_iter = 400
tokenizer_path = '/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/tokenizer_checkpoints'


train_df = pd.read_csv("/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/data/train_finetune.csv",delimiter=';')
eval_df = pd.read_csv("/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/data/val_finetune.csv",delimiter=';')
test_df = pd.read_csv("/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/data/test_finetune.csv",delimiter=';')


args_dict = {
    # "model_name_or_path": 'google/byt5-small',
    # "max_len": 4096,
    "output_dir": './model_checkpoints',
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "warmup_steps": 250,
    "logging_steps": 100,
    "evaluation_strategy": "steps",
    "eval_steps": 1000,
    "num_train_epochs": 4,
    "do_train": True,
    "do_eval": True,
    "fp16": False,
    # "use_cache": False,
    "max_steps": load_checkpoint_iter+finetune_iter,
    'save_steps':100,
    'save_strategy':'steps',
    # 'load_best_model_at_end': True,
    # 'metric_for_best_model':'eval_loss',
    # 'greater_is_better':False
}
parser = HfArgumentParser(
        (TrainingArguments))
training_args = parser.parse_dict(args_dict)
# set_seed(training_args.seed)
args = training_args[0]

# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(
    f"/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/model_checkpoints/checkpoint-{load_checkpoint_iter}",
).to(device)

# overwriting the default max_length of 20 
# tokenizer.model_max_length=4096
# model.config.max_length=4096



class GPReviewDataset(Dataset):
  def __init__(self, Text, Label):
    self.Text = Text
    self.Label = Label
    # self.tokenizer = tokenizer
    # self.max_len = max_len
  def __len__(self):
    return len(self.Text)
  def __getitem__(self, item):
    Text = str(self.Text[item])
    Label = self.Label[item]
    inputs = tokenizer(Text, padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(Label, padding="max_length", truncation=True, max_length=512)
    return {
      "input_ids":inputs.input_ids,
      "attention_mask" : inputs.attention_mask,
      "labels" : outputs.input_ids,
      "decoder_attention_mask" : outputs.attention_mask,
      # "labels" : lbz
    }

ds_train = GPReviewDataset(
  Text=train_df.input_text.to_numpy(),
  Label=train_df.target_text.to_numpy()
  # tokenizer=tokenizer,
  # max_len=max_len
)


ds_test = GPReviewDataset(
  Text=eval_df.input_text.to_numpy(),
  Label=eval_df.target_text.to_numpy()
  # tokenizer=tokenizer,
  # max_len=max_len
)


train_dataset = ds_train
valid_dataset = ds_test


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    # compute_metrics=compute_metrics

)

trainer.args.save_total_limit = 10
trainer.train(f'/home/ocr_proj/OCR/post_correction/pe-ocr-sanskrit/model_checkpoints/checkpoint-{load_checkpoint_iter}')
