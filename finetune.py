import unsloth
import datasets
import pandas as pd
import transformers
import trl

#load dataset
data = pd.read_json("./alpaca_cleaned.json")

#finetuning standard dataset template
data_prompt = """{}

### Input:
{}

### Response:
{}"""

#max sequence length (5020 is good enough)
max_seq_length = 5020

#load model + tokenizer
model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

#create blank peft addon
model = unsloth.FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)

#mapping function to format entries
format_prompt = lambda examples: {
    'text': [data_prompt.format(system_instruction, input_, output) + tokenizer.eos_token for system_instruction, input_, output in zip(examples['instruction'], examples["input"], examples["output"])]
}

# load from dataframe to dataset & map into proper format
training_data = datasets.Dataset.from_pandas(data)
training_data = training_data.map(format_prompt, batched=True)

#setup trainer
trainer=trl.SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=transformers.TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=40,
        fp16=not unsloth.is_bfloat16_supported(),
        bf16=unsloth.is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

#train
trainer.train()

# save
model.save_pretrained("superagent/1B_finetuned_llama3.2")
tokenizer.save_pretrained("superagent/1B_finetuned_llama3.2")
