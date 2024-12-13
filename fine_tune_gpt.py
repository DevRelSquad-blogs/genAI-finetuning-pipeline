from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_gpt(model_name="gpt2", dataset_path="path/to/dataset"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    dataset = load_dataset('text', data_files=dataset_path)['train']
    tokenized_data = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_gpt")

if __name__ == "__main__":
    fine_tune_gpt()
