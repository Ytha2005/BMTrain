import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import bmtrain as bmt
from bmtrain import optim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--dataset", type=str, default="trl-lib/Capybara")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    return p.parse_args()


class SFTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        prompt_messages = messages[:-1]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

        full_enc = self.tokenizer(
            full_text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        prompt_len = self.tokenizer(
            prompt_text, max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )["input_ids"].shape[1]

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels


def main():
    args = parse_args()

    bmt.init_distributed(seed=42)

    bmt.print_rank(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = bmt.BMTrainModelWrapper(model)

    bmt.print_rank(f"Loading dataset {args.dataset} ...")
    raw_dataset = load_dataset(args.dataset, split="train")
    dataset = SFTDataset(raw_dataset, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    optimizer = optim.AdamOffloadOptimizer(
        model.parameters(), lr=args.lr, weight_decay=0.01,
    )
    lr_scheduler = bmt.lr_scheduler.Cosine(
        optimizer, start_lr=args.lr,
        warmup_iter=args.warmup_steps, end_iter=max(total_steps, 1),
    )
    optim_manager = optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    bmt.synchronize()

    bmt.print_rank(f"Training {args.epochs} epochs, {total_steps} optimizer steps")
    avg_loss = bmt.utils.AverageRecorder()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for step, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / args.grad_accum
            global_loss = bmt.distributed.all_reduce(loss, "avg").item()

            optim_manager.backward(loss)

            if (step + 1) % args.grad_accum == 0:
                optim_manager.step()
                optim_manager.zero_grad()
                global_step += 1

            avg_loss.record(global_loss)
            bmt.print_rank(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Step {step+1}/{len(dataloader)} | "
                f"Loss {global_loss:.4f} (avg {avg_loss.value:.4f}) | "
                f"LR {lr_scheduler.current_lr:.2e} | "
                f"Scale {optim_manager.loss_scale:.0f}"
            )

    os.makedirs(args.save_dir, exist_ok=True)
    bmt.save(model, os.path.join(args.save_dir, "sft_model.pt"))
    bmt.print_rank(f"Saved to {args.save_dir}/sft_model.pt")


if __name__ == "__main__":
    main()
