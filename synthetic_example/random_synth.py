from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from pathlib import Path
import numpy as np
import csv
import random 

import torch
import transformers
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, BertModel, BertConfig    
from transformers import TrainerCallback, Trainer, TrainingArguments, EarlyStoppingCallback

# =========== AL Specific Import ===========
from alien.selection import RandomSelector
# ==========================================

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# For trainer to log test loss 
class TestLossCallback(TrainerCallback):
    def __init__(self, logging_dir, file_name="test_loss.csv"):
        # Set the output file path to the specified logging directory and file name
        self.output_file = os.path.join(logging_dir, file_name)
        
        # Write header to the CSV file
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "test_loss"])

    def on_evaluate(self, args, state, control, **kwargs):
        # Get validation loss from the logs
        test_loss = state.log_history[-1]["eval_loss"]
        epoch = state.epoch

        # Append validation loss to the file
        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, test_loss])

# The model
class BertEmbeddor(BertModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.out_size = getattr(config, 'out_size', 0) or config.hidden_size
        assert self.out_size <= config.hidden_size
        self.loss = torch.nn.MSELoss()
        
    def forward(self, input_ids, labels=None, **kwargs):
        output = super().forward(input_ids=input_ids, **kwargs).last_hidden_state[...,:self.out_size]
        if labels is not None:
            loss = self.loss(output, labels)
            return loss, output.detach()
        return output     

# The dataset
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels, attention_mask=None):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float)
        self.attention_mask = torch.as_tensor(attention_mask) if attention_mask is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.attention_mask is not None:
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx],   # embeddings as labels
            }
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],  # embeddings as labels
        }

def load_data(dataset_dir, n_seqs, seq_length, target_dimension): 
    inseqs_path = os.path.join(dataset_dir, "input_sequences.txt")
    outseqs_path = os.path.join(dataset_dir, "target_sequences.txt")
    in_seqs = np.loadtxt(inseqs_path, dtype=int).reshape(n_seqs, seq_length)
    out_seqs = np.loadtxt(outseqs_path, dtype=np.float32).reshape(n_seqs, seq_length, target_dimension)
    return in_seqs, out_seqs

def random_selection(unlabeled, select_bs=100): 
    # random selection 
    random_selector = RandomSelector(
        batch_size=select_bs,
        samples=unlabeled,
        random_seed=42
    )
    new_batch_indices = random_selector.select(return_indices=True)
    selected_data_indices = [unlabeled[ind] for ind in new_batch_indices]
    return selected_data_indices

def main(args): 
    dataset_dir = args.dataset_dir 
    selection_rounds = args.selection_rounds
    selection_batch_size = args.selection_batch_size
    shuffle = args.shuffle
    target_dimension = args.target_dimension
    test_size = args.test_size
    version = args.v

    n_seqs = args.n_seqs
    seq_length = args.seq_length
    target_dimension = args.target_dimension

    current_selection_round = 0 
    in_seqs, out_seqs = load_data(dataset_dir, n_seqs, seq_length, target_dimension)
    in_seqs, out_seqs = torch.as_tensor(in_seqs), torch.as_tensor(out_seqs)
    if shuffle: 
        indices = list(range(len(in_seqs)))
        indices = random.shuffle(indices)
        in_seqs = in_seqs[indices].squeeze()
        out_seqs = out_seqs[indices].squeeze()
    trainin, testin = in_seqs[test_size:], in_seqs[:test_size]
    trainout, testout = out_seqs[test_size:], out_seqs[:test_size]
    test_data = EmbeddingDataset(testin, labels=testout)
    
    # both are sample indices in the dataset 
    unlabeled = list(range(len(trainin)))
    labeled = []

    model_config = BertConfig(
        out_size=target_dimension,
        vocab_size=32,
        hidden_size=128,             # Increased hidden size for larger embedding representations
        num_hidden_layers=12,        # Increased number of layers for greater depth
        num_attention_heads=16,      # More attention heads to enhance capacity in attention layers
        intermediate_size=512,       # Larger intermediate size for deeper feed-forward transformations
        hidden_act="gelu",           # Activation function remains GELU
        hidden_dropout_prob=0.1,     # Regular dropout remains the same for regularization
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024, # Maximum sequence length remains the same
        layer_norm_eps=1e-5,          # Epsilon for layer normalization
    )

    model = BertEmbeddor(model_config)

    while current_selection_round < selection_rounds:
        print(f"""
            RUNING ROUND {current_selection_round+1} / {selection_rounds} 
        """)

        # ================= RANDOM Selection ================= 
        selection_indices = random_selection(unlabeled, selection_batch_size)
        selection_indices.sort(reverse=True)
        for ind in selection_indices: 
            labeled.append(ind) # add to labeled
            unlabeled.remove(ind) # remove from unlabeled
        # ===================================================
        temp_in, temp_out = trainin[labeled], trainout[labeled]
        train_data = EmbeddingDataset(input_ids=temp_in, labels=temp_out)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=100, # usually converge in less than 100 epochs
            per_device_train_batch_size=selection_batch_size,
            no_cuda=False,
            warmup_steps=10,
            weight_decay=0.01,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            save_strategy='epoch',
            logging_dir='./logs',
            evaluation_strategy="epoch",
            load_best_model_at_end = True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=1e-5),
                TestLossCallback(logging_dir=training_args.logging_dir, file_name=f"random_bs={selection_batch_size}_round={current_selection_round}_v{version}.csv")
            ], 
        )

        trainer.train()

        print(trainer.evaluate())

        current_selection_round += 1


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", default="./synthetic_data/", help="Location to train and test datasets")
    parser.add_argument("--selection_rounds", type=int, default=5, help="Number of selection rounds to run for retrospective experiment")
    parser.add_argument("--selection_batch_size", type=int, default=100, help="Size of the selection batch size")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle dataset before spliting")
    parser.add_argument("--test_size", type=int, default=100, help="Size of the test set")
    parser.add_argument("-v", type=int, default=1, help="Version number of the run")
    # Using the followings to match tensor shape and load them from txt
    parser.add_argument("--n_seqs", type=int, default=1100, help="Number of sequences")
    parser.add_argument("--seq_length", type=int, default=100, help="Length of each sequence")
    parser.add_argument("--target_dimension", type=int, default=100, help="Length of each sequence")
    args = parser.parse_args()
    main(args)