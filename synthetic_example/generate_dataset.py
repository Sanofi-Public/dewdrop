from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import numpy as np
import torch
import transformers
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers import BertModel, BertConfig

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

def main(args): 
    # Sequence parameter
    n_seqs = args.n_seqs
    seq_length = args.seq_length
    target_dimension = args.target_dimension
    dataset_dir = args.dataset_dir

    # GPT2's output embedding as input sequence
    gen_config = GPT2Config(
        vocab_size=32,              # Vocabulary size as per your custom token set
        n_positions=1032,           # Increased to handle 1000 tokens after the 32-token prompt
        n_embd=16,                  # Reduced embedding dimension based on your original setup
        n_layer=2,                  # Keeping it shallow for reduced memory usage
        n_head=4,                   # Number of attention heads, manageable for a small model
        n_inner=None,               # Defaults to 4*n_embd if None; for simplicity, leaving this as None
        activation_function="gelu_new",  # Default GELU activation variant for GPT-2
        resid_pdrop=0.1,            # Residual dropout for regularization
        embd_pdrop=0.1,             # Dropout for embedding layer
        attn_pdrop=0.1,             # Dropout for attention layers
        initializer_range=0.02,     # Standard initialization range for weights
        layer_norm_epsilon=1e-5,    # Epsilon for layer normalization
        output_hidden_states=False, # Only output the final hidden state by default
        output_attentions=False,    # Do not output attention scores
        use_cache=True,             # Enable caching for faster generation
        bos_token_id=0,             # Beginning of sequence token ID
        eos_token_id=1,             # End of sequence token ID
    )

    gen_model = GPT2LMHeadModel(gen_config)

    # prompt with random input
    prompts = torch.randint(32, (n_seqs, 32))

    # create different sections of data with different settings
    num_sections = args.num_sections
    section_size = n_seqs // num_sections
    sections = [prompts[i*section_size:(i+1)*section_size,:] for i in range(num_sections)]
    diversity_seqs = []
    for i, section in enumerate(sections):
        temperature = 2.0 - (i * 1.2 / num_sections)  
        top_k = int(30 + i * 70 / num_sections)       
        top_p = 0.8 + (i * 0.15 / num_sections)       

        # diversity generation configuration
        one_sec_seq = gen_model.generate(
            inputs=section,  # Assume the first half is for high diversity
            max_new_tokens=seq_length,
            do_sample=True,
            early_stopping=False, 
            temperature=temperature,  # High temperature for high randomness
            top_k=top_k,         # Low top_k to allow more token diversity
            top_p=top_p,
            eos_token_id=None        
        )[:, 32:].detach()
        diversity_seqs.append(one_sec_seq)

    # Combine results for further use
    in_seqs = torch.cat(diversity_seqs, dim=0)

    # BERT's output embedding as target series
    true_config = BertConfig( 
        vocab_size=32, # max 32
        hidden_size=target_dimension, # if out_size isn't defined, this is the output size.
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
    )

    true_process = BertEmbeddor(true_config)

    out_seqs = true_process(in_seqs).detach()

    # Store the input output datasets 
    os.makedirs(dataset_dir, exist_ok=True)
    np.savetxt(os.path.join(dataset_dir, "input_sequences.txt"), in_seqs.numpy().reshape(in_seqs.shape[0], -1), delimiter=" ", fmt="%d")
    np.savetxt(os.path.join(dataset_dir, "target_sequences.txt"), out_seqs.numpy().reshape(in_seqs.shape[0], -1), delimiter=" ", fmt="%.8e")

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter) 
    parser.add_argument("--dataset_dir", default="./synthetic_data/", help="Location to train and test datasets")
    parser.add_argument("--n_seqs", type=int, default=1100, help="Number of sequences")
    parser.add_argument("--seq_length", type=int, default=100, help="Length of each sequence")
    parser.add_argument("--target_dimension", type=int, default=100, help="Length of each sequence")
    parser.add_argument("--num_sections", type=int, default=10, help="Number of bins for data generation")
    args = parser.parse_args()
    main(args)