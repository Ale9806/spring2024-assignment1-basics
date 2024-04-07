import argparse
import os 

from tokenizer       import train_bpe_tokenizer
from tokenizer_tools import save_vocab_files

def train_and_save_tokenizer(input_path, vocab_size, special_tokens,output_dir):

    vocab, merges = train_bpe_tokenizer(input_path     = input_path, 
                                        vocab_size     = vocab_size, 
                                        special_tokens = special_tokens)

    save_vocab_files(input_path = input_path,
                     output_dir = output_dir,
                     vocab      = vocab,
                     merges     = merges)

def main():
    parser = argparse.ArgumentParser(description='Train and save BPE tokenizer')
    parser.add_argument('--input_path',    type=str,   help='Path to input data')
    parser.add_argument('--vocab_size',    type=int,   help='Size of vocabulary')
    parser.add_argument('--special_tokens', nargs='+', default=["<|endoftext|>"], help='List of special tokens')
    parser.add_argument('--output_dir',    type=str,   default="outputs", help='Path to input data')

    args = parser.parse_args()
    train_and_save_tokenizer(args.input_path, args.vocab_size, args.special_tokens,args.output_dir)

if __name__ == "__main__":
    main()


# python src/train_tokenizer.py --input_path tests/fixtures/corpus.en  --vocab_size 300 --output_dir outputs  
# python src/train_tokenizer.py --input_path data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --output_dir outputs