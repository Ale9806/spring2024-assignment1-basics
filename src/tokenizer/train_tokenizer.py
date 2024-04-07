import argparse
import os 
import sys

if __name__ == "__main__":
    ## Only add src path when calling this function directly ##
    current_dir = os.path.abspath(os.path.dirname(__file__))
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    sys.path.insert(0, current_dir)
    print("added path")
    print(current_dir)

from src.tokenizer.tokenizer       import train_bpe_tokenizer
from src.tokenizer.tokenizer_tools import save_vocab_files



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

#on own computer 
# python src/tokenizer/train_tokenizer.py --input_path tests/fixtures/corpus.en  --vocab_size 300 --output_dir outputs  
# python src/tokenizer/train_tokenizer.py --input_path data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --output_dir outputs

# on sail culster
# python src/tokenizer/train_tokenizer.py --input_path tests/fixtures/corpus.en  --vocab_size 300 --output_dir /pasteur/data/c336/outputs
#python src/tokenizer/train_tokenizer.py --input_path /pasteur/data/c336/data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --output_dir /pasteur/data/c336/outputs
#python src/tokenizer/train_tokenizer.py --input_path /pasteur/data/c336/data/owt_train.txt --vocab_size 10000 --output_dir /pasteur/data/c336/outputs
