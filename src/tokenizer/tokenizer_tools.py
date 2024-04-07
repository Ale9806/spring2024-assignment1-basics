from typing      import Optional
from collections import defaultdict
import regex as re

import os
import json
import pickle

from src.tokenizer.constants import PAT


def read_corpus(input_path,encoding:Optional[str]='utf-8') -> str:
    """
    Read text data from a file.

    Parameters:
        input_path (str or os.PathLike): The path to the input file.
        encoding (str, optional): The encoding of the input file. Defaults to 'utf-8'.

    Returns:
        str: The content of the file as a string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the specified file cannot be read due to permission issues.
        UnicodeDecodeError: If there's an encoding error while reading the file.
    """
    with open(input_path, 'r', encoding=encoding) as file:
        return file.read()
    
def pre_tokenize_corpus(corpus:str) -> list[str]:
    """
    Pre-tokenize a corpus string using a regular expression pattern.

    Parameters:
        corpus (str): The input corpus text to be tokenized.

    Returns:
        List[str]: A list of tokens extracted from the corpus based on the pattern.

    Notes:
        This function uses the `re.findall()` method to extract tokens from the corpus string
        based on the provided regular expression pattern (`PAT`).

    Example:
        >>> text = "This is a sample sentence."
        >>> pre_tokenize_corpus(text)
        ['This', 'is', 'a', 'sample', 'sentence']
    """
    return re.findall(PAT,corpus)

def initialize_vocabulary(special_tokens:Optional[list[str]]= None) -> dict[int, bytes]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. Adapted from the GPT-2 code.
    """
    vocab:dict = {}
    ## Add special tokens if provided ##
    if special_tokens:
        for i,special_token in enumerate(special_tokens):
            vocab[i] = special_token.encode('utf-8')
    
    ## Initalize first 256 bytes ##
    n = len(vocab)
    for i in range(0,2**8): 
        vocab[n+i]  = bytes([i])

    return vocab 

def compute_pair_freqs(splits:dict[str, list[bytes]],word_freq):
    pair_freqs = defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a:bytes, b:bytes, splits,word_freq) -> dict[str, list[bytes]]:
    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def initialize_vocabulary_from_corpus(word_freq:dict,special_tokens:list):
    byte_alphabet:list[int]   = []
    char_alphabet:list[bytes] = []
    vocab:dict[int,bytes]     = {}

    for word in word_freq.keys():
        for char in word:

            byte = ord(char)
            char =  char.encode('utf-8')
            if byte not in byte_alphabet:
                byte_alphabet.append(byte)
                char_alphabet.append(char)
                vocab[byte] = char
            
    assert len(byte_alphabet) == len(char_alphabet)

    if special_tokens:
            for i,special_token in enumerate(special_tokens):
                value     = 0
                for byte in special_tokens[0].encode('utf-8'):
                    value += byte
                vocab[value] = special_token
    return vocab

def save_vocab_files(input_path:str , 
                     output_dir:str, 
                     vocab:dir,
                     merges:list[tuple[bytes]]) -> None:
    """
    Save vocabulary files.

    Parameters
    ----------
    input_path : str
        Path to the input file.
    output_dir : str
        Directory where output files will be saved.
    vocab : dict
        Dictionary containing vocabulary.
    merges : list of tuples
        List of tuples containing merge information.

    Returns
    -------
    None
        This function does not return anything; it saves files to disk.
    """
    input_file_name  = os.path.basename(input_path).split(".")[0]
    output_dir       = os.path.join(output_dir, input_file_name)
    os.makedirs(output_dir, exist_ok=True)

    vocab_chr = {key: value.decode("utf-8", errors="replace") for key, value in vocab.items()}
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab_chr, f, indent=4)

    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    with open(os.path.join(output_dir, 'merges.txt'), 'w') as f:
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")

    with open(os.path.join(output_dir, 'merges.pkl'), 'wb') as f:
        pickle.dump(merges, f)