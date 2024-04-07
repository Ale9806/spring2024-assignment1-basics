
from src.tokenizer.tokenizer_tools import read_corpus,pre_tokenize_corpus,initialize_vocabulary,merge_pair,compute_pair_freqs


from collections         import Counter

def train_bpe_tokenizer(input_path: str, 
                        vocab_size: int, 
                        special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a (byte-level) BPE tokenizer given a path to an input text file.

    Args:
    input_path (str): Path to a text file with BPE tokenizer training data.
    vocab_size (int): A non-negative integer that defines the maximum final vocabulary size.
    special_tokens (list[str]): A list of strings to add to the vocabulary.
    
    Returns:
    Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]: The resulting vocabulary and merges.
        vocab (Dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges (List[Tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that
    """
    ### Initalize Vocab ###
    corpus_str:str                = read_corpus(input_path)
    corpus_list:list[str]         = pre_tokenize_corpus(corpus_str)
    word_freq:dict[str,int]       = Counter(corpus_list)
    vocab: dict[int,bytes]        =  initialize_vocabulary(special_tokens)
    splits:dict[str,tuple[bytes]] = {word: list(bytes((i,)) for i in word.encode('utf-8')) for word in word_freq.keys()}
    merges     = []

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits,word_freq)
        best_pair  = ""
        max_freq   = None
        
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq

            elif max_freq == freq:
                best_pair = max(pair,best_pair)

        if best_pair == '':
            break
        
        else:
            splits = merge_pair(*best_pair, splits,word_freq)
            merges.append(best_pair)
            vocab[len(vocab)] = best_pair[0] + best_pair[1]
    
    return vocab, merges