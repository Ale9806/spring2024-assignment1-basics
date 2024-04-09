
from collections import Counter
from typing      import Optional, Iterable, Iterator
import pickle

from src.tokenizer.tokenizer_tools import read_corpus,pre_tokenize_corpus,initialize_vocabulary,merge_pair,compute_pair_freqs

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
    del corpus_list, corpus_str   # Delete uncessary variables to save memory
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


class Tokenizer:
    def __init__(self, vocab:  dict[int, bytes], 
                       merges: list[tuple[bytes, bytes]], 
                       special_tokens: Optional[list[str]] = None):
        
        self.special_tokens  = special_tokens
        self.vocab           = vocab
        self.merges          = merges
        self.add_special_tokens()
        

        
        self.inv_vocab       = {v: k for k, v in self.vocab.items()}

    def add_special_tokens(self):
        if self.special_tokens:
            for i,special_token in enumerate(self.special_tokens):
                self.vocab[len(self.vocab) + i] = special_token.encode('utf-8')

    @staticmethod
    def load_pickle(file):
        with open(file, 'rb') as f:
            file_content = pickle.load(f)
        return file_content
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        """ Instancaite tokenizer from files """
        vocab  = cls.load_pickle(vocab_filepath)
        merges = cls.load_pickle(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        There are words that might be repeated in our input corpus, therefore we will go from:
        list -> dict -> merge -> list ->ids
        """
        
        pre_tokenize_text:list[str]     = pre_tokenize_corpus(text)
        splits:dict[str,tuple[bytes]]   = {word: list(bytes((i,)) for i in word.encode('utf-8')) for word in pre_tokenize_text}
        splits:dict[str,tuple[bytes]]   = self.merge_encodings(splits,self.merges)   
        pre_tokenize_text_m:list[bytes] = self.parse_merges_to_text(pre_tokenize_text,splits)
        ids:list[int]                   = self.bytes_to_ids(pre_tokenize_text_m)

        return ids
    
    @staticmethod
    def merge_encodings(splits,merges):
        for word,values in splits.items():
            word_length  = len(values)
            i = 0 
            while i +1 < word_length:
                if (values[i],values[i+1]) in merges:
                    j = i
                    values[i+1] = values[i] + values[i+1]
                    values.pop(i)
                    word_length  = len(values)
                    i= j
                else:
                    i+=1
               
             
            splits[word] = values

        return splits
    
    @staticmethod
    def parse_merges_to_text(encode_text,splits) -> list[bytes]:

        encode_text_merged:list[bytes] = []
        for word in encode_text:
            encode_text_merged.extend(splits[word])
        return encode_text_merged
    
    def bytes_to_ids(self,byte_list:list[bytes]) -> list[int]:
        ids = self.dic_lookup(byte_list,self.inv_vocab)
        return ids
    
    @staticmethod
    def dic_lookup(input_list:list,vocab) ->list :
        n   =  len(input_list)
        ids = [None]*n

        for i in range(0,n):
            ids[i] = vocab[input_list[i]]

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Lazily encode iterable of strings into token IDs
        for text in iterable:
            yield self.encode(text)
    
    def decode(self,ids: list[int],
                    return_string:bool=True) -> str:
        # Decode token IDs into text
        # Implement decoding logic
        byte_list = self.dic_lookup(ids,self.vocab)

        if return_string:
            concatenated_bytes = b''.join(byte_list)
            text               = concatenated_bytes.decode("utf-8", errors="replace")
            return text

        else:
            return byte_list