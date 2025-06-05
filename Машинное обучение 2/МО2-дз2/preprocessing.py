from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """


    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read().replace('&', '&amp;')

    root = ET.fromstring(content)

    sentence_pairs = []
    alignments = []
    
    for sentence_elem in root.findall('s'):
        source_elem = sentence_elem.find('english')
        target_elem = sentence_elem.find('czech')
        source_tokens = source_elem.text.strip().split() if source_elem.text is not None else []
        target_tokens = target_elem.text.strip().split() if target_elem.text is not None else []
        sentence_pair = SentencePair(source=source_tokens, target=target_tokens)
        sentence_pairs.append(sentence_pair)

        sure = [(int(pair.split('-')[0]), int(pair.split('-')[1])) for pair in sentence_elem.find('sure').text.strip().split()] if sentence_elem.find('sure').text else []
        possible = [(int(pair.split('-')[0]), int(pair.split('-')[1])) for pair in sentence_elem.find('possible').text.strip().split()] if sentence_elem.find('possible').text else []
        alignments.append(LabeledAlignment(sure=sure, possible=possible))

    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_tokens_freq = Counter()
    target_tokens_freq = Counter()

    [source_tokens_freq.update(pair.source) for pair in sentence_pairs]
    [target_tokens_freq.update(pair.target) for pair in sentence_pairs]

    if freq_cutoff is not None:
        source_vocab = [token for token, freq in source_tokens_freq.most_common(freq_cutoff)]
        target_vocab = [token for token, freq in target_tokens_freq.most_common(freq_cutoff)]
    else:
        source_vocab = list(source_tokens_freq.keys())
        target_vocab = list(target_tokens_freq.keys())

    source_index = {token: idx for idx, token in enumerate(source_vocab)}
    target_index = {token: idx for idx, token in enumerate(target_vocab)}

    return source_index, target_index


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    to_add = True
    
    for sentence_pair in sentence_pairs:
        to_add = True
        source_tokens = []
        target_tokens = []
        source_sentence = sentence_pair.source
        target_sentence = sentence_pair.target
        for token in source_sentence:
            index = source_dict.get(token, -1)
            if index == -1:
                to_add = False
                break
            source_tokens.append(index)
        for token in target_sentence:
            index = target_dict.get(token, -1)
            if index == -1:
                to_add = False
                break
            target_tokens.append(index)    
        if (to_add):
            tokenized_sentence_pairs.append(TokenizedSentencePair(np.array(source_tokens), np.array(target_tokens)))
    return tokenized_sentence_pairs
