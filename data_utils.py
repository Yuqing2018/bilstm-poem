import collections
import numpy as np


def build_word_dict1(filename):
    poems = []
    with open(filename, "r") as f:

        for line in f:  # every line is a poem
            title, author, poem = line.strip().split("::")  # get title and poem
            poem = poem.replace(' ', '')
            if len(poem) < 10 or len(poem) > 512:  # filter poem
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            poem = '[' + poem + ']'  # add start and end signs
            poems.append(poem)
            # print(title, author, poem)

        # counting words
        wordFreq = collections.Counter()
        for poem in poems:
            wordFreq.update(poem)
        wordFreq[" "] = -1
        wordPairs = sorted(wordFreq.items(), key=lambda x: -x[1])
        words, freq = zip(*wordPairs)
        wordNum = len(words)
        word_dict = dict(zip(words, range(wordNum)))  # word to ID
        return word_dict


def build_dataset1(filename, word_dict):
    data=[]
    with open(filename, "r") as f:
        for line in f:  # every line is a poem
            title, author, poem = line.strip().split("::")  # get title and poem
            poem = poem.replace(' ', '')
            if len(poem) < 10 or len(poem) > 512:  # filter poem
                continue
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            poem = '[' + poem + ']'  # add start and end signs
            data.append(poem)

    max_document_len = max([len(s) for s in data])
    data = list(map(lambda s: [word_dict.get(w, word_dict[" "]) for w in s], data))
    data = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict[" "]], data))

    return data

def build_word_dict(filename):
    with open(filename, "r") as f:
        words = f.read().replace("\n", "").split()

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<bos>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)

    return word_dict


def build_dataset(filename, word_dict):
    with open(filename, "r") as f:
        lines = f.readlines()
        data = list(map(lambda s: s.strip().split(), lines))

    max_document_len = max([len(s) for s in data]) + 2
    data = list(map(lambda s: ["<bos>"] + s + ["<eos>"], data))
    data = list(map(lambda s: [word_dict.get(w, word_dict["<unk>"]) for w in s], data))
    data = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], data))

    return data
def batch_iter(inputs, batch_size, num_epochs):
    inputs = np.array(inputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index]
