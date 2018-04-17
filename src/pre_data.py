import codecs
import random
import copy
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# data structure for SVO sequence
class Svo:
    def __init__(self, line):
        line = line.strip().split("\t")
        if len(line) == 2:
            instance = line[0]
            svo = instance.split("_")
            self.id = int(svo[0])
            self.s_id = int(svo[3])
            self.s_lemma = svo[4]
            self.s_surface_form = svo[5]
            self.v_id = int(svo[6])
            self.v_lemma = svo[7]
            self.v_surface_form = svo[8]
            self.o_id = int(svo[9])
            self.o_lemma = svo[10]
            self.o_surface_form = svo[11]
        else:
            self.s_lemma = line[0]
            self.s_surface_form = None
            self.v_lemma = line[1]
            self.v_surface_form = None
            self.o_lemma = line[2]
            self.o_surface_form = None


# data structure for Dep word
class DepWord:
    def __init__(self, line):
        tokens = line.strip().replace(u'\uFFFD', '?').split('\t')
        # u'\uFFFD' represent an unknown or unrepresentable character
        self.id = int(tokens[0]) - 1
        self.surface_form = tokens[1].split()[-1].lower()  # Only the last word.
        self.pos_tag = tokens[3]
        self.head = int(tokens[6]) - 1
        self.dep_rel = tokens[7]
        self.lemma = tokens[2].lower()
        if self.lemma in [u'-', u'_']:
            if self.pos_tag.startswith("V"):
                lemmatizer_pos = 'v'
            elif self.pos_tag.startswith("N"):
                lemmatizer_pos = 'n'
            else:
                lemmatizer_pos = 'a'
            self.lemma = lemmatizer.lemmatize(self.surface_form, lemmatizer_pos).lower()


# data structure for Dep sequence
class DepSentence:
    def __init__(self):
        self.words = []

    def add_word(self, line):
        word = DepWord(line)
        self.words.append(word)

    def get_dep_seq(self, w, sequence):
        if w not in sequence:
            sequence.append(w)
        for word in self.words:
            if word.head == w.id and word not in sequence:
                sequence.append(word)
            elif word.id == w.head and word not in sequence:
                sequence.append(word)
        return sequence


# load the train data from the svo and the dep file
def load_train_data(svo_file, dep_file, label):
    svo = []
    dep = []
    sen = []
    for line in codecs.open(svo_file, "r", "utf-8"):
        svo.append(Svo(line))
    sentence = DepSentence()
    seq_num = 0
    svo_idx = 0
    for line in codecs.open(dep_file, "r", "utf-8'"):
        if svo_idx == len(svo):
            break
        if len(line.strip()) == 0:
            while svo_idx < len(svo) and svo[svo_idx].id == seq_num:
                sen.append(sentence.words)
                sequence = []
                if svo[svo_idx].s_id != 999:
                    sequence = sentence.get_dep_seq(sentence.words[svo[svo_idx].s_id], sequence)
                sequence = sentence.get_dep_seq(sentence.words[svo[svo_idx].v_id], sequence)
                if svo[svo_idx].o_id != 999:
                    sequence = sentence.get_dep_seq(sentence.words[svo[svo_idx].o_id], sequence)
                dep.append(sorted(sequence, key=lambda x: x.id))
                svo_idx += 1
            sentence = DepSentence()
            seq_num += 1
        else:
            sentence.add_word(line)
    pairs = []
    for i in range(len(svo)):
        pairs.append((svo[i], dep[i], sen[i], label))
    return pairs


# load the test data from the svo and the dep file
def load_test_data(svo_file, dep_file, lemma2word, label):
    svo = []
    dep = []
    sen = []
    line_num = 0
    lemma2word_num = 0
    for line in codecs.open(svo_file, "r", "utf-8"):
        temp_svo = Svo(line)
        while lemma2word_num < len(lemma2word):
            l2w = lemma2word[lemma2word_num].split("_")
            if line_num == int(l2w[0]):
                lemma2word_num += 1
                if temp_svo.s_lemma == l2w[1]:
                    temp_svo.s_surface_form = l2w[2]
                if temp_svo.v_lemma == l2w[1]:
                    temp_svo.v_surface_form = l2w[2]
                if temp_svo.o_lemma == l2w[1]:
                    temp_svo.o_surface_form = l2w[2]
            else:
                break
        svo.append(temp_svo)
        line_num += 1
    sentence = DepSentence()
    line_num = 0
    for line in codecs.open(dep_file, "r", "utf-8'"):
        if len(line.strip()) == 0:
            temp_svo = svo[line_num]
            sen.append(sentence.words)
            sequence = []
            for w in sentence.words:
                if temp_svo.s_surface_form is None and temp_svo.s_lemma == w.lemma:
                    temp_svo.s_surface_form = w.surface_form

                if temp_svo.v_surface_form == w.surface_form:
                    sequence = sentence.get_dep_seq(sentence.words[w.id], sequence)
                elif temp_svo.s_lemma == w.lemma:
                    sequence = sentence.get_dep_seq(sentence.words[w.id], sequence)

                if temp_svo.o_surface_form is None and temp_svo.o_lemma == w.lemma:
                    temp_svo.o_surface_form = w.surface_form

                if temp_svo.o_surface_form == w.surface_form:
                    sequence = sentence.get_dep_seq(sentence.words[w.id], sequence)
                elif temp_svo.o_lemma == w.lemma:
                    sequence = sentence.get_dep_seq(sentence.words[w.id], sequence)

                if temp_svo.v_surface_form is None:
                    if temp_svo.v_lemma == w.lemma:
                        temp_svo.v_surface_form = w.surface_form
                        sequence = sentence.get_dep_seq(sentence.words[w.id], sequence)
                        sequence = sorted(sequence, key=lambda x: x.id)
                        dep.append(sequence)
                elif temp_svo.v_surface_form == w.surface_form:
                    sequence = sentence.get_dep_seq(sentence.words[w.id], sequence)
                    sequence = sorted(sequence, key=lambda x: x.id)
                    dep.append(sequence)
            sentence = DepSentence()
            line_num += 1
        else:
            sentence.add_word(line)
    pairs = []
    for i in range(len(svo)):
        pairs.append((svo[i], dep[i], sen[i], label))
    return pairs


# prepare the batch to train in every epoch
def pre_single_batch(batch_data_, batch_size, label):
    # prepare the batch from the dataset
    batch_data = copy.deepcopy(batch_data_)
    batches = []
    len_batch = len(batch_data)
    start_batch = 0
    while start_batch + batch_size < len_batch:
        batches.append(batch_data[start_batch: start_batch + batch_size])
        start_batch += batch_size
    batches.append(batch_data[start_batch:])

    # padding
    if label > 0:
        input_lengths = []
        input_batchs = []
        target_batechs = []
        for batch in batches:
            batch = sorted(batch, key=lambda tp: len(tp[label]), reverse=True)
            input_length = []
            input_batch = []
            target_batch = []
            for i in batch:
                input_length.append(len(i[label]))
                input_batch.append(i[label])
                target_batch.append(i[-1])
            input_lengths.append(input_length)
            input_batchs.append(input_batch)
            target_batechs.append(target_batch)
        return input_batchs, input_lengths, target_batechs
    else:
        input_batchs = []
        target_batechs = []
        for batch in batches:
            input_batch = []
            target_batch = []
            for i in batch:
                input_batch.append(i[0])
                target_batch.append(i[-1])
            input_batchs.append(input_batch)
            target_batechs.append(target_batch)
        return input_batchs, target_batechs


# prepare the batch to train in every epoch
def pre_multi_batch(batch_data_, batch_size):
    # prepare the batch from the dataset
    batch_data = copy.deepcopy(batch_data_)
    batches = []
    len_batch = len(batch_data)
    start_batch = 0
    while start_batch + batch_size < len_batch:
        batches.append(batch_data[start_batch: start_batch + batch_size])
        start_batch += batch_size
    batches.append(batch_data[start_batch:])

    # padding
    dep_lengths = []
    sen_lengths = []
    svo_batchs = []
    dep_batchs = []
    sen_batchs = []
    target_batchs = []

    for batch in batches:
        dep_length = []
        sen_length = []
        svo_batch = []
        dep_batch = []
        sen_batch = []
        target_batch = []

        for p in batch:
            dep_length.append(len(p[1]))
            sen_length.append(len(p[2]))
            svo_batch.append(p[0])
            dep_batch.append(p[1])
            sen_batch.append(p[2])
            target_batch.append(p[-1])
        svo_batchs.append(svo_batch)
        dep_batchs.append(dep_batch)
        sen_batchs.append(sen_batch)
        dep_lengths.append(dep_length)
        sen_lengths.append(sen_length)
        target_batchs.append(target_batch)
    return svo_batchs, dep_batchs, sen_batchs, dep_lengths, sen_lengths, target_batchs

