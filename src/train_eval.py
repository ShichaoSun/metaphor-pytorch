import math
import numpy
import torch.optim
from torch.autograd import Variable
from .model import *
from .loss import *


def time_since(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def train_svo(svo, target, word2vec, model, optimizer, clip):
    oov = word2vec.index2word[0]
    input_list = []
    for s in svo:
        temp = []
        if s.s_id == 999 or (s.s_lemma not in word2vec and s.s_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.s_surface_form in word2vec:
            temp.append(word2vec[s.s_surface_form].tolist())
        else:
            temp.append(word2vec[s.s_lemma].tolist())
        if s.v_id == 999 or (s.v_lemma not in word2vec and s.v_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.v_surface_form in word2vec:
            temp.append(word2vec[s.v_surface_form].tolist())
        else:
            temp.append(word2vec[s.v_lemma].tolist())
        if s.o_id == 999 or (s.o_lemma not in word2vec and s.o_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.o_surface_form in word2vec:
            temp.append(word2vec[s.o_surface_form].tolist())
        else:
            temp.append(word2vec[s.o_lemma].tolist())
        input_list.append(temp)
    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    target_var = Variable(torch.FloatTensor(target))
    if torch.cuda.is_available():
        input_var = input_var.cuda()
        target_var = target_var.cuda()
    model.train()
    optimizer.zero_grad()

    prediction_var = model(input_var)
    prediction_var = prediction_var.squeeze()
    loss = squared_error(prediction_var, target_var)
    loss.backward()

    # Clip gradient norms
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    # Update parameters with optimizers
    optimizer.step()
    return loss.data[0]


def valid_svo(svo, word2vec, model):
    oov = word2vec.index2word[0]
    input_list = []
    for s in svo:
        temp = []
        if s.s_id == 999 or (s.s_lemma not in word2vec and s.s_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.s_surface_form in word2vec:
            temp.append(word2vec[s.s_surface_form].tolist())
        else:
            temp.append(word2vec[s.s_lemma].tolist())
        if s.v_id == 999 or (s.v_lemma not in word2vec and s.v_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.v_surface_form in word2vec:
            temp.append(word2vec[s.v_surface_form].tolist())
        else:
            temp.append(word2vec[s.v_lemma].tolist())
        if s.o_id == 999 or (s.o_lemma not in word2vec and s.o_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.o_surface_form in word2vec:
            temp.append(word2vec[s.o_surface_form].tolist())
        else:
            temp.append(word2vec[s.o_lemma].tolist())
        input_list.append(temp)
    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    if torch.cuda.is_available():
        input_var = input_var.cuda()
    model.eval()
    prediction_var = model(input_var)
    prediction_var = prediction_var.squeeze()
    prediction_var = prediction_var >= 0.5
    return prediction_var.data.tolist()


def eval_svo(svo, word2vec, model):
    oov = word2vec.index2word[0]
    input_list = []
    for s in svo:
        temp = []
        if s.s_surface_form in word2vec:
            temp.append(word2vec[s.s_surface_form].tolist())
        elif s.s_lemma in word2vec:
            temp.append(word2vec[s.s_lemma].tolist())
        else:
            temp.append(word2vec[oov].tolist())
        if s.v_surface_form in word2vec:
            temp.append(word2vec[s.v_surface_form].tolist())
        elif s.v_lemma in word2vec:
            temp.append(word2vec[s.v_lemma].tolist())
        else:
            temp.append(word2vec[oov].tolist())
        if s.o_surface_form in word2vec:
            temp.append(word2vec[s.o_surface_form].tolist())
        elif s.o_lemma in word2vec:
            temp.append(word2vec[s.o_lemma].tolist())
        else:
            temp.append(word2vec[oov].tolist())
        input_list.append(temp)
    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    if torch.cuda.is_available():
        input_var = input_var.cuda()
    model.eval()
    prediction_var = model(input_var)
    prediction_var = prediction_var.squeeze()
    prediction_var = prediction_var >= 0.5
    return prediction_var.data.tolist()


def train_dep(dep, lengths, target, word2vec, model, optimizer, clip, sort_flag=True):
    oov = word2vec.index2word[0]
    input_list = []
    max_length = max(lengths)
    for idx, s in enumerate(dep):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - lengths[idx])])

    if not sort_flag:
        input_list = [input_dep for _, input_dep in sorted(zip(lengths, input_list), key=lambda x: x[0], reverse=True)]
        target = [input_dep for _, input_dep in sorted(zip(lengths, target), key=lambda x: x[0], reverse=True)]
        lengths = sorted(lengths, key=lambda x: x, reverse=True)

    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    target_var = Variable(torch.FloatTensor(target))
    if torch.cuda.is_available():
        input_var = input_var.cuda()
        target_var = target_var.cuda()
    model.train()
    optimizer.zero_grad()

    prediction_var = model(input_var, lengths)
    prediction_var = prediction_var.squeeze()
    loss = squared_error(prediction_var, target_var)
    loss.backward()

    # Clip gradient norms
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    # Update parameters with optimizers
    optimizer.step()
    return loss.data[0]


def eval_dep(dep, lengths, word2vec, model, sort_flag=True):
    oov = word2vec.index2word[0]
    input_list = []
    max_length = max(lengths)
    for idx, s in enumerate(dep):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - lengths[idx])])

    if not sort_flag:
        input_list = [input_dep for _, input_dep in sorted(zip(lengths, input_list), key=lambda x: x[0], reverse=True)]
        pre_lengths = lengths
        lengths = sorted(lengths, key=lambda x: x, reverse=True)

    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    if torch.cuda.is_available():
        input_var = input_var.cuda()
    model.eval()
    prediction_var = model(input_var, lengths)
    prediction_var = prediction_var.squeeze()

    if not sort_flag:
        # for index to restore the input_list
        pre_index = sorted(range(len(pre_lengths)), key=lambda x: pre_lengths[x], reverse=True)
        post_index = sorted(range(len(pre_lengths)), key=lambda x: pre_index[x])
        post_index = Variable(torch.LongTensor(post_index))
        if USE_CUDA:
            post_index = post_index.cuda()
        prediction_var = torch.index_select(prediction_var, 0, post_index)
    prediction_var = prediction_var >= 0.5
    return prediction_var.data.tolist()


def train_sen(sen, lengths, target, word2vec, model, optimizer, clip, sort_flag=True):
    oov = word2vec.index2word[0]
    input_list = []
    max_length = max(lengths)
    for idx, s in enumerate(sen):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - lengths[idx])])

    if not sort_flag:
        input_list = [input_dep for _, input_dep in sorted(zip(lengths, input_list), key=lambda x: x[0], reverse=True)]
        target = [input_dep for _, input_dep in sorted(zip(lengths, target), key=lambda x: x[0], reverse=True)]
        lengths = sorted(lengths, key=lambda x: x, reverse=True)

    seq_mask_list = []
    for idx, l in enumerate(lengths):
        seq_mask_list.append([0 for _ in range(l)] + [1 for _ in range(max_length - l)])

    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    target_var = Variable(torch.FloatTensor(target))
    seq_mask = Variable(torch.ByteTensor(seq_mask_list))

    if torch.cuda.is_available():
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        seq_mask = seq_mask.cuda()

    model.train()
    optimizer.zero_grad()

    prediction_var = model(input_var, lengths, seq_mask)
    prediction_var = prediction_var.squeeze()
    loss = squared_error(prediction_var, target_var)
    loss.backward()

    # Clip gradient norms
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    # Update parameters with optimizers
    optimizer.step()
    return loss.data[0]


def eval_sen(sen, lengths, word2vec, model, sort_flag=True):
    oov = word2vec.index2word[0]
    input_list = []
    max_length = max(lengths)
    for idx, s in enumerate(sen):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - lengths[idx])])

    if not sort_flag:
        input_list = [input_dep for _, input_dep in sorted(zip(lengths, input_list), key=lambda x: x[0], reverse=True)]
        pre_lengths = lengths
        lengths = sorted(lengths, key=lambda x: x, reverse=True)

    seq_mask_list = []
    for idx, l in enumerate(lengths):
        seq_mask_list.append([0 for _ in range(l)] + [1 for _ in range(max_length - l)])

    input_var = Variable(torch.FloatTensor(input_list)).transpose(0, 1)
    seq_mask = Variable(torch.ByteTensor(seq_mask_list))

    if torch.cuda.is_available():
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    model.eval()
    prediction_var = model(input_var, lengths, seq_mask)
    prediction_var = prediction_var.squeeze()

    if not sort_flag:
        # for index to restore the input_list
        pre_index = sorted(range(len(pre_lengths)), key=lambda x: pre_lengths[x], reverse=True)
        post_index = sorted(range(len(pre_lengths)), key=lambda x: pre_index[x])
        post_index = Variable(torch.LongTensor(post_index))
        if USE_CUDA:
            post_index = post_index.cuda()
        prediction_var = torch.index_select(prediction_var, 0, post_index)

    prediction_var = prediction_var >= 0.5
    return prediction_var.data.tolist()


def valid_multi(svo, dep, sen, dep_lengths, sen_lengths, word2vec, svo_model, dep_model, sen_model):
    oov = word2vec.index2word[0]
    svo_input_list = []
    for s in svo:
        temp = []
        if s.s_id == 999 or (s.s_lemma not in word2vec and s.s_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.s_surface_form in word2vec:
            temp.append(word2vec[s.s_surface_form].tolist())
        else:
            temp.append(word2vec[s.s_lemma].tolist())
        if s.v_id == 999 or (s.v_lemma not in word2vec and s.v_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.v_surface_form in word2vec:
            temp.append(word2vec[s.v_surface_form].tolist())
        else:
            temp.append(word2vec[s.v_lemma].tolist())
        if s.o_id == 999 or (s.o_lemma not in word2vec and s.o_surface_form not in word2vec):
            temp.append(word2vec[oov].tolist())
        elif s.o_surface_form in word2vec:
            temp.append(word2vec[s.o_surface_form].tolist())
        else:
            temp.append(word2vec[s.o_lemma].tolist())
        svo_input_list.append(temp)
    svo_input_var = Variable(torch.FloatTensor(svo_input_list)).transpose(0, 1)
    if torch.cuda.is_available():
        svo_input_var = svo_input_var.cuda()
    svo_model.eval()
    svo_prediction = svo_model(svo_input_var)
    svo_prediction = svo_prediction.squeeze()

    dep_input_list = []
    max_length = max(dep_lengths)
    for idx, s in enumerate(dep):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        dep_input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - dep_lengths[idx])])

    dep_lengths_input = sorted(dep_lengths, key=lambda x: x, reverse=True)
    dep_input_list = [dep_input for _, dep_input in sorted(zip(dep_lengths, dep_input_list),
                                                           key=lambda x: x[0], reverse=True)]

    # for index to restore the input_list
    pre_index = sorted(range(len(dep_lengths)), key=lambda x: dep_lengths[x], reverse=True)
    post_index = sorted(range(len(dep_lengths)), key=lambda x: pre_index[x])

    dep_input_var = Variable(torch.FloatTensor(dep_input_list)).transpose(0, 1)
    post_index = Variable(torch.LongTensor(post_index))

    if torch.cuda.is_available():
        dep_input_var = dep_input_var.cuda()
        post_index = post_index.cuda()

    dep_model.eval()
    dep_prediction = dep_model(dep_input_var, dep_lengths_input)
    dep_prediction = dep_prediction.squeeze()
    dep_prediction = torch.index_select(dep_prediction, 0, post_index)

    sen_input_list = []
    max_length = max(sen_lengths)
    for idx, s in enumerate(sen):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        sen_input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - sen_lengths[idx])])

    sen_lengths_input = sorted(sen_lengths, key=lambda x: x, reverse=True)
    sen_input_list = [sen_input for _, sen_input in sorted(zip(sen_lengths, sen_input_list),
                                                           key=lambda x: x[0], reverse=True)]

    # for index to restore the input_list
    pre_index = sorted(range(len(sen_lengths)), key=lambda x: sen_lengths[x], reverse=True)
    post_index = sorted(range(len(sen_lengths)), key=lambda x: pre_index[x])

    seq_mask_list = []
    for idx, l in enumerate(sen_lengths_input):
        seq_mask_list.append([0 for _ in range(l)] + [1 for _ in range(max_length - l)])

    sen_input_var = Variable(torch.FloatTensor(sen_input_list)).transpose(0, 1)
    seq_mask = Variable(torch.ByteTensor(seq_mask_list))
    post_index = Variable(torch.LongTensor(post_index))

    if torch.cuda.is_available():
        sen_input_var = sen_input_var.cuda()
        seq_mask = seq_mask.cuda()
        post_index = post_index.cuda()

    sen_model.eval()
    sen_prediction = sen_model(sen_input_var, sen_lengths_input, seq_mask)
    sen_prediction = sen_prediction.squeeze()
    sen_prediction = torch.index_select(sen_prediction, 0, post_index)

    prediction = svo_prediction + dep_prediction
    prediction += sen_prediction
    prediction = prediction / 3
    prediction = prediction >= 0.5
    return prediction.data.tolist()


def eval_multi(svo, dep, sen, dep_lengths, sen_lengths, word2vec, svo_model, dep_model, sen_model):
    oov = word2vec.index2word[0]
    svo_input_list = []
    for s in svo:
        temp = []
        if s.s_surface_form in word2vec:
            temp.append(word2vec[s.s_surface_form].tolist())
        elif s.s_lemma in word2vec:
            temp.append(word2vec[s.s_lemma].tolist())
        else:
            temp.append(word2vec[oov].tolist())
        if s.v_surface_form in word2vec:
            temp.append(word2vec[s.v_surface_form].tolist())
        elif s.v_lemma in word2vec:
            temp.append(word2vec[s.v_lemma].tolist())
        else:
            temp.append(word2vec[oov].tolist())
        if s.o_surface_form in word2vec:
            temp.append(word2vec[s.o_surface_form].tolist())
        elif s.o_lemma in word2vec:
            temp.append(word2vec[s.o_lemma].tolist())
        else:
            temp.append(word2vec[oov].tolist())
        svo_input_list.append(temp)
    svo_input_var = Variable(torch.FloatTensor(svo_input_list)).transpose(0, 1)
    if torch.cuda.is_available():
        svo_input_var = svo_input_var.cuda()
    svo_model.eval()
    svo_prediction = svo_model(svo_input_var)
    svo_prediction = svo_prediction.squeeze()

    dep_input_list = []
    max_length = max(dep_lengths)
    for idx, s in enumerate(dep):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        dep_input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - dep_lengths[idx])])

    dep_lengths_input = sorted(dep_lengths, key=lambda x: x, reverse=True)
    dep_input_list = [dep_input for _, dep_input in sorted(zip(dep_lengths, dep_input_list),
                                                           key=lambda x: x[0], reverse=True)]

    # for index to restore the input_list
    pre_index = sorted(range(len(dep_lengths)), key=lambda x: dep_lengths[x], reverse=True)
    post_index = sorted(range(len(dep_lengths)), key=lambda x: pre_index[x])

    dep_input_var = Variable(torch.FloatTensor(dep_input_list)).transpose(0, 1)
    post_index = Variable(torch.LongTensor(post_index))

    if torch.cuda.is_available():
        dep_input_var = dep_input_var.cuda()
        post_index = post_index.cuda()

    dep_model.eval()
    dep_prediction = dep_model(dep_input_var, dep_lengths_input)
    dep_prediction = dep_prediction.squeeze()
    dep_prediction = torch.index_select(dep_prediction, 0, post_index)

    sen_input_list = []
    max_length = max(sen_lengths)
    for idx, s in enumerate(sen):
        temp = []
        for w in s:
            if w.lemma not in word2vec and w.surface_form not in word2vec:
                temp.append(word2vec[oov].tolist())
            elif w.surface_form in word2vec:
                temp.append(word2vec[w.surface_form].tolist())
            else:
                temp.append(word2vec[w.lemma].tolist())
        sen_input_list.append(temp + [word2vec[oov].tolist() for _ in range(max_length - sen_lengths[idx])])

    sen_lengths_input = sorted(sen_lengths, key=lambda x: x, reverse=True)
    sen_input_list = [sen_input for _, sen_input in sorted(zip(sen_lengths, sen_input_list),
                                                           key=lambda x: x[0], reverse=True)]

    # for index to restore the input_list
    pre_index = sorted(range(len(sen_lengths)), key=lambda x: sen_lengths[x], reverse=True)
    post_index = sorted(range(len(sen_lengths)), key=lambda x: pre_index[x])

    seq_mask_list = []
    for idx, l in enumerate(sen_lengths_input):
        seq_mask_list.append([0 for _ in range(l)] + [1 for _ in range(max_length - l)])

    sen_input_var = Variable(torch.FloatTensor(sen_input_list)).transpose(0, 1)
    seq_mask = Variable(torch.ByteTensor(seq_mask_list))
    post_index = Variable(torch.LongTensor(post_index))

    if torch.cuda.is_available():
        sen_input_var = sen_input_var.cuda()
        seq_mask = seq_mask.cuda()
        post_index = post_index.cuda()

    sen_model.eval()
    sen_prediction = sen_model(sen_input_var, sen_lengths_input, seq_mask)
    sen_prediction = sen_prediction.squeeze()
    sen_prediction = torch.index_select(sen_prediction, 0, post_index)

    prediction = svo_prediction + dep_prediction
    prediction += sen_prediction
    prediction = prediction / 3
    prediction = prediction >= 0.5
    return prediction.data.tolist()
