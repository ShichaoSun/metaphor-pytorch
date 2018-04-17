import torch
import torch.nn as nn
import torch.nn.functional as f

USE_CUDA = torch.cuda.is_available()


class BaseSSN(nn.Module):
    def __init__(self, embedding_size, metaphor_size, d_size, dropout=0.5):
        super(BaseSSN, self).__init__()
        self.metaphor_size = metaphor_size
        self.dropout = dropout
        self.embedding_size = embedding_size

        self.em_dropout = nn.Dropout(dropout)
        self.g_gate = nn.Linear(embedding_size, embedding_size)
        self.first_to_metaphor = nn.Linear(embedding_size, metaphor_size)
        self.second_to_metaphor = nn.Linear(embedding_size, metaphor_size)
        self.to_sim_feature = nn.Linear(metaphor_size, d_size)
        self.to_prediction = nn.Linear(d_size, 1)

    def ssn(self, x1, x2):
        x2_ = f.sigmoid(self.g_gate(x1))*x2
        z1 = f.tanh(self.first_to_metaphor(x1))
        z2 = f.tanh(self.second_to_metaphor(x2_))
        m = z1 * z2
        d = f.tanh(self.to_sim_feature(m))
        return f.sigmoid(self.to_prediction(d))

    def forward(self, input_seqs_embedded, hidden=None):
        # Get the word embedding of bi-context in SVO tuple
        embedded = self.em_dropout(input_seqs_embedded)  # 3 x B x E
        y1 = self.ssn(embedded[0], embedded[1])
        y2 = self.ssn(embedded[1], embedded[2])
        return torch.max(y1, y2)


class SvoMismatchDegree(nn.Module):
    def __init__(self, embedding_size, hidden_size, metaphor_size, d_size, dropout=0.5):
        super(SvoMismatchDegree, self).__init__()

        self.hidden_size = hidden_size
        self.metaphor_size = metaphor_size
        self.dropout = dropout
        self.embedding_size = embedding_size

        self.em_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout, bidirectional=True)

        self.to_metaphor = nn.Linear(hidden_size, metaphor_size)
        self.to_sim_feature = nn.Linear(metaphor_size, d_size)
        self.to_prediction = nn.Linear(d_size, 1)

    def ssn(self, x1, x2):
        z1 = f.tanh(self.to_metaphor(x1))
        z2 = f.tanh(self.to_metaphor(x2))
        m = z1 * z2
        d = f.tanh(self.to_sim_feature(m))
        return f.sigmoid(self.to_prediction(d))

    def forward(self, input_seqs_embedded, hidden=None):
        # Get the word embedding of bi-context in SVO tuple
        embedded = self.em_dropout(input_seqs_embedded)  # 3 x B x E
        outputs, hidden = self.lstm(embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # 3 x B x H
        y1 = self.ssn(outputs[0], outputs[1])
        y2 = self.ssn(outputs[0], outputs[2])
        y3 = self.ssn(outputs[1], outputs[2])
        y4 = torch.max(y1, y2)
        return torch.max(y3, y4)


class DepMismatchDegree(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.5):
        super(DepMismatchDegree, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding_size = embedding_size

        self.em_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout, bidirectional=True)

        self.to_prediction = nn.Linear(hidden_size, 1)

    def forward(self, input_seqs_embedded, input_lengths, hidden=None):
        embedded = self.em_dropout(input_seqs_embedded)  # S x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # S x B x H

        return f.sigmoid(self.to_prediction(outputs[0]))


class SenSemantic(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.5):
        super(SenSemantic, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding_size = embedding_size

        self.em_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout, bidirectional=True)

        self.to_prediction = nn.Linear(hidden_size, 1)
        self.attn = nn.Linear(hidden_size, 1)

    def attention(self, lstm_outputs, seq_mask):
        batch_size = lstm_outputs.size(1)
        lstm_outputs = lstm_outputs.view(-1, self.hidden_size)  # (S x B) x H
        outputs_attn = self.attn(lstm_outputs)
        outputs_attn = outputs_attn.view(-1, batch_size, 1)  # S x B x 1
        outputs_attn = outputs_attn.transpose(0, 1)  # B x S x 1
        outputs_attn = outputs_attn.squeeze(2)
        outputs_attn.masked_fill_(seq_mask, -float('1e12'))
        return f.softmax(outputs_attn, 1)  # B x S

    def forward(self, input_seqs_embedded, input_lengths, seq_mask, hidden=None):
        embedded = self.em_dropout(input_seqs_embedded)  # S x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # S x B x H
        # Sum bidirectional outputs
        attn = self.attention(outputs, seq_mask)  # B x S
        attn = attn.unsqueeze(1)  # B x 1 X S
        outputs = outputs.transpose(0, 1)  # B x S x H
        feature = attn.bmm(outputs)  # B x 1 x H
        feature = feature.squeeze(1)  # B x H
        return f.sigmoid(self.to_prediction(feature))

