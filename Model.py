import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):
    """
        Args:
            input: seq_len, batch
        Returns:
            attn: batch, seq_len, hidden_size
            outputs: batch, seq_len, hidden_size
    """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # self.vocab_size = vocab_size
        # self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        self.in_channels = 1
        self.out_channels = opt.hidden_size * 2
        self.kernel_size = opt.kernel_size
        self.kernel = (opt.kernel_size, opt.hidden_size * 2)
        self.stride = 1
        self.padding = ((opt.kernel_size -1) / 2, 0)
        self.layers = opt.enc_layers

        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(opt.word_vec_size, 2*self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride,self.padding)

        self.mapping = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        # self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.bn1 = nn.BatchNorm1d(self.hidden_size)
        # self.bn2 = nn.BatchNorm1d(self.hidden_size * 2)

    def forward(self, questions):
        # inputs = self.embedding(input[0])
        _inputs = questions.view(-1, questions.size(2)) 
        _outputs = self.affine(_inputs)
        _outputs = _outputs.view(questions.size(0), questions.size(1), -1).t() 
        outputs = _outputs
        for i in range(self.layers):
            outputs = outputs.unsqueeze(1) # batch, 1, seq_len, 2*hidden
            outputs = self.conv(outputs) # batch, out_channels, seq_len, 1
            outputs = F.relu(outputs)
            outputs = outputs.squeeze(3).transpose(1,2) # batch, seq_len, 2*hidden
            A, B = outputs.split(self.hidden_size, 2) # A, B: batch, seq_len, hidden
            A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
            B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
            attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
            attn2 = self.mapping(attn) # attn2: batch * seq_len, 2 * hidden
            outputs = attn2.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden
        # outputs = torch.sum(outputs, 2).squeeze(2)
        out = attn2.view(A.size(0), A.size(1), -1) + _outputs # batch, seq_len, 2 * hidden_size
        # print "_outputs", _outputs
        # print "out", out

        return attn, out


class Decoder(nn.Module):
    """
    Decoder
        Args:
            Input: seq_len, batch_size
        return:
            out:
    """

    def __init__(self, opt):
        super(Decoder, self).__init__()

        # self.vocab_size = vocab_size
        # self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size


        self.in_channels = 1
        self.out_channels = opt.hidden_size * 2
        self.kernel_size = opt.kernel_size
        self.kernel = (opt.kernel_size, opt.hidden_size * 2)
        self.stride = 1
        self.padding = ((opt.kernel_size - 1) / 2, 0)
        self.layers = opt.dec_layers

        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(opt.word_vec_size, 2 * self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride, self.padding)
        
        self.mapping = nn.Linear(self.hidden_size, 2*self.hidden_size)

        self.softmax = nn.Softmax()
    # attn_src: src_seq_len, hidden_size
    def forward(self, answers, enc_attn, source_seq_out):
        # inputs = self.embedding(target)
        _inputs = answers.view(-1, answers.size(2))
        outputs = self.affine(_inputs)
        outputs = outputs.view(answers.size(0), answers.size(1), -1).t()
        scrores = []
        for i in range(self.layers):
            outputs = outputs.unsqueeze(1) # batch, 1, seq_len, 2*hidden
            outputs = self.conv(outputs) # batch, out_channels, seq_len + self.kernel_size - 1, 1
            # outputs = outputs.narrow(2, 0, outputs.size(2)-self.kernel_size) # remove the last k elements

            # This is the residual connection,
            # for the output of the conv will add kernel_size/2 elements 
            # before and after the origin input
            if i > 0:
                conv_out = conv_out + outputs

            outputs = F.relu(outputs)
            outputs = outputs.squeeze(3).transpose(1,2) # batch, seq_len, 2*hidden
            A, B = outputs.split(self.hidden_size, 2) # A, B: batch, seq_len, hidden
            A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
            B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
            dec_attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden

            dec_attn2 = self.mapping(dec_attn)
            dec_attn2 = dec_attn2.view(A.size(0), A.size(1), -1) # batch, seq_len_tgt, 2 * hidden_size

            enc_attn = enc_attn.view(A.size(0), -1, A.size(2)) # enc_attn1: batch, seq_len_src, hidden_size
            dec_attn = dec_attn.view(A.size(0), -1, A.size(2)) # dec_attn1: batch, seq_len_tgt, hidden_size

            

            _attn_matrix = torch.bmm(dec_attn, enc_attn.transpose(1,2)) # attn_matrix: batch, seq_len_tgt, seq_len_src
            
            tgt_attn_matrix = self.softmax(_attn_matrix.view(-1, _attn_matrix.size(2)))
            tgt_attn_matrix = tgt_attn_matrix.view(_attn_matrix.size(0), _attn_matrix.size(1), -1) # normalized attn_matrix: batch, seq_len_tgt, seq_len_src
            tgt_attns = torch.bmm(tgt_attn_matrix, source_seq_out) # attns: batch, seq_len_tgt, 2 * hidden_size
            
            tgt_hidden = torch.sum(tgt_attns, 1) # sum | average | max etc.

            src_attn_matrix = self.softmax(_attn_matrix.transpose(1,2).contiguous().view(-1, _attn_matrix.size(1)))
            src_attn_matrix = src_attn_matrix.view(_attn_matrix.size(0), _attn_matrix.size(2), -1) # normalized attn_matrix: batch, seq_len_src, seq_len_tgt
            
            # print "tgt_attn_matrix", tgt_attn_matrix
            # print "tgt_attns", tgt_attns
            # print "src_attn_matrix", src_attn_matrix
            # print "dec_attn2", dec_attn2
            src_attns = torch.bmm(src_attn_matrix, dec_attn2) # attns: batch, seq_len_src, 2 * hidden_size
            src_hidden = torch.sum(src_attns, 1)

            scrore = torch.bmm(src_hidden, tgt_hidden.transpose(1,2)).squeeze(2).squeeze(1)
            scrores += [scrore]
            outputs = dec_attn2 + tgt_attns # outpus: batch, seq_len_tgt - 1, 2 * hidden_size

        scrores = torch.stack(scrores, 1)

        return scrores


class AnswerSelectModel(nn.Module):
    """
    AnswerSelectModel:
    Input:
        encoder:
        decoder:
        attention:
        generator:
    return:
    """
    def __init__(self, encoder, decocer, opt, vocab_size):
        super(AnswerSelectModel, self).__init__()
        self.encoder = encoder
        self.decocer = decocer

        self.word_lut = nn.Embedding(vocab_size, opt.word_vec_size, padding_idx=0)
    
    def forward(self, source, target):
        # (1) Embedding
        if isinstance(source, tuple):
            lengths = source[1].data.view(-1).tolist() # lengths data is wraped inside a Variable
            source_embedding = pack(self.word_lut(source[0]), lengths)
        else:
            source_embedding = self.word_lut(source)
        target_embedding = self.word_lut(target)

        # (2) QuestionEncoder
        # attn: batch, seq_len, hidden
        # out: batch, seq_len, 2 * hidden_size
        attn, source_seq_out = self.encoder(source_embedding)

        # (3) AnswerEncoder
        # batch, seq_len_tgt, hidden_size
        scrores = self.decocer(target_embedding, attn, source_seq_out)

        return scrores

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs is not None:
            pretrained = torch.load(opt.pre_word_vecs)
            self.word_lut.weight.data.copy_(pretrained)