from typing import List
from collections import namedtuple
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch import nn
from embedding_model import EmbeddingModel
import torch.nn.functional as F
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate, device):
        super(NMT, self).__init__()
        self.device = device
        self.model_embedding = EmbeddingModel(embed_size, vocab)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.encoder = nn.LSTM(embed_size, hidden_size,
                               bidirectional=True, batch_first=True)
        self.decoder = nn.LSTMCell(hidden_size + embed_size, hidden_size)
        self.h_projection = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.c_projection = nn.Linear(2 * self.hidden_size, hidden_size)
        self.att_projection = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.combined_output_projection = nn.Linear(
            self.hidden_size * 3, hidden_size)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt))
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    def encode(self, source_padded, source_lengths):
        X = self.model_embedding.source(source_padded)
        packed_seq = pack_padded_sequence(
            X, source_lengths, enforce_sorted=False, batch_first=True)
        enc_hidden, (last_hidden, last_cell) = self.encoder(packed_seq)
        enc_hidden, _ = pad_packed_sequence(enc_hidden, batch_first=True)
        init_decode_hidden = self.h_projection(
            torch.cat((last_hidden[0], last_hidden[1]), 1))
        init_decode_cell = self.c_projection(
            torch.cat((last_cell[0], last_cell[1]), 1))
        return enc_hidden, (init_decode_hidden, init_decode_cell)

    def decode(self, enc_hidden, enc_masks, dec_init_state, target_padded):
        target_padded = target_padded[:, :-1]
        batch_size = enc_hidden.shape[0]
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []
        enc_hidden_proj = self.att_projection(enc_hidden)
        Y = self.model_embedding.target(target_padded)
        Y_ts = torch.split(Y, 1, dim=1)
        dec_state = dec_init_state
        for y_t in Y_ts:
            y_t = y_t.squeeze(1)
            ybar_t = torch.cat((y_t, o_prev), 1)
            dec_state, o_prev, _ = self.step(
                ybar_t, dec_state, enc_hidden, enc_hidden_proj, enc_masks)
            combined_outputs.append(o_prev)
        combined_outputs = torch.stack(combined_outputs)
        return combined_outputs

    def step(self, ybar_t, dec_state, enc_hidden, enc_hidden_proj, enc_masks):
        dec_hidden, dec_cell = self.decoder(ybar_t, dec_state)
        dec_state = dec_hidden, dec_cell
        e_t = torch.bmm(enc_hidden_proj, dec_hidden.unsqueeze(2))
        e_t = e_t.squeeze(2)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
        alpha_t = torch.softmax(e_t, 1).unsqueeze(1)
        a_t = torch.bmm(alpha_t, enc_hidden).squeeze(1)
        U_t = torch.cat((dec_hidden, a_t), 1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        combined_output = O_t
        return dec_state, combined_output, e_t

    def forward(self, source, target):
        source_lengths = [len(s) for s in source]
        target_lengths = torch.tensor([len(s) - 1 for s in target])
        source_padded = self.vocab.src.to_input_tensor(source, self.device)
        target_padded = self.vocab.tgt.to_input_tensor(target, self.device)
        enc_hidden, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hidden, source_lengths)
        combined_outputs = self.decode(
            enc_hidden, enc_masks, dec_init_state, target_padded)
        P = self.target_vocab_projection(combined_outputs)
        P = P.permute(1, 0, 2)

        P = pack_padded_sequence(
            P, target_lengths, enforce_sorted=False, batch_first=True).data

        target_label = pack_padded_sequence(
            target_padded[:, 1:], target_lengths, enforce_sorted=False, batch_first=True).data

        return nn.CrossEntropyLoss()(P, target_label)

    def generate_sent_masks(self, enc_hidden, source_lengths):
        enc_masks = torch.zeros(enc_hidden.size(
            0), enc_hidden.size(1), dtype=torch.float, device=self.device)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        print("load modeling------")
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        print(args)
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
        model = model.to(args["device"])
        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate, device=self.device),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    # def beam_serach(self, source_sents, beam_size = 5, max_length = 50):
    #     src = self.vocab.src.to_input_tensor([source_sents], self.device)
    #     encode_hidden, decode_init = self.encode(src, [len(source_sents)])
    #     encode_atten = self.att_projection(encode_hidden)
    #
    #     eos_id = self.vocab.tgt['</s>']
    #     hypotheses = [["<s>"]]
    #     scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
    #     complete_hypotheses = []
    #     t = 0
    #     print(encode_hidden)
    #     while t < max_length and len(target) < beam_size:
    #         t = t + 1
    #         hyp_num = len(hypotheses)
            
            
    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embedding.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word_id == eos_id:
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses
