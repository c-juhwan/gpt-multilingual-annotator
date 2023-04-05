# Standard Library Modules
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
from collections import defaultdict
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class CaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(CaptioningModel, self).__init__()
        self.args = args

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, images: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): (batch_size, 3, 224, 224)
            caption_ids (torch.Tensor): (batch_size, max_seq_len)
        """

        features = self.encoder(images)
        seq_logits = self.decoder(features, caption_ids)

        return seq_logits

    def inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): (batch_size, 3, 224, 224)
        """

        features = self.encoder(images)

        if self.args.decoding_strategy == 'greedy':
            seq_output = self.decoder.greedy_generate(features)
        elif self.args.decoding_strategy == 'beam':
            seq_output = self.decoder.beam_generate(features)
        else:
            raise NotImplementedError(f'Invalid decoding strategy: {self.args.decoding_strategy}')

        return seq_output

class Encoder(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(Encoder, self).__init__()
        self.args = args

        self.encoder_type = args.encoder_type
        self.encoder_pretrained = args.encoder_pretrained

        if self.encoder_type == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT' if self.encoder_pretrained else None)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_output_dim = resnet.fc.in_features
        elif self.encoder_type == 'efficientnet_b0':
            effnet = models.efficientnet_b0(weights='DEFAULT' if self.encoder_pretrained else None)
            self.feature_extractor = nn.Sequential(*list(effnet.children())[:-1])
            self.feature_output_dim = effnet.classifier[1].in_features
        elif self.encoder_type == 'vit_b_16':
            self.vit = models.vit_b_16(weights='DEFAULT' if self.encoder_pretrained else None)
            self.feature_extractor = nn.Sequential(*list(self.vit.children())[:-1])
            self.feature_output_dim = self.vit.heads[0].in_features
        else:
            raise NotImplementedError(f'Invalid encoder type: {self.encoder_type}')

        # Define the output layer
        self.out = nn.Sequential(
            nn.Linear(in_features=self.feature_output_dim, out_features=self.feature_output_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=self.feature_output_dim // 2, out_features=args.embed_size)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): [batch_size, 3, 224, 224]
        """

        if 'vit' in self.encoder_type:
            encoder = self.feature_extractor[1]
            processed_img = self.vit._process_input(images)

            n = processed_img.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            processed_img = torch.cat([batch_class_token, processed_img], dim=1)

            encoded_img = encoder(processed_img)
            features = encoded_img[:, 0]
        else:
            features = self.feature_extractor(images)

        features = features.view(features.size(0), -1) # Flatten to (batch_size, features_dim)
        features = self.out(features) # (batch_size, output_dim)

        return features

class Decoder(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(Decoder, self).__init__()
        self.args = args

        self.word_embed = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embed_size)
        self.pos_embed = nn.Embedding(num_embeddings=args.max_seq_len, embedding_dim=args.embed_size)

        self.decoder_type = args.decoder_type
        # Define the decoder
        if self.decoder_type == 'lstm':
            self.decoder = nn.LSTM(input_size=args.embed_size, hidden_size=args.hidden_size,
                                   num_layers=args.decoder_lstm_nlayers,
                                   bidirectional=False, batch_first=True)
        elif self.decoder_type == 'transformer':
            decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size,
                                                       nhead=args.decoder_transformer_nhead)
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                                 num_layers=args.decoder_transformer_nlayers)
        else:
            raise NotImplementedError(f'Invalid decoder type: {self.decoder_type}')

        # Define the output layer
        self.out = nn.Sequential(
            nn.Linear(in_features=args.embed_size, out_features=args.embed_size * 4),
            nn.GELU(),
            nn.LayerNorm(normalized_shape=args.embed_size * 4),
            nn.Linear(in_features=args.embed_size * 4, out_features=args.vocab_size)
        )

    def forward(self, features: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): (batch_size, embed_size)
            caption_ids (torch.Tensor): (batch_size, max_seq_len)
        """

        feature_embed = features.unsqueeze(1) # (batch_size, 1, embed_size)
        feature_embed = feature_embed.repeat(1, caption_ids.size(1), 1) # (batch_size, max_seq_len, embed_size)
        word_embed = self.word_embed(caption_ids) # (batch_size, max_seq_len, embed_size)
        pos_embed = self.pos_embed(torch.arange(caption_ids.size(1)).to(caption_ids.device)) # (max_seq_len, embed_size)
        pos_embed = pos_embed.unsqueeze(0).repeat(caption_ids.size(0), 1, 1) # (batch_size, max_seq_len, embed_size)

        if self.decoder_type == 'lstm':
            decoder_input = word_embed + feature_embed # (batch_size, max_seq_len, embed_size) - RNNs doesn't require positional embedding

            # Initialize the hidden state as the feature vector
            h_init = features.unsqueeze(0).repeat(self.args.decoder_lstm_nlayers, 1, 1) # (nlayers, batch_size, hidden_size)
            c_init = torch.zeros_like(h_init) # (nlayers, batch_size, hidden_size)

            # Pass the input through the LSTM
            decoder_output, _ = self.decoder(decoder_input, (h_init, c_init)) # (batch_size, max_seq_len, hidden_size)
        elif self.decoder_type == 'transformer':
            decoder_input = word_embed + feature_embed + pos_embed # (batch_size, max_seq_len, embed_size) - Transformers require positional embedding

            tgt_mask = self.generate_square_subsequent_mask(caption_ids.size(1), device=caption_ids.device) # (max_seq_len, max_seq_len)
            tgt_key_padding_mask = (caption_ids == self.args.pad_token_id) # (batch_size, max_seq_len)

            # Pass the input through the Transformer
            decoder_input = decoder_input.permute(1, 0, 2) # (max_seq_len, batch_size, embed_size) - no batch_first for TransformerDecoder
            decoder_memory = feature_embed.permute(1, 0, 2) # (max_seq_len, batch_size, embed_size) - no batch_first for TransformerDecoder
            decoder_output = self.decoder(decoder_input, decoder_memory,
                                          tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask) # (max_seq_len, batch_size, embed_size) - no batch_first
            decoder_output = decoder_output.permute(1, 0, 2) # (batch_size, max_seq_len, embed_size)

        # Pass the output through the output layer
        decoder_logits = self.out(decoder_output) # (batch_size, max_seq_len, vocab_size)
        return decoder_logits

    def greedy_generate(self, features: torch.Tensor) -> torch.Tensor:
        # Greedy decoding
        batch_size = features.size(0)
        feature_embed = features.unsqueeze(1) # (batch_size, 1, embed_size)

        decoder_input = torch.tensor([self.args.bos_token_id] * batch_size) # (batch_size)
        decoder_input = decoder_input.unsqueeze(1).to(features.device) # (batch_size, 1)

        for step in range(self.args.max_seq_len - 1): # -1 for the <bos> token
            if self.decoder_type == 'lstm':
                word_embed = self.word_embed(decoder_input)
                decoder_input_embed = word_embed + feature_embed.repeat(1, decoder_input.size(1), 1)

                # Initialize the hidden state as the feature vector
                h_init = features.unsqueeze(0).repeat(self.args.decoder_lstm_nlayers, 1, 1) # (nlayers, batch_size, hidden_size)
                c_init = torch.zeros_like(h_init) # (nlayers, batch_size, hidden_size)

                # Pass the input through the LSTM
                decoder_output, _ = self.decoder(decoder_input_embed, (h_init, c_init)) # (batch_size, cur_seq_len, hidden_size)

            elif self.decoder_type == 'transformer':
                word_embed = self.word_embed(decoder_input)
                pos_embed = self.pos_embed(torch.arange(decoder_input.size(1)).to(decoder_input.device)) # (cur_seq_len, embed_size)
                pos_embed = pos_embed.unsqueeze(0).repeat(decoder_input.size(0), 1, 1) # (batch_size, cur_seq_len, embed_size)

                decoder_input_embed = word_embed + feature_embed.repeat(1, decoder_input.size(1), 1) + pos_embed # (batch_size, cur_seq_len, embed_size)

                tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1), device=decoder_input.device) # (cur_seq_len, cur_seq_len)
                tgt_key_padding_mask = (decoder_input == self.args.pad_token_id) # (batch_size, cur_seq_len)

                # Pass the input through the Transformer
                decoder_input_embed = decoder_input_embed.permute(1, 0, 2) # (cur_seq_len, batch_size, embed_size) - no batch_first for TransformerDecoder
                decoder_memory = feature_embed.permute(1, 0, 2) # (1, batch_size, embed_size) - no batch_first for TransformerDecoder
                decoder_output = self.decoder(decoder_input_embed, decoder_memory,
                                              tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask) # (cur_seq_len, batch_size, embed_size) - no batch_first
                decoder_output = decoder_output.permute(1, 0, 2) # (batch_size, cur_seq_len, embed_size)

            # Pass the output through the output layer
            decoder_logits = self.out(decoder_output) # (batch_size, cur_seq_len, vocab_size)
            next_token_logits = decoder_logits[:, -1, :] # (batch_size, vocab_size)
            # Avoid generating <s> and <pad> tokens
            next_token_logits[:, self.args.bos_token_id] = -float('inf')
            next_token_logits[:, self.args.pad_token_id] = -float('inf')
            # Generate the next token
            next_token = torch.argmax(next_token_logits, dim=1).unsqueeze(1) # (batch_size, 1)
            # Concatenate next token to decoder_input
            decoder_input = torch.cat([decoder_input, next_token], dim=1) # (batch_size, cur_seq_len + 1)

        # Remove <bos> token from the output
        best_seq = decoder_input[:, 1:]

        return best_seq

    def beam_generate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Beam search with single batch

        Args:
            features: (batch_size, embed_size)
        """
        batch_size = features.size(0)
        assert batch_size == 1, 'Beam search only supports batch size of 1'
        beam_size = self.args.beam_size
        feature_embed = features.unsqueeze(1) # (1, 1, embed_size)

        # Initialize the decoder input with <bos> token
        decoder_input = torch.tensor([self.args.bos_token_id] * beam_size, device=features.device).unsqueeze(1) # (beam_size, 1)
        transformer_decoder_memory = features.repeat(beam_size, 1).unsqueeze(0) # (1, beam_size, embed_size) - no batch_first for TransformerDecoder

        # Initialize beam search variables
        current_beam_scores = torch.zeros(beam_size, device=features.device) # (beam_size)
        final_beam_scores = torch.zeros(beam_size, device=features.device) # (beam_size)
        final_beam_seqs = torch.zeros(beam_size, self.args.max_seq_len, device=features.device).long() # (beam_size, max_seq_len-1)
        beam_complete = torch.zeros(beam_size, device=features.device).bool() # (beam_size)

        # Beam search
        for step in range(self.args.max_seq_len - 1): # -1 for <bos>
            if self.decoder_type == 'lstm':
                word_embed = self.word_embed(decoder_input)
                decoder_input_embed = word_embed + feature_embed.repeat(1, decoder_input.size(1), 1) # (beam_size, cur_seq_len, embed_size)

                # Initialize the hidden state as the feature vector
                h_init = features.unsqueeze(0).repeat(self.args.decoder_lstm_nlayers, 1, 1) # (nlayers, beam_size, hidden_size)
                c_init = torch.zeros_like(h_init) # (nlayers, beam_size, hidden_size)

                # Pass the input through the LSTM
                decoder_output, _ = self.decoder(decoder_input_embed, (h_init, c_init)) # (beam_size, cur_seq_len, hidden_size)
            elif self.decoder_type == 'transformer':
                word_embed = self.word_embed(decoder_input)
                pos_embed = self.pos_embed(torch.arange(decoder_input.size(1)).to(decoder_input.device)) # (cur_seq_len, embed_size)
                pos_embed = pos_embed.unsqueeze(0).repeat(decoder_input.size(0), 1, 1) # (beam_size, cur_seq_len, embed_size)

                decoder_input_embed = word_embed + feature_embed.repeat(beam_size, decoder_input.size(1), 1) + pos_embed # (beam_size, cur_seq_len, embed_size)

                tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1), device=decoder_input.device) # (cur_seq_len, cur_seq_len)
                tgt_key_padding_mask = (decoder_input == self.args.pad_token_id) # (beam_size, cur_seq_len)

                # Pass the input through the Transformer
                decoder_input_embed = decoder_input_embed.permute(1, 0, 2) # (cur_seq_len, beam_size, embed_size) - no batch_first for TransformerDecoder
                decoder_output = self.decoder(decoder_input_embed, transformer_decoder_memory,
                                              tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask) # (cur_seq_len, beam_size, embed_size) - no batch_first
                decoder_output = decoder_output.permute(1, 0, 2) # (beam_size, cur_seq_len, embed_size)

            # Pass the output through the output layer
            decoder_logits = self.out(decoder_output) # (beam_size, cur_seq_len, vocab_size)
            decoder_score = F.log_softmax(decoder_logits[:, -1, :], dim=1) # (beam_size, vocab_size)

            decoder_score[:, self.args.bos_token_id] = -float('inf') # Avoid generating <s> token
            decoder_score[:, self.args.pad_token_id] = -float('inf') # Avoid generating <pad> token
            if step == 0:
                decoder_score[:, self.args.eos_token_id] = -float('inf')

                # As we are using the same decoder input for all beams, we need to make sure that the first token of each beam is different
                # Get the top-k tokens for first beam
                topk_score, topk_token = decoder_score[0, :].topk(beam_size, dim=0, largest=True, sorted=True) # (beam_size)
                topk_beam_idx = torch.arange(beam_size, device=features.device) # (beam_size)
                topk_token_idx = topk_token # (beam_size)
            else:
                next_token_score = current_beam_scores.unsqueeze(1) + decoder_score # (beam_size, vocab_size)
                next_token_score = next_token_score.view(-1) # (beam_size * vocab_size)

                # Get the top k tokens but avoid getting the same token across different beams
                topk_score, topk_token = torch.topk(next_token_score, beam_size, dim=0, largest=True, sorted=True) # (beam_size)
                topk_beam_idx = topk_token // self.args.vocab_size # (beam_size)
                topk_token_idx = topk_token % self.args.vocab_size # (beam_size)

            # Update the current beam tokens and scores
            current_beam_scores = topk_score # (beam_size)

            # Update the beam sequences - attach the new word to the end of the current beam sequence
            # load the top beam_size sequences for each batch
            # and attach the new word to the end of the current beam sequence
            cur_beam_seq = decoder_input.view(beam_size, -1) # (beam_size, cur_seq_len)
            new_beam_seq = cur_beam_seq[topk_beam_idx, :] # (beam_size, cur_seq_len) - topk_beam_idx is broadcasted to (beam_size, cur_seq_len)
            decoder_input = torch.cat([new_beam_seq, topk_token_idx.unsqueeze(1)], dim=1) # (beam_size, cur_seq_len + 1)

            # If the <eos> token is generated,
            # set the score of the <eos> token to -inf so that it is not generated again
            # and save the sequence
            for beam_idx, token_idx in enumerate(topk_token_idx):
                if beam_complete[beam_idx]: # If the beam has already generated the <eos> token, skip
                    continue
                if token_idx == self.args.eos_token_id:
                    final_beam_scores[beam_idx] = current_beam_scores[beam_idx] # Save the sequence score
                    current_beam_scores[beam_idx] = -float('inf') # Set the score of the <eos> token to -inf so that it is not generated again
                    final_beam_seqs[beam_idx, :decoder_input.size(1)] = decoder_input[beam_idx, :] # Save the sequence
                    beam_complete[beam_idx] = True

            # If all the sequences have generated the <eos> token, break
            if beam_complete.all():
                break

        # If there are no completed sequences, save current sequences
        if not beam_complete.any():
            final_beam_seqs = decoder_input
            final_beam_scores = current_beam_scores

        # Beam Length Normalization
        each_seq_len = torch.sum(final_beam_seqs != self.args.pad_token_id, dim=1).float() # (beam_size)
        length_penalty = (((each_seq_len + beam_size) ** self.args.beam_alpha) / ((beam_size +1) ** self.args.beam_alpha))
        final_beam_scores = final_beam_scores / length_penalty

        # Find the best sequence
        best_seq_idx = torch.argmax(final_beam_scores).item()
        best_seq = final_beam_seqs[best_seq_idx, 1:] # Remove the <bos> token

        return best_seq.unsqueeze(0) # (1, max_seq_len - 1) - remove the <bos> token

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask
