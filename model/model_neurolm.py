"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.model import *
from dataset import standard_1020
from torch.autograd import Function
from model.model_neural_transformer import NTConfig
from model.model_neural_transformer import NeuralTransformer
from collections import OrderedDict


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class NeuroLM(nn.Module):

    def __init__(self,
                 GPT_config,
                 tokenizer_ckpt_path=None,
                 init_from='gpt2',
                 n_embd=768,
                 eeg_vocab_size=8192,
                 tokenizer_ckpt_path_ecg=None,
                 tokenizer_ckpt_path_eog=None,
                 tokenizer_ckpt_path_emg=None,
                 ):
        super().__init__()

        if init_from == 'scratch':
            self.GPT2 = GPT(GPT_config)
        elif init_from.startswith('gpt2'):
            override_args = dict(dropout=0.0)
            self.GPT2 = GPT.from_pretrained(init_from, override_args)
        self.GPT2.enlarge_wte(50304)
        self.GPT2.enlarge_lm_head(self.GPT2.config.vocab_size + eeg_vocab_size)

        if tokenizer_ckpt_path is not None:
            print('loading weight from VQ_align')
            encoder_args = dict(n_layer=12, n_head=10, n_embd=400, block_size=1024,
                                bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
            tokenizer_checkpoint = torch.load(tokenizer_ckpt_path)
            tokenizer_checkpoint_model_args = tokenizer_checkpoint['encoder_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                encoder_args[k] = tokenizer_checkpoint_model_args[k]
            tokenizer_checkpoint_model_args = tokenizer_checkpoint['decoder_args']
            # create the model
            encoder_conf = NTConfig(**encoder_args)
            self.tokenizer = NeuralTransformer(encoder_conf)
            tokenizer_state_dict = tokenizer_checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(tokenizer_state_dict.items()):
                if k.startswith(unwanted_prefix):
                    tokenizer_state_dict[k[len(unwanted_prefix):]] = tokenizer_state_dict.pop(k)
            
            all_keys = list(tokenizer_state_dict.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('VQ.encoder.'):
                    new_dict[key[11:]] = tokenizer_state_dict[key]
            self.tokenizer.load_state_dict(new_dict)
            # optional ECG tokenizer
            if tokenizer_ckpt_path_ecg is not None:
                print('loading ECG tokenizer from VQ_align')
                tokenizer_checkpoint_ecg = torch.load(tokenizer_ckpt_path_ecg)
                tokenizer_checkpoint_model_args = tokenizer_checkpoint_ecg['encoder_args']
                for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                    encoder_args[k] = tokenizer_checkpoint_model_args[k]
                encoder_conf_ecg = NTConfig(**encoder_args)
                self.tokenizer_ecg = NeuralTransformer(encoder_conf_ecg)
                tokenizer_state_dict_ecg = tokenizer_checkpoint_ecg['model']
                unwanted_prefix = '_orig_mod.'
                for k,v in list(tokenizer_state_dict_ecg.items()):
                    if k.startswith(unwanted_prefix):
                        tokenizer_state_dict_ecg[k[len(unwanted_prefix):]] = tokenizer_state_dict_ecg.pop(k)
                all_keys_ecg = list(tokenizer_state_dict_ecg.keys())
                new_dict_ecg = OrderedDict()
                for key in all_keys_ecg:
                    if key.startswith('VQ.encoder.'):
                        new_dict_ecg[key[11:]] = tokenizer_state_dict_ecg[key]
                self.tokenizer_ecg.load_state_dict(new_dict_ecg)
            # optional EOG tokenizer
            if tokenizer_ckpt_path_eog is not None:
                print('loading EOG tokenizer from VQ_align')
                tokenizer_checkpoint_eog = torch.load(tokenizer_ckpt_path_eog)
                tokenizer_checkpoint_model_args = tokenizer_checkpoint_eog['encoder_args']
                for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                    encoder_args[k] = tokenizer_checkpoint_model_args[k]
                encoder_conf_eog = NTConfig(**encoder_args)
                self.tokenizer_eog = NeuralTransformer(encoder_conf_eog)
                tokenizer_state_dict_eog = tokenizer_checkpoint_eog['model']
                unwanted_prefix = '_orig_mod.'
                for k,v in list(tokenizer_state_dict_eog.items()):
                    if k.startswith(unwanted_prefix):
                        tokenizer_state_dict_eog[k[len(unwanted_prefix):]] = tokenizer_state_dict_eog.pop(k)
                all_keys_eog = list(tokenizer_state_dict_eog.keys())
                new_dict_eog = OrderedDict()
                for key in all_keys_eog:
                    if key.startswith('VQ.encoder.'):
                        new_dict_eog[key[11:]] = tokenizer_state_dict_eog[key]
                self.tokenizer_eog.load_state_dict(new_dict_eog)
            # optional EMG tokenizer
            if tokenizer_ckpt_path_emg is not None:
                print('loading EMG tokenizer from VQ_align')
                tokenizer_checkpoint_emg = torch.load(tokenizer_ckpt_path_emg)
                tokenizer_checkpoint_model_args = tokenizer_checkpoint_emg['encoder_args']
                for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                    encoder_args[k] = tokenizer_checkpoint_model_args[k]
                encoder_conf_emg = NTConfig(**encoder_args)
                self.tokenizer_emg = NeuralTransformer(encoder_conf_emg)
                tokenizer_state_dict_emg = tokenizer_checkpoint_emg['model']
                unwanted_prefix = '_orig_mod.'
                for k,v in list(tokenizer_state_dict_emg.items()):
                    if k.startswith(unwanted_prefix):
                        tokenizer_state_dict_emg[k[len(unwanted_prefix):]] = tokenizer_state_dict_emg.pop(k)
                all_keys_emg = list(tokenizer_state_dict_emg.keys())
                new_dict_emg = OrderedDict()
                for key in all_keys_emg:
                    if key.startswith('VQ.encoder.'):
                        new_dict_emg[key[11:]] = tokenizer_state_dict_emg[key]
                self.tokenizer_emg.load_state_dict(new_dict_emg)
        else:
            encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                                bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
            encoder_conf = NTConfig(**encoder_args)
            self.tokenizer = NeuralTransformer(encoder_conf)
            self.tokenizer_ecg = None
            self.tokenizer_eog = None
            self.tokenizer_emg = None
        
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        if hasattr(self, 'tokenizer_ecg') and self.tokenizer_ecg is not None:
            for p in self.tokenizer_ecg.parameters():
                p.requires_grad = False
        if hasattr(self, 'tokenizer_eog') and self.tokenizer_eog is not None:
            for p in self.tokenizer_eog.parameters():
                p.requires_grad = False
        if hasattr(self, 'tokenizer_emg') and self.tokenizer_emg is not None:
            for p in self.tokenizer_emg.parameters():
                p.requires_grad = False

        self.pos_embed = nn.Embedding(256, self.GPT2.config.n_embd)
        # modality embedding: 0=EEG, 1=ECG, 2=EOG, 3=EMG, 4=PAD
        self.modality_embed = nn.Embedding(5, self.GPT2.config.n_embd)

        # optional gated fusion (EEG<->ECG) before LLM
        self.enable_gated_fusion = True
        self.body_to_eeg = nn.Linear(self.GPT2.config.n_embd, self.GPT2.config.n_embd)
        self.eeg_to_body = nn.Linear(self.GPT2.config.n_embd, self.GPT2.config.n_embd)
        gate_hidden = max(32, self.GPT2.config.n_embd // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.GPT2.config.n_embd * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        # Cross-Attn based fusion (replaces mean pooling):
        # - external multi-query aggregation for peripheral tokens
        # - bidirectional EEG<->PERIPH cross attention per time step
        self.num_periph_queries = 4
        self.periph_queries = nn.Parameter(torch.randn(self.num_periph_queries, self.GPT2.config.n_embd))
        self.periph_agg_attn = nn.MultiheadAttention(embed_dim=self.GPT2.config.n_embd, num_heads=max(1, self.GPT2.config.n_head // 2), batch_first=True)
        self.eeg_from_body_attn = nn.MultiheadAttention(embed_dim=self.GPT2.config.n_embd, num_heads=self.GPT2.config.n_head, batch_first=True)
        self.body_from_eeg_attn = nn.MultiheadAttention(embed_dim=self.GPT2.config.n_embd, num_heads=self.GPT2.config.n_head, batch_first=True)
        # task-specific gate bias (per-task learnable parameter), 16 tasks by default
        self.task_gate_bias = nn.Embedding(16, 1)
        nn.init.zeros_(self.task_gate_bias.weight)

        # task layer
        self.encode_transform_layer = nn.Sequential(
            nn.Linear(n_embd, self.GPT2.config.n_embd),
            nn.GELU(),
        ) if n_embd != self.GPT2.config.n_embd else nn.Identity()

        self.encode_transform_layer.apply(self._init_weights)
        self.body_to_eeg.apply(self._init_weights)
        self.eeg_to_body.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_eeg=None, y_eeg=None, x_text=None, y_text=None, input_chans=None, input_time=None, input_mask=None, eeg_mask=None, eeg_text_mask=None, task_ids=None):
        """
        x_eeg: shape [B, N1, T]
        x_text: shape [B, N2]
        """
        if x_eeg is not None:
            input_mask_exp = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
            # if we have a dedicated ECG tokenizer, split tokens by channel type and run separately
            if hasattr(self, 'tokenizer_ecg') and self.tokenizer_ecg is not None:
                # indices in standard_1020
                ecg_idx = standard_1020.index('ECG') if 'ECG' in standard_1020 else -1
                e1_idx = standard_1020.index('E1') if 'E1' in standard_1020 else -1
                e2_idx = standard_1020.index('E2') if 'E2' in standard_1020 else -1
                emg_idx = standard_1020.index('EMG') if 'EMG' in standard_1020 else -1
                pad_idx = standard_1020.index('pad') if 'pad' in standard_1020 else -1
                B, N, T = x_eeg.shape
                device = x_eeg.device
                hidden = self.GPT2.config.n_embd
                encoded = torch.zeros((B, N, hidden), device=device)
                # build per-batch masks
                eeg_token_mask = (input_chans != ecg_idx) & (input_chans != e1_idx) & (input_chans != e2_idx) & (input_chans != emg_idx) & (input_chans != pad_idx)
                ecg_token_mask = (input_chans == ecg_idx)
                eog_token_mask = (input_chans == e1_idx) | (input_chans == e2_idx)
                emg_token_mask = (input_chans == emg_idx)
                # process EEG slice
                for b in range(B):
                    idx_eeg = torch.nonzero(eeg_token_mask[b]).squeeze(-1)
                    if idx_eeg.numel() > 0:
                        x_part = x_eeg[b:b+1, idx_eeg, :]
                        ch_part = input_chans[b:b+1, idx_eeg]
                        time_part = input_time[b:b+1, idx_eeg]
                        mask_part = input_mask_exp[b:b+1, :, idx_eeg, :]
                        mask_part = mask_part[:, :, :, idx_eeg]
                        feats = self.tokenizer(x_part, ch_part, time_part, mask_part, return_all_tokens=True)
                        feats = self.encode_transform_layer(feats)
                        feats += self.pos_embed(ch_part)
                        feats += self.modality_embed(torch.zeros_like(ch_part))
                        encoded[b, idx_eeg] = feats[0]
                    idx_ecg = torch.nonzero(ecg_token_mask[b]).squeeze(-1)
                    if idx_ecg.numel() > 0:
                        x_part = x_eeg[b:b+1, idx_ecg, :]
                        ch_part = input_chans[b:b+1, idx_ecg]
                        time_part = input_time[b:b+1, idx_ecg]
                        mask_part = input_mask_exp[b:b+1, :, idx_ecg, :]
                        mask_part = mask_part[:, :, :, idx_ecg]
                        feats = self.tokenizer_ecg(x_part, ch_part, time_part, mask_part, return_all_tokens=True)
                        feats = self.encode_transform_layer(feats)
                        feats += self.pos_embed(ch_part)
                        feats += self.modality_embed(torch.ones_like(ch_part))
                        encoded[b, idx_ecg] = feats[0]
                    if hasattr(self, 'tokenizer_eog') and self.tokenizer_eog is not None:
                        idx_eog = torch.nonzero(eog_token_mask[b]).squeeze(-1)
                        if idx_eog.numel() > 0:
                            x_part = x_eeg[b:b+1, idx_eog, :]
                            ch_part = input_chans[b:b+1, idx_eog]
                            time_part = input_time[b:b+1, idx_eog]
                            mask_part = input_mask_exp[b:b+1, :, idx_eog, :]
                            mask_part = mask_part[:, :, :, idx_eog]
                            feats = self.tokenizer_eog(x_part, ch_part, time_part, mask_part, return_all_tokens=True)
                            feats = self.encode_transform_layer(feats)
                            feats += self.pos_embed(ch_part)
                            feats += self.modality_embed(torch.full_like(ch_part, 2))
                            encoded[b, idx_eog] = feats[0]
                    if hasattr(self, 'tokenizer_emg') and self.tokenizer_emg is not None:
                        idx_emg = torch.nonzero(emg_token_mask[b]).squeeze(-1)
                        if idx_emg.numel() > 0:
                            x_part = x_eeg[b:b+1, idx_emg, :]
                            ch_part = input_chans[b:b+1, idx_emg]
                            time_part = input_time[b:b+1, idx_emg]
                            mask_part = input_mask_exp[b:b+1, :, idx_emg, :]
                            mask_part = mask_part[:, :, :, idx_emg]
                            feats = self.tokenizer_emg(x_part, ch_part, time_part, mask_part, return_all_tokens=True)
                            feats = self.encode_transform_layer(feats)
                            feats += self.pos_embed(ch_part)
                            feats += self.modality_embed(torch.full_like(ch_part, 3))
                            encoded[b, idx_emg] = feats[0]
                x_eeg = encoded
            else:
                x_eeg = self.tokenizer(x_eeg, input_chans, input_time, input_mask_exp, return_all_tokens=True)
                x_eeg = self.encode_transform_layer(x_eeg)
                x_eeg += self.pos_embed(input_chans)
                # mark all as EEG modality
                x_eeg += self.modality_embed(torch.zeros_like(input_chans))

            # gated fusion across modalities per time step (Cross-Attn formulation)
            if self.enable_gated_fusion and hasattr(self, 'tokenizer_ecg') and self.tokenizer_ecg is not None:
                ecg_idx_val = standard_1020.index('ECG') if 'ECG' in standard_1020 else -1
                e1_idx_val = standard_1020.index('E1') if 'E1' in standard_1020 else -1
                e2_idx_val = standard_1020.index('E2') if 'E2' in standard_1020 else -1
                emg_idx_val = standard_1020.index('EMG') if 'EMG' in standard_1020 else -1
                pad_idx_val = standard_1020.index('pad') if 'pad' in standard_1020 else -1
                B = x_eeg.size(0)
                # vectorized over time to avoid per-time reentrant use in DDP; compute per token masks
                is_eeg_tok = (input_chans != ecg_idx_val) & (input_chans != e1_idx_val) & (input_chans != e2_idx_val) & (input_chans != emg_idx_val) & (input_chans != pad_idx_val)
                is_periph_tok = ~is_eeg_tok & (input_chans != pad_idx_val)
                B, N, D = x_eeg.shape
                for b in range(B):
                    t_vals = input_time[b].unique()
                    for t in t_vals:
                        mask_time = (input_time[b] == t)
                        eeg_idx = torch.nonzero(mask_time & is_eeg_tok[b]).squeeze(-1)
                        per_idx = torch.nonzero(mask_time & is_periph_tok[b]).squeeze(-1)
                        if eeg_idx.numel()==0 or per_idx.numel()==0:
                            continue
                        eeg_feats = x_eeg[b, eeg_idx].unsqueeze(0)    # [1, Ne, D]
                        per_feats = x_eeg[b, per_idx].unsqueeze(0)    # [1, Np, D]
                        # 1) peripheral multi-query aggregation
                        queries = self.periph_queries.unsqueeze(0).expand(1, self.num_periph_queries, D)  # [1, Q, D]
                        body_q, _ = self.periph_agg_attn(query=queries, key=per_feats, value=per_feats, need_weights=False)
                        body_ctx = body_q.mean(dim=1, keepdim=True)  # [1,1,D]
                        # 2) bidirectional cross attention
                        eeg_enh, _ = self.eeg_from_body_attn(query=eeg_feats, key=per_feats, value=per_feats, need_weights=False)
                        body_enh, _ = self.body_from_eeg_attn(query=per_feats, key=eeg_feats, value=eeg_feats, need_weights=False)
                        # 3) scalar gate from pooled EEG/Body (retain scalar for stability)
                        eeg_mean = eeg_feats.mean(dim=1)      # [1,D]
                        body_mean = body_ctx.squeeze(1)       # [1,D]
                        g = self.gate_mlp(torch.cat([eeg_mean, body_mean], dim=-1))  # [1,1]
                        if task_ids is not None:
                            bias = self.task_gate_bias(task_ids[b].clamp(min=0, max=self.task_gate_bias.num_embeddings-1))
                            g = torch.sigmoid(torch.logit(g.clamp(1e-6, 1-1e-6)) + bias)
                        # 4) gated residual injection (token-wise)
                        x_eeg[b, eeg_idx] = x_eeg[b, eeg_idx] + g * eeg_enh.squeeze(0)
                        x_eeg[b, per_idx] = x_eeg[b, per_idx] + (1.0 - g) * body_enh.squeeze(0)

        logits, loss, accuracy = self.GPT2(x_eeg, y_eeg, x_text, y_text, input_time, eeg_mask, eeg_text_mask)

        log = {}
        split="train" if self.training else "val"
        if loss is not None:
            log[f'{split}/loss'] = loss.item()
        if accuracy is not None:
            log[f'{split}/accuracy'] = accuracy.item()

        return loss, log, logits
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.GPT2.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, x_eeg, x_text, input_chans, input_time, input_mask, eeg_mask=None, eeg_text_mask=None, max_new_tokens=10, temperature=1.0, top_k=1):
        if x_eeg is not None:
            input_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
            x_eeg = self.tokenizer(x_eeg, input_chans, input_time, input_mask, return_all_tokens=True)
            x_eeg = self.encode_transform_layer(x_eeg)
            x_eeg += self.pos_embed(input_chans)
            #input_time = torch.zeros((x_eeg.size(0), x_eeg.size(1)), device=x_eeg.device).int()
        
        for _ in range(max_new_tokens):
            logits, _, _ = self.GPT2(x_eeg=x_eeg, x_text=x_text, eeg_time_idx=input_time, eeg_mask=eeg_mask, eeg_text_mask=eeg_text_mask)
            logits = logits[:, -1, :50257] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #_, idx_next = logits.max(-1)

            x_text = torch.cat((x_text, idx_next), dim=1)
            if eeg_text_mask is not None:
                eeg_text_mask = torch.cat((eeg_text_mask, torch.zeros((eeg_text_mask.size(0), eeg_text_mask.size(1), eeg_text_mask.size(2), 1), device=eeg_text_mask.device)), dim=-1)
                eeg_text_mask = torch.cat((eeg_text_mask, torch.ones((eeg_text_mask.size(0), eeg_text_mask.size(1), 1, eeg_text_mask.size(3)), device=eeg_text_mask.device)), dim=-2)
        
        return x_text
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
