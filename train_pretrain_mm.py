"""
Multi-modal MC-AR pretraining (HMC) with peripheral Cross-Attn query (simplified as attention pooling) + TAB-GF inside NeuroLM.
Parallel code prediction: map all modalities' code indices into the shared code block (size = n_embed, e.g., 8192),
and place targets at positions t to predict token at t+num_chans as in original MC-AR.

This is a minimal working script to validate EEG+EOG+ECG+EMG on HMC.
"""
import os
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model_neurolm import NeuroLM
from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig
from model.model import GPTConfig
from downstream_dataset import HMC_EEG_Peripheral_PretrainLoader
from dataset import standard_1020
from pathlib import Path
from utils import cosine_scheduler
from collections import OrderedDict


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    backend = 'nccl'
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def load_vq_align(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                        bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                        bias=False, dropout=0., num_classes=0, in_chans=128)
    enc_args_ckpt = checkpoint['encoder_args']
    for k in ['n_layer','n_head','n_embd','block_size','bias']:
        encoder_args[k] = enc_args_ckpt[k]
    dec_args_ckpt = checkpoint['decoder_args']
    for k in ['n_layer','n_head','n_embd','block_size','bias']:
        decoder_args[k] = dec_args_ckpt[k]
    enc_conf = NTConfig(**encoder_args)
    dec_conf = NTConfig(**decoder_args)
    model = VQ_Align(enc_conf, dec_conf).to(device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/NeuroLM-MM')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # text data loader
    data_dir = os.path.join(args.out_dir, 'text')
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # sample Python ints, not torch tensors
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,)).tolist()
        x_list = []
        y_list = []
        for i in ix:
            i = int(i)
            # robust conversion: force copy to standard ndarray then build tensor
            x_np = np.array(data[i:i + args.block_size], dtype=np.int64, copy=True)
            y_np = np.array(data[i + 1:i + 1 + args.block_size], dtype=np.int64, copy=True)
            x_list.append(torch.tensor(x_np, dtype=torch.long))
            y_list.append(torch.tensor(y_np, dtype=torch.long))
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # dataset (EEG + Peripheral)
    eeg_root = Path(args.dataset_dir_eeg)
    eog_root = Path(args.dataset_dir_eog)
    ecg_root = Path(args.dataset_dir_ecg)
    emg_root = Path(args.dataset_dir_emg)
    eeg_train = [f.name for f in Path(eeg_root, 'train').glob('*.pkl')]
    eeg_eval = [f.name for f in Path(eeg_root, 'eval').glob('*.pkl')] if Path(eeg_root, 'eval').exists() else [f.name for f in Path(eeg_root, 'val').glob('*.pkl')]
    eog_train = [f.name for f in Path(eog_root, 'train').glob('*.pkl')] if eog_root.exists() else []
    eog_eval = [f.name for f in Path(eog_root, 'eval').glob('*.pkl')] if Path(eog_root, 'eval').exists() else ([f.name for f in Path(eog_root, 'val').glob('*.pkl')] if eog_root.exists() else [])
    ecg_train = [f.name for f in Path(ecg_root, 'train').glob('*.pkl')]
    ecg_eval = [f.name for f in Path(ecg_root, 'eval').glob('*.pkl')] if Path(ecg_root, 'eval').exists() else [f.name for f in Path(ecg_root, 'val').glob('*.pkl')]
    emg_train = [f.name for f in Path(emg_root, 'train').glob('*.pkl')] if emg_root.exists() else []
    emg_eval = [f.name for f in Path(emg_root, 'eval').glob('*.pkl')] if Path(emg_root, 'eval').exists() else ([f.name for f in Path(emg_root, 'val').glob('*.pkl')] if emg_root.exists() else [])
    dataset_train = HMC_EEG_Peripheral_PretrainLoader(
        str(Path(eeg_root, 'train')), eeg_train,
        str(Path(eog_root, 'train')) if eog_train else None, eog_train if eog_train else None,
        str(Path(ecg_root, 'train')), ecg_train,
        str(Path(emg_root, 'train')) if emg_train else None, emg_train if emg_train else None,
        block_size=args.block_size
    )
    dataset_val = HMC_EEG_Peripheral_PretrainLoader(
        str(Path(eeg_root, 'eval')), eeg_eval,
        str(Path(eog_root, 'eval')) if eog_eval else None, eog_eval if eog_eval else None,
        str(Path(ecg_root, 'eval')), ecg_eval,
        str(Path(emg_root, 'eval')) if emg_eval else None, emg_eval if emg_eval else None,
        block_size=args.block_size
    )

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
            batch_size=args.eeg_batch_size, num_workers=10, pin_memory=True, drop_last=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val,
            batch_size=int(1.5*args.eeg_batch_size), num_workers=10, pin_memory=True, drop_last=False)
    else:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.eeg_batch_size,
            num_workers=10, pin_memory=True, drop_last=True, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=int(1.5*args.eeg_batch_size),
            num_workers=10, pin_memory=True, drop_last=False, shuffle=False)

    # load VQ for each modality
    vq_eeg = load_vq_align(os.path.join(args.out_dir, args.tokenizer_path_eeg)) if args.tokenizer_path_eeg else None
    vq_ecg = load_vq_align(os.path.join(args.out_dir, args.tokenizer_path_ecg)) if args.tokenizer_path_ecg else None
    vq_eog = load_vq_align(os.path.join(args.out_dir, args.tokenizer_path_eog)) if args.tokenizer_path_eog else None
    vq_emg = load_vq_align(os.path.join(args.out_dir, args.tokenizer_path_emg)) if args.tokenizer_path_emg else None

    # model init (expand LM head/wte for code-vocab)
    n_layer = 12; n_head = 12; n_embd = 768; dropout = 0.0; bias = False
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                      bias=bias, vocab_size=50257, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = NeuroLM(gptconf,
                    os.path.join(args.out_dir, args.tokenizer_path_eeg) if args.tokenizer_path_eeg else None,
                    init_from='gpt2',
                    tokenizer_ckpt_path_ecg=os.path.join(args.out_dir, args.tokenizer_path_ecg) if args.tokenizer_path_ecg else None,
                    tokenizer_ckpt_path_eog=os.path.join(args.out_dir, args.tokenizer_path_eog) if args.tokenizer_path_eog else None,
                    tokenizer_ckpt_path_emg=os.path.join(args.out_dir, args.tokenizer_path_emg) if args.tokenizer_path_emg else None,
                    )
    # enlarge embeddings for code-vocab mapping (shared block)
    # NEW: Handle shared+private codebook layout
    def get_vq_shared_private_size(vq):
        """Return (shared_size, private_size) for a VQ model"""
        try:
            if vq is not None and hasattr(vq.VQ, 'shared_codes') and hasattr(vq.VQ, 'private_codes'):
                return vq.VQ.shared_codes, vq.VQ.private_codes
            else:
                total = int(vq.VQ.get_number_of_tokens()) if vq is not None else 0
                return 0, total  # fallback: treat as all private
        except Exception:
            return 0, 0
    
    # Get shared/private sizes for each modality
    shared_eeg, private_eeg = get_vq_shared_private_size(vq_eeg)
    shared_ecg, private_ecg = get_vq_shared_private_size(vq_ecg)
    shared_eog, private_eog = get_vq_shared_private_size(vq_eog)
    shared_emg, private_emg = get_vq_shared_private_size(vq_emg)
    
    # Verify all modalities use the same shared size (cross-modal consistency)
    shared_sizes = [shared_eeg, shared_ecg, shared_eog, shared_emg]
    shared_sizes = [s for s in shared_sizes if s > 0]
    if len(set(shared_sizes)) > 1:
        print(f"WARNING: Inconsistent shared codebook sizes: {shared_sizes}")
    
    # Vocabulary layout: [GPT_base] + [Shared_once] + [EEG_private] + [ECG_private] + [EOG_private] + [EMG_private]
    global_shared_size = shared_sizes[0] if shared_sizes else 0
    base_text_vocab = 50257
    
    offsets = {
        'SHARED': base_text_vocab,
        'EEG_PRIVATE': base_text_vocab + global_shared_size,
        'ECG_PRIVATE': base_text_vocab + global_shared_size + private_eeg,
        'EOG_PRIVATE': base_text_vocab + global_shared_size + private_eeg + private_ecg,
        'EMG_PRIVATE': base_text_vocab + global_shared_size + private_eeg + private_ecg + private_eog,
    }
    
    code_vocab_size = global_shared_size + private_eeg + private_ecg + private_eog + private_emg
    target_vocab = base_text_vocab + code_vocab_size
    
    print(f"Vocabulary layout: GPT({base_text_vocab}) + Shared({global_shared_size}) + Private(EEG:{private_eeg}, ECG:{private_ecg}, EOG:{private_eog}, EMG:{private_emg})")
    print(f"Total vocabulary size: {target_vocab}")
    model.GPT2.enlarge_wte(target_vocab)
    model.GPT2.enlarge_lm_head(target_vocab)
    model.to(device)
    if ddp:
        try:
            model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True, static_graph=True)
        except TypeError:
            model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    scaler = torch.amp.GradScaler(device_type, enabled=(dtype=='float16'))
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    num_training_steps_per_epoch = max(1, len(dataset_train) // args.eeg_batch_size // max(1, ddp_world_size))
    lr_schedule_values = cosine_scheduler(args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
                                          warmup_epochs=args.warmup_epochs)

    # indices
    IDX_ECG = standard_1020.index('ECG') if 'ECG' in standard_1020 else -1
    IDX_E1 = standard_1020.index('E1') if 'E1' in standard_1020 else -1
    IDX_E2 = standard_1020.index('E2') if 'E2' in standard_1020 else -1
    IDX_EMG = standard_1020.index('EMG') if 'EMG' in standard_1020 else -1
    IDX_PAD = standard_1020.index('pad') if 'pad' in standard_1020 else -1

    def get_codes_for_sample(X, input_chans, input_time, input_mask):
        """Return code indices for full token sequence by calling modality-specific VQ.
        X: [N_tokens, 200] (float)
        input_chans/input_time: [N_tokens]
        """
        N = X.size(0)
        code_idx = torch.full((1, N), fill_value=-1, device=X.device, dtype=torch.long)
        # helper: slice positions
        def fill_codes(idx_list, vq):
            if vq is None or len(idx_list)==0:
                return
            pos = torch.tensor(idx_list, device=X.device, dtype=torch.long)
            X_mod = X[pos].unsqueeze(0)  # [1, M, 200]
            chans_mod = input_chans[pos].unsqueeze(0)
            time_mod = input_time[pos].unsqueeze(0)
            mask_mod = torch.ones((1, X_mod.size(1)), device=X.device).bool()
            codes = vq.VQ.get_codebook_indices(X_mod, chans_mod, time_mod, mask_mod)
            code_idx[0, pos] = codes[0].to(code_idx.dtype)
        # EEG: everything except peripheral/pad
        eeg_pos = (input_chans != IDX_ECG) & (input_chans != IDX_E1) & (input_chans != IDX_E2) & (input_chans != IDX_EMG) & (input_chans != IDX_PAD)
        ecg_pos = (input_chans == IDX_ECG)
        eog_pos = (input_chans == IDX_E1) | (input_chans == IDX_E2)
        emg_pos = (input_chans == IDX_EMG)
        fill_codes(torch.nonzero(eeg_pos).squeeze(-1).tolist(), vq_eeg)
        fill_codes(torch.nonzero(ecg_pos).squeeze(-1).tolist(), vq_ecg)
        fill_codes(torch.nonzero(eog_pos).squeeze(-1).tolist(), vq_eog)
        fill_codes(torch.nonzero(emg_pos).squeeze(-1).tolist(), vq_emg)
        # NEW: Map indices to shared+private layout
        valid = code_idx >= 0
        for pos_idx in torch.nonzero(valid).squeeze(-1):
            b, n = divmod(pos_idx.item(), code_idx.size(1))
            original_idx = code_idx[b, n].item()
            
            # Determine modality
            if eeg_pos[n]:
                modality = 'EEG'
            elif ecg_pos[n]:
                modality = 'ECG'
            elif eog_pos[n]:
                modality = 'EOG'
            elif emg_pos[n]:
                modality = 'EMG'
            else:
                continue
                
            # Map to shared or private space
            if global_shared_size > 0 and original_idx < global_shared_size:
                # Shared code: map to global shared space
                code_idx[b, n] = offsets['SHARED'] + original_idx
            else:
                # Private code: map to modality-specific private space
                private_idx = original_idx - global_shared_size if global_shared_size > 0 else original_idx
                code_idx[b, n] = offsets[f'{modality}_PRIVATE'] + private_idx
        return code_idx

    iter_num = 0
    X_text2, Y_text2 = get_batch('train')
    for epoch in range(args.epochs):
        for step, batch in enumerate(data_loader_train):
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for pg in optimizer.param_groups: pg['lr'] = lr

            X, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens, eeg_chans = batch
            X = X.float().to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)
            gpt_mask = gpt_mask.to(device, non_blocking=True)

            # build targets (shared code block), autoregressive shift by num_chans
            with torch.no_grad():
                Y_codes = torch.full((X.size(0), X.size(1)), fill_value=-1, device=device, dtype=torch.long)
                for b in range(X.size(0)):
                    codes = get_codes_for_sample(X[b], input_chans[b], input_time[b], input_mask[b])  # [1, N]
                    N = X.size(1)
                    # predict next time step token per channel
                    nc = int(num_chans[b].item()) if isinstance(num_chans, torch.Tensor) else int(num_chans)
                    Y_codes[b, :N-nc] = codes[0, nc:]
                # sanitize targets to valid range
                max_allowed = raw_model.GPT2.config.vocab_size - 1
                Y_codes[(Y_codes < -1) | (Y_codes > max_allowed)] = -1
                # do not predict base text vocab in this pass
                Y_codes[(Y_codes >= 0) & (Y_codes < base_text_vocab)] = -1
                if master_process and step == 0:
                    n_classes = raw_model.GPT2.lm_head.weight.shape[0]
                    valid = Y_codes >= 0
                    if valid.any():
                        print('debug target range:', int(Y_codes[valid].min().item()), int(Y_codes[valid].max().item()), 'n_classes', n_classes)

            if ddp:
                model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

            with ctx:
                # eeg forward with targets; text forward for regularization
                # missing-modality simulation: randomly mask one peripheral modality tokens
                if args.mm_drop_prob > 0:
                    drop_mask = torch.rand_like(input_chans.float()) < args.mm_drop_prob
                    Y_codes[drop_mask] = -1
                # forward to obtain logits; ignore built-in loss and compute per-modality weighted loss
                _, _, logits = model(X, Y_codes, None, None, input_chans, input_time, input_mask, eeg_mask=gpt_mask)
                V = logits.size(-1)
                logits_flat = logits.view(-1, V)
                targets_flat = Y_codes.view(-1)
                # modality masks
                B, N = input_chans.size()
                chans_flat = input_chans.view(-1)
                valid = targets_flat.ge(0)
                eeg_mask_f = valid & (chans_flat != IDX_ECG) & (chans_flat != IDX_E1) & (chans_flat != IDX_E2) & (chans_flat != IDX_EMG) & (chans_flat != IDX_PAD)
                ecg_mask_f = valid & (chans_flat == IDX_ECG)
                eog_mask_f = valid & ((chans_flat == IDX_E1) | (chans_flat == IDX_E2))
                emg_mask_f = valid & (chans_flat == IDX_EMG)
                def masked_ce(mask):
                    if mask.any():
                        return torch.nn.functional.cross_entropy(logits_flat[mask], targets_flat[mask], reduction='mean')
                    return logits_flat.new_tensor(0.0)
                loss_eeg = masked_ce(eeg_mask_f)
                loss_ecg = masked_ce(ecg_mask_f)
                loss_eog = masked_ce(eog_mask_f)
                loss_emg = masked_ce(emg_mask_f)
                # weights
                w_eeg, w_ecg, w_eog, w_emg = args.w_eeg, args.w_ecg, args.w_eog, args.w_emg
                loss_mm = (w_eeg*loss_eeg + w_ecg*loss_ecg + w_eog*loss_eog + w_emg*loss_emg)
                # text loss
                loss2, log2, _ = model(None, None, X_text2, Y_text2)
                loss = (loss_mm + loss2) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            X_text2, Y_text2 = get_batch('train')

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                # logging breakdown
                ce_vals = {
                    'loss_eeg': float(loss_eeg.detach().item()) if torch.is_tensor(loss_eeg) else float(loss_eeg),
                    'loss_ecg': float(loss_ecg.detach().item()) if torch.is_tensor(loss_ecg) else float(loss_ecg),
                    'loss_eog': float(loss_eog.detach().item()) if torch.is_tensor(loss_eog) else float(loss_eog),
                    'loss_emg': float(loss_emg.detach().item()) if torch.is_tensor(loss_emg) else float(loss_emg),
                }
                total_loss_val = ce_vals['loss_eeg']*w_eeg + ce_vals['loss_ecg']*w_ecg + ce_vals['loss_eog']*w_eog + ce_vals['loss_emg']*w_emg + float(log2['train/loss'])
                print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: total {total_loss_val:.4f} mm(eeg/ecg/eog/emg) {ce_vals['loss_eeg']:.4f}/{ce_vals['loss_ecg']:.4f}/{ce_vals['loss_eog']:.4f}/{ce_vals['loss_emg']:.4f} text {log2['train/loss']:.4f}")
            iter_num += 1

        # simple consistency proxy on val: compute InfoNCE over peripheral vs EEG representations (TODO: true features hook)
        # Skipped here for brevity; place-holder for future commit.

        # save
        if master_process:
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(ckpt, os.path.join(checkpoint_out_dir, 'ckpt.pt'))

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser('MM MC-AR pretrain', add_help=False)
    parser.add_argument('--out_dir', default='./', help='output root')
    parser.add_argument('--dataset_dir_eeg', default='/root/autodl-tmp/NeuroLM/NeuroLM_fix/HMC')
    parser.add_argument('--dataset_dir_ecg', default='/root/autodl-tmp/NeuroLM/NeuroLM_fix/HMC_ECG')
    parser.add_argument('--dataset_dir_eog', default='/root/autodl-tmp/NeuroLM/NeuroLM_fix/HMC_EOG')
    parser.add_argument('--dataset_dir_emg', default='/root/autodl-tmp/NeuroLM/NeuroLM_fix/HMC_EMG')
    parser.add_argument('--tokenizer_path_eeg', default='checkpoints/VQ_EEG.pt')
    parser.add_argument('--tokenizer_path_ecg', default='checkpoints/VQ_ECG.pt')
    parser.add_argument('--tokenizer_path_eog', default='checkpoints/VQ_EOG.pt')
    parser.add_argument('--tokenizer_path_emg', default='checkpoints/VQ_EMG.pt')
    parser.add_argument('--log_interval', default=10, type=int)
    # training
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=8, type=int)
    parser.add_argument('--text_batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--block_size', default=1024, type=int)
    parser.add_argument('--code_vocab_eeg', default=8192, type=int)
    parser.add_argument('--code_vocab_ecg', default=8192, type=int)
    parser.add_argument('--code_vocab_eog', default=4096, type=int)
    parser.add_argument('--code_vocab_emg', default=4096, type=int)
    parser.add_argument('--mm_drop_prob', default=0.0, type=float, help='random drop probability for a token to simulate missing modality')
    parser.add_argument('--w_eeg', type=float, default=1.0)
    parser.add_argument('--w_ecg', type=float, default=1.0)
    parser.add_argument('--w_eog', type=float, default=1.0)
    parser.add_argument('--w_emg', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)


