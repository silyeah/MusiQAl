from __future__ import print_function
import sys
sys.path.append("./AVST")
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader_avst import *
from net_avst import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb
import os
import contextlib
import random
import secrets
import csv
import math
# from .net_avst import AVQA_Fusion_Net

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/net_avst/'+TIMESTAMP)

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")


class Tee:
    """Simple tee for stdout: write to multiple file-like objects."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

def batch_organize(out_match_posi,out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    # print("audio data: ", audio_data.shape)
    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):
        audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

        optimizer.zero_grad()
        out_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)  
        out_match,match_label=batch_organize(out_match_posi,out_match_nega)  
        out_match,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()
    
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        loss_match=criterion(out_match,match_label)
        loss_qa = criterion(out_qa, target)
        loss = loss_qa + 0.5*loss_match

        writer.add_scalar('run/match',loss_match.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/qa_test',loss_qa.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/both',loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

            preds_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)

            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


# Helper functions
def _safe_pct(lst):
    """Calculate percentage from a list of 0/1 results."""
    return (100.0 * sum(lst) / len(lst)) if len(lst) > 0 else 0.0

def _safe_pct_ratio(numer, denom):
    """Calculate percentage from numerator/denominator, returns 0 if denom is 0."""
    return (100.0 * numer / denom) if denom > 0 else 0.0

def _assign_targets(orig_vids, unique_vids, rng, cap=4, banned=None, partner_targets=None, label=None):
    """Assign each sample to a different video target, max 'cap' uses per target."""
    n = len(orig_vids)
    uniq = list(unique_vids)
    
    for _ in range(2000):  # max tries
        used = {vid: 0 for vid in uniq}
        avail = {vid: cap for vid in uniq}
        targets = [None] * n
        remaining = list(range(n))
        rng.shuffle(remaining)
        
        progress = True
        while remaining and progress:
            progress = False
            new_remaining = []
            for i in remaining:
                bans = banned[i] if banned is not None else set()
                cands = [vid for vid in uniq
                         if vid != orig_vids[i]
                         and avail.get(vid, 0) > 0
                         and vid not in bans
                         and (partner_targets is None or vid != partner_targets[i])]
                if not cands:
                    new_remaining.append(i)
                    continue
                min_use = min(used[vid] for vid in cands)
                lvl_cands = [vid for vid in cands if used[vid] == min_use]
                t = rng.choice(lvl_cands)
                targets[i] = t
                used[t] += 1
                avail[t] -= 1
                progress = True
            remaining = new_remaining
        
        if all(t is not None for t in targets):
            # Print usage stats
            usage_vals = [used[v] for v in uniq]
            hist = {}
            for u in usage_vals:
                hist[u] = hist.get(u, 0) + 1
            tag = f" [{label}]" if label else ""
            print(f"Target usage fairness{tag}: min={min(usage_vals)}, max={max(usage_vals)}, histogram={{{', '.join(f'{k}:{v}' for k,v in sorted(hist.items()))}}}")
            return targets
    
    raise RuntimeError("Failed to assign balanced targets; relax cap or check constraints")

def _print_accuracy_breakdown(buckets, A_correct, A_total, V_correct, V_total, AV_correct, AV_total, overall_acc):
    """Print accuracy breakdown to console."""
    print('Audio Existential Accuracy: %.2f %%' % _safe_pct(buckets['A_ext']))
    print('Audio Counting Accuracy: %.2f %%' % _safe_pct(buckets['A_count']))
    print('Audio Cmp Accuracy: %.2f %%' % _safe_pct(buckets['A_cmp']))
    print('Audio Temp Accuracy: %.2f %%' % _safe_pct(buckets['A_temp']))
    print('Audio Caus Accuracy: %.2f %%' % _safe_pct(buckets['A_caus']))
    print('Audio Accuracy: %.2f %%' % _safe_pct_ratio(A_correct, A_total))
    print('Visual Ext Accuracy: %.2f %%' % _safe_pct(buckets['V_ext']))
    print('Visual Loc Accuracy: %.2f %%' % _safe_pct(buckets['V_loc']))
    print('Visual Counting Accuracy: %.2f %%' % _safe_pct(buckets['V_count']))
    print('Visual Temp Accuracy: %.2f %%' % _safe_pct(buckets['V_temp']))
    print('Visual Caus Accuracy: %.2f %%' % _safe_pct(buckets['V_caus']))
    print('Visual Accuracy: %.2f %%' % _safe_pct_ratio(V_correct, V_total))
    print('AV Ext Accuracy: %.2f %%' % _safe_pct(buckets['AV_ext']))
    print('AV counting Accuracy: %.2f %%' % _safe_pct(buckets['AV_count']))
    print('AV Loc Accuracy: %.2f %%' % _safe_pct(buckets['AV_loc']))
    print('AV Cmp Accuracy: %.2f %%' % _safe_pct(buckets['AV_cmp']))
    print('AV Temporal Accuracy: %.2f %%' % _safe_pct(buckets['AV_temp']))
    print('AV Caus Accuracy: %.2f %%' % _safe_pct(buckets['AV_caus']))
    print('AV Purp Accuracy: %.2f %%' % _safe_pct(buckets['AV_purp']))
    print('AV Accuracy: %.2f %%' % _safe_pct_ratio(AV_correct, AV_total))
    print('Overall Accuracy: %.2f %%' % overall_acc)

def _write_summary_file(out_path, conflict_mode, total, overall_acc, buckets,
                        A_correct, A_total, V_correct, V_total, AV_correct, AV_total,
                        swapped_audio_count, swapped_video_count, distinct_ok_count):
    """Write evaluation summary to file."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        with open(out_path, 'w') as f:
            f.write('Sensory conflict evaluation\n')
            f.write(f'Mode: {conflict_mode}\n')
            f.write(f'Timestamp: {TIMESTAMP}\n')
            f.write(f'Num samples: {total}\n')
            f.write(f'Overall Accuracy: {overall_acc:.2f} %\n')
            if A_total > 0:
                f.write(f'Audio (All) Accuracy: {_safe_pct_ratio(A_correct, A_total):.2f} %  ({A_correct}/{A_total})\n')
            if V_total > 0:
                f.write(f'Visual (All) Accuracy: {_safe_pct_ratio(V_correct, V_total):.2f} %  ({V_correct}/{V_total})\n')
            if AV_total > 0:
                f.write(f'Audio-Visual (All) Accuracy: {_safe_pct_ratio(AV_correct, AV_total):.2f} %  ({AV_correct}/{AV_total})\n')
            if conflict_mode in ('audio','both'):
                pct_a = _safe_pct_ratio(swapped_audio_count, total)
                f.write(f'Verification (audio): swapped different media for {swapped_audio_count}/{total} samples ({pct_a:.2f}%).')
                f.write(' ALL swapped.\n' if swapped_audio_count == total else ' NOT all swapped.\n')
            if conflict_mode in ('video','both'):
                pct_v = _safe_pct_ratio(swapped_video_count, total)
                f.write(f'Verification (video): swapped different media for {swapped_video_count}/{total} samples ({pct_v:.2f}%).')
                f.write(' ALL swapped.\n' if swapped_video_count == total else ' NOT all swapped.\n')
            if conflict_mode == 'both':
                pct_b = _safe_pct_ratio(distinct_ok_count, total)
                f.write(f'Verification (both): distinct audio/video media targets for {distinct_ok_count}/{total} samples ({pct_b:.2f}%).')
                f.write(' ALL distinct.\n' if distinct_ok_count == total else ' NOT all distinct.\n')
            f.write('\nBreakdown:\n')
            for key, label in [('A_ext','Audio Existential'), ('A_count','Audio Counting'), 
                               ('A_cmp','Audio Cmp'), ('A_temp','Audio Temp'), ('A_caus','Audio Caus'),
                               ('V_ext','Visual Ext'), ('V_loc','Visual Loc'), ('V_count','Visual Counting'),
                               ('V_temp','Visual Temp'), ('V_caus','Visual Caus'),
                               ('AV_ext','AV Ext'), ('AV_count','AV counting'), ('AV_loc','AV Loc'),
                               ('AV_cmp','AV Cmp'), ('AV_temp','AV Temporal'), ('AV_caus','AV Caus'), ('AV_purp','AV Purp')]:
                f.write(f'{label} Accuracy: {_safe_pct(buckets[key]):.2f} %\n')
        print(f"Saved summary results to: {out_path}")
    except Exception as e:
        print(f"Warning: failed to write results summary to {out_path}: {e}")


def test(model, val_loader, seed=None, out_path=None, conflict_mode='audio', banned_audio=None, banned_video=None, available_vids=None):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/data/json/avqa-test.json', 'r'))
    
    # Accuracy buckets for each question type
    buckets = {
        'A_ext': [], 'A_count': [], 'A_cmp': [], 'A_temp': [], 'A_caus': [],
        'V_ext': [], 'V_loc': [], 'V_count': [], 'V_temp': [], 'V_caus': [],
        'AV_ext': [], 'AV_count': [], 'AV_loc': [], 'AV_cmp': [], 'AV_temp': [], 'AV_caus': [], 'AV_purp': [],
    }
    csv_rows = []
    
    # Setup for sensory conflict
    ds = val_loader.dataset
    n = len(ds)
    if conflict_mode in ('audio', 'video', 'both') and n < 2:
        raise ValueError("Sensory conflict evaluation requires at least 2 samples")
    
    rng = random.Random(seed) if seed is not None else secrets.SystemRandom()
    media_keys = [ds.samples[i]['video_id'] for i in range(n)]
    unique_vids = sorted(set(media_keys)) if available_vids is None else sorted(set(available_vids))
    cap = min(4, max(1, math.ceil(n / max(1, len(unique_vids)))))  # ~1183/310 ≈ 3.8
    
    if banned_audio is None:
        banned_audio = [set() for _ in range(n)]
    if banned_video is None:
        banned_video = [set() for _ in range(n)]
    
    # Setup target mappings
    audio_targets = media_keys[:]
    video_targets = media_keys[:]
    if conflict_mode == 'both':
        audio_targets = _assign_targets(media_keys, unique_vids, rng, cap=cap, banned=banned_audio, label='audio')
        video_targets = _assign_targets(media_keys, unique_vids, rng, cap=cap, banned=banned_video, partner_targets=audio_targets, label='video')
        same_media = sum(1 for i in range(n) if audio_targets[i] == video_targets[i])
        print(f"Both-mode mapping check: distinct target media for all samples: {'YES' if same_media == 0 else 'NO'} ({n - same_media}/{n}).")
    elif conflict_mode == 'audio':
        audio_targets = _assign_targets(media_keys, unique_vids, rng, cap=cap, banned=banned_audio, label='audio')
    elif conflict_mode == 'video':
        video_targets = _assign_targets(media_keys, unique_vids, rng, cap=cap, banned=banned_video, label='video')
    
    print(f"Sensory conflict mode: {conflict_mode} | rng={'seeded' if seed is not None else 'secure'} | N={n} | targets={len(unique_vids)} unique video_ids")
    ans_list = val_loader.dataset.ans_vocab
    
    # Counters for verification
    swapped_audio_count = 0
    swapped_video_count = 0
    distinct_ok_count = 0
    
    # Main evaluation loop
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio = sample['audio'].to('cuda')
            visual_posi = sample['visual_posi'].to('cuda')
            visual_nega = sample['visual_nega'].to('cuda')
            target = sample['label'].to('cuda')
            question = sample['question'].to('cuda')
            
            orig_key = ds.samples[batch_idx]['video_id']
            a_name, v_name = orig_key, orig_key
            
            # Swap audio if needed
            if conflict_mode in ('audio', 'both'):
                a_name = audio_targets[batch_idx]
                a_np = np.load(os.path.join(ds.audio_dir, a_name + '.npy'))
                audio = torch.from_numpy(a_np[::6, :]).unsqueeze(0).to(audio.device)
            
            # Swap video if needed
            if conflict_mode in ('video', 'both'):
                v_name = video_targets[batch_idx]
                v_np = np.load(os.path.join(ds.video_res14x14_dir, v_name + '.npy'))
                visual_posi = torch.from_numpy(v_np[::6, :]).unsqueeze(0).to(visual_posi.device)
            
            # Model inference
            preds_qa, _, _ = model(audio, visual_posi, visual_nega, question)
            _, predicted = torch.max(preds_qa.data, 1)
            
            total += preds_qa.size(0)
            correct += (predicted == target).sum().item()
            
            # Track swap stats
            x = samples[batch_idx]
            qtype = ast.literal_eval(x['type'])
            swapped_audio = int(audio_targets[batch_idx] != orig_key) if conflict_mode in ('audio','both') else 0
            swapped_video = int(video_targets[batch_idx] != orig_key) if conflict_mode in ('video','both') else 0
            
            if conflict_mode == 'both' and audio_targets[batch_idx] != video_targets[batch_idx]:
                distinct_ok_count += 1
            if swapped_audio:
                swapped_audio_count += 1
            if swapped_video:
                swapped_video_count += 1
            
            if conflict_mode in ('audio','both'):
                print(f"Audio swap: {orig_key} -> {a_name}")
            if conflict_mode in ('video','both'):
                print(f"Video swap: {orig_key} -> {v_name}")
            
            # Categorize result
            is_correct = (predicted == target).sum().item()
            bucket_map = {
                ('Audio', 'Existential'): 'A_ext', ('Audio', 'Counting'): 'A_count',
                ('Audio', 'Comparative'): 'A_cmp', ('Audio', 'Temporal'): 'A_temp', ('Audio', 'Causal'): 'A_caus',
                ('Visual', 'Existential'): 'V_ext', ('Visual', 'Location'): 'V_loc',
                ('Visual', 'Counting'): 'V_count', ('Visual', 'Temporal'): 'V_temp', ('Visual', 'Causal'): 'V_caus',
                ('Audio-Visual', 'Existential'): 'AV_ext', ('Audio-Visual', 'Counting'): 'AV_count',
                ('Audio-Visual', 'Location'): 'AV_loc', ('Audio-Visual', 'Comparative'): 'AV_cmp',
                ('Audio-Visual', 'Temporal'): 'AV_temp', ('Audio-Visual', 'Causal'): 'AV_caus',
                ('Audio-Visual', 'Purpose'): 'AV_purp',
            }
            key = (qtype[0], qtype[1])
            if key in bucket_map:
                buckets[bucket_map[key]].append(is_correct)
            
            # Log sample details
            prob_dist = F.softmax(preds_qa, dim=1)
            pred_idx = torch.argmax(prob_dist).item()
            pred_answer = ans_list[pred_idx]
            
            qid = x.get('question_id', None)
            if qid:
                print(f"Question ID: {qid}")
            print(f"Video ID: {x.get('video_id','')}")
            print(f"Question: {question}")
            
            # Extract question text
            question_text = None
            try:
                q_tokens = x['question_content'].rstrip().split(' ')
                if q_tokens:
                    q_tokens[-1] = q_tokens[-1][:-1]
                p_sub = 0
                for ii in range(len(q_tokens)):
                    if '<' in q_tokens[ii]:
                        q_tokens[ii] = ast.literal_eval(x['templ_values'])[p_sub]
                        p_sub += 1
                question_text = ' '.join(q_tokens) + '?'
                print(f"Question Text: {question_text}")
            except:
                pass
            
            type_primary = qtype[0] if len(qtype) > 0 else ''
            type_secondary = qtype[1] if len(qtype) > 1 else ''
            print(f"Modality Type: {type_primary}/{type_secondary}")
            print(f"Swapped Audio: {bool(swapped_audio)} | Swapped Video: {bool(swapped_video)}")
            
            gt_answer = None
            try:
                gt_idx = target.item()
                gt_answer = ans_list[gt_idx]
                print(f"Ground Truth: {gt_answer}")
            except:
                print("Ground Truth: <unavailable>")
            
            print(f"Predicted Answer: {pred_answer}")
            
            top_pairs = []
            try:
                k = min(5, prob_dist.shape[1])
                topv, topi = torch.topk(prob_dist, k=k, dim=1)
                top_pairs = [(ans_list[idx], float(prob)) for idx, prob in zip(topi.squeeze(0).tolist(), topv.squeeze(0).tolist())]
                print(f"Top-{k}: {top_pairs}")
            except:
                pass
            print(f"Probability Distribution: {prob_dist.tolist()}")
            print()
            
            # CSV row
            csv_rows.append({
                'question_id': qid or '',
                'question_text': question_text or '',
                'ground_truth': gt_answer or '',
                'predicted': pred_answer,
                'correct': is_correct,
                'video_id': x.get('video_id',''),
                'type_primary': type_primary,
                'type_secondary': type_secondary,
                'swapped_audio': swapped_audio,
                'swapped_video': swapped_video,
                'audio_from': orig_key if conflict_mode in ('audio','both') else '',
                'audio_to': a_name if conflict_mode in ('audio','both') else '',
                'video_from': orig_key if conflict_mode in ('video','both') else '',
                'video_to': v_name if conflict_mode in ('video','both') else '',
                'top5': "; ".join([f"{lbl}:{prob:.4f}" for lbl, prob in top_pairs]),
            })
    
    # Compute aggregates
    A_correct = sum(sum(buckets[k]) for k in ['A_ext','A_count','A_cmp','A_temp','A_caus'])
    A_total = sum(len(buckets[k]) for k in ['A_ext','A_count','A_cmp','A_temp','A_caus'])
    V_correct = sum(sum(buckets[k]) for k in ['V_ext','V_loc','V_count','V_temp','V_caus'])
    V_total = sum(len(buckets[k]) for k in ['V_ext','V_loc','V_count','V_temp','V_caus'])
    AV_correct = sum(sum(buckets[k]) for k in ['AV_ext','AV_count','AV_loc','AV_cmp','AV_temp','AV_caus','AV_purp'])
    AV_total = sum(len(buckets[k]) for k in ['AV_ext','AV_count','AV_loc','AV_cmp','AV_temp','AV_caus','AV_purp'])
    overall_acc = 100 * correct / total
    
    # Print results
    _print_accuracy_breakdown(buckets, A_correct, A_total, V_correct, V_total, AV_correct, AV_total, overall_acc)
    
    # Build stats dict
    def pct(numer, denom):
        return 100 * numer / denom if denom > 0 else None
    
    stats = {
        'overall_acc': overall_acc,
        'A_ext': pct(sum(buckets['A_ext']), len(buckets['A_ext'])),
        'A_count': pct(sum(buckets['A_count']), len(buckets['A_count'])),
        'A_cmp': pct(sum(buckets['A_cmp']), len(buckets['A_cmp'])),
        'A_temp': pct(sum(buckets['A_temp']), len(buckets['A_temp'])),
        'A_caus': pct(sum(buckets['A_caus']), len(buckets['A_caus'])),
        'A_acc': pct(A_correct, A_total),
        'V_ext': pct(sum(buckets['V_ext']), len(buckets['V_ext'])),
        'V_loc': pct(sum(buckets['V_loc']), len(buckets['V_loc'])),
        'V_count': pct(sum(buckets['V_count']), len(buckets['V_count'])),
        'V_temp': pct(sum(buckets['V_temp']), len(buckets['V_temp'])),
        'V_caus': pct(sum(buckets['V_caus']), len(buckets['V_caus'])),
        'V_acc': pct(V_correct, V_total),
        'AV_ext': pct(sum(buckets['AV_ext']), len(buckets['AV_ext'])),
        'AV_count': pct(sum(buckets['AV_count']), len(buckets['AV_count'])),
        'AV_loc': pct(sum(buckets['AV_loc']), len(buckets['AV_loc'])),
        'AV_cmp': pct(sum(buckets['AV_cmp']), len(buckets['AV_cmp'])),
        'AV_temp': pct(sum(buckets['AV_temp']), len(buckets['AV_temp'])),
        'AV_caus': pct(sum(buckets['AV_caus']), len(buckets['AV_caus'])),
        'AV_purp': pct(sum(buckets['AV_purp']), len(buckets['AV_purp'])),
        'AV_acc': pct(AV_correct, AV_total),
        'total_samples': total,
        'swapped_audio_count': swapped_audio_count,
        'swapped_video_count': swapped_video_count,
        'all_audio_swapped': (swapped_audio_count == total) if conflict_mode in ('audio','both') else None,
        'all_video_swapped': (swapped_video_count == total) if conflict_mode in ('video','both') else None,
        'distinct_ok_count': distinct_ok_count,
        'all_distinct_targets': (distinct_ok_count == total) if conflict_mode == 'both' else None,
        'A_total': A_total, 'A_correct': A_correct,
        'V_total': V_total, 'V_correct': V_correct,
        'AV_total': AV_total, 'AV_correct': AV_correct,
    }
    
    # Update banned sets for next run
    if conflict_mode in ('audio','both') and banned_audio is not None:
        for i in range(n):
            banned_audio[i].add(audio_targets[i])
    if conflict_mode in ('video','both') and banned_video is not None:
        for i in range(n):
            banned_video[i].add(video_targets[i])
    
    # Write output files
    if out_path is not None:
        _write_summary_file(out_path, conflict_mode, total, overall_acc, buckets,
                           A_correct, A_total, V_correct, V_total, AV_correct, AV_total,
                           swapped_audio_count, swapped_video_count, distinct_ok_count)
        # CSV
        try:
            csv_path = os.path.splitext(out_path)[0] + "_preds.csv"
            with open(csv_path, 'w', newline='') as cf:
                fieldnames = ['question_id','video_id','question_text','type_primary','type_secondary','ground_truth','predicted','correct','swapped_audio','swapped_video','audio_from','audio_to','video_from','video_to','top5']
                csv_writer = csv.DictWriter(cf, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(csv_rows)
            print(f"Saved per-sample predictions CSV to: {csv_path}")
        except Exception as e:
            print(f"Warning: failed to write per-sample CSV: {e}")
    
    return stats

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default='/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/data/feats/vggish/', help="audio dir")
    # parser.add_argument(
    #     "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps', help="video dir")
    parser.add_argument(
        "--video_res14x14_dir", type=str, default='/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/data/feats/res18_14x14/', help="res14x14 dir")
    
    parser.add_argument(
        "--label_train", type=str, default="/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/data/json/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/data/json/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/data/json/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='test', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='/fp/homes01/u01/ec-hallvaih/MusiQAl_extracted/MusiQAl/AVST/net_grd_avst/avst_models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='avst_lr', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')
    parser.add_argument(
        '--results_out', type=str, default=None, help='Write a concise test summary to this file')
    parser.add_argument(
        '--conflict_mode', type=str, choices=['none','audio','video','both'], default='audio', help='Derangement mode for sensory conflict during test')
    parser.add_argument(
        '--num_runs', type=int, default=5, help='Number of randomized runs to evaluate when swapping audio/video (modes: audio, video, both). Ignored for none')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net()
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


        # ===================================== load pretrained model ===============================================
        ####### concat model
        pretrained_file = "./AVST/grounding_gen/models_grounding_gen/main_grounding_gen_best.pt"
        checkpoint = torch.load(pretrained_file)
        print("\n-------------- loading pretrained models --------------")
        model_dict = model.state_dict()
        tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
        tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
        pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in checkpoint.items() if k in tmp2}

        model_dict.update(pretrained_dict1) #利用预训练模型的参数，更新模型
        model_dict.update(pretrained_dict2) #利用预训练模型的参数，更新模型
        model.load_state_dict(model_dict)

        print("\n-------------- load pretrained models --------------")

        # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader, epoch)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")

    else:
        test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                   transform=transforms.Compose([ToTensor()]), mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
    # Tee all prints from test() to a txt file as well as stdout
        log_dir = os.path.join(args.model_save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"test_output_{TIMESTAMP.rstrip('/').replace(':','-')}.txt")
        with open(log_path, 'w') as lf:
            orig_stdout = sys.stdout
            sys.stdout = Tee(orig_stdout, lf)
            try:
                # Use provided seed for deterministic derangement in sensory conflict eval
                # Also write a concise, easy-to-find summary file
                summary_path = args.results_out
                if summary_path is None:
                    summary_path = os.path.join(log_dir, f"summary_{TIMESTAMP.rstrip('/').replace(':','-')}.txt")
                # Build global set of available video_ids from train/val/test for robust target pool
                def _collect_vids(json_path):
                    try:
                        data = json.load(open(json_path, 'r'))
                        return {str(x.get('video_id', '')) for x in data if 'video_id' in x}
                    except Exception:
                        return set()
                all_vids = set()
                all_vids |= _collect_vids(args.label_train)
                all_vids |= _collect_vids(args.label_val)
                all_vids |= _collect_vids(args.label_test)
                # Verify coverage of expected id range 0..309; normalize zero-padded strings to ints
                norm_all_vids = set()
                non_numeric = set()
                for v in all_vids:
                    try:
                        norm_all_vids.add(int(v))
                    except Exception:
                        non_numeric.add(v)
                expected_ints = set(range(310))
                missing_ints = sorted(list(expected_ints - norm_all_vids))
                extra_ints = sorted(list(norm_all_vids - expected_ints))
                has_full_range = (len(missing_ints) == 0 and len(extra_ints) == 0)
                # Heuristic to detect zero-padding style for clarity
                raw_len_sample = None
                try:
                    raw_len_sample = max(len(s) for s in all_vids) if all_vids else None
                except Exception:
                    pass
                note = "normalized ids from zero-padded strings" if raw_len_sample and raw_len_sample >= 3 else "normalized ids"
                print(f"Video ID coverage check: expected 0-309 present: {'YES' if has_full_range else 'NO'} | present={len(norm_all_vids)} | missing={len(missing_ints)} | extra={len(extra_ints)} | note={note}")
                if non_numeric:
                    print(f"Non-numeric video_ids ignored in coverage check (count={len(non_numeric)}, first 10): {list(sorted(non_numeric))[:10]}")
                if missing_ints:
                    print(f"Missing ints (first 25): {missing_ints[:25]}")
                if extra_ints:
                    print(f"Extra ints not in 0-309 (first 25): {extra_ints[:25]}")
                # Multi-run evaluation for swap modes
                if args.conflict_mode in ('audio','video','both') and args.num_runs > 1:
                    base, ext = os.path.splitext(summary_path)
                    metrics_runs = []
                    # Track per-sample targets across runs (avoid repeats for same modality)
                    n_ds = len(test_loader.dataset)
                    used_audio = [set() for _ in range(n_ds)]
                    used_video = [set() for _ in range(n_ds)]
                    for i in range(1, args.num_runs+1):
                        run_seed = (args.seed + i) if (args.seed is not None) else None
                        run_out = f"{base}_run{i}{ext if ext else '.txt'}"
                        print(f"\n===== Run {i}/{args.num_runs} (seed={run_seed}) =====")
                        m = test(model, test_loader, seed=run_seed, out_path=run_out, conflict_mode=args.conflict_mode, banned_audio=used_audio, banned_video=used_video, available_vids=all_vids)
                        metrics_runs.append(m)
                    # Cross-run uniqueness audit (no sample reused the same target across runs)
                    orig_vids = [str(test_loader.dataset.samples[i]['video_id']) for i in range(n_ds)]
                    audit_lines = []
                    if args.conflict_mode in ('audio','both'):
                        unique_ok_a = sum(1 for i in range(n_ds) if len(used_audio[i]) == args.num_runs)
                        self_free_a = sum(1 for i in range(n_ds) if orig_vids[i] not in used_audio[i])
                        audit_lines.append(
                            f"Cross-run uniqueness (audio): unique-per-sample targets across {args.num_runs} runs: {unique_ok_a}/{n_ds}; excludes original: {self_free_a}/{n_ds}"
                        )
                    if args.conflict_mode in ('video','both'):
                        unique_ok_v = sum(1 for i in range(n_ds) if len(used_video[i]) == args.num_runs)
                        self_free_v = sum(1 for i in range(n_ds) if orig_vids[i] not in used_video[i])
                        audit_lines.append(
                            f"Cross-run uniqueness (video): unique-per-sample targets across {args.num_runs} runs: {unique_ok_v}/{n_ds}; excludes original: {self_free_v}/{n_ds}"
                        )
                    for line in audit_lines:
                        print(line)
                    # Aggregate to main summary_path
                    def mean_of(key):
                        vals = [m.get(key) for m in metrics_runs if m.get(key) is not None]
                        return (sum(vals) / len(vals)) if vals else None
                    # Verification across runs
                    def count_true(key):
                        return sum(1 for m in metrics_runs if m.get(key) is True)
                    agg_keys = ['overall_acc','A_ext','A_count','A_cmp','A_temp','A_caus','A_acc','V_ext','V_loc','V_count','V_temp','V_caus','V_acc','AV_ext','AV_count','AV_loc','AV_cmp','AV_temp','AV_caus','AV_purp','AV_acc']
                    try:
                        with open(summary_path, 'w') as f:
                            f.write('Sensory conflict evaluation - Aggregate\n')
                            f.write(f'Mode: {args.conflict_mode}\n')
                            f.write(f'Timestamp: {TIMESTAMP}\n')
                            f.write(f'Num runs: {args.num_runs}\n')
                            # Write the same ID coverage audit (normalized to ints)
                            f.write(f"Video ID coverage check: expected 0-309 present: {'YES' if has_full_range else 'NO'} | present={len(norm_all_vids)} | missing={len(missing_ints)} | extra={len(extra_ints)} | note={note}\n")
                            if non_numeric:
                                f.write(f"Non-numeric video_ids ignored in coverage check (count={len(non_numeric)}): {sorted(list(non_numeric))}\n")
                            if missing_ints:
                                f.write(f"Missing ints: {missing_ints}\n")
                            if extra_ints:
                                f.write(f"Extra ints not in 0-309: {extra_ints}\n")
                            if metrics_runs:
                                ts = metrics_runs[0].get('total_samples')
                                if ts is not None:
                                    f.write(f'Num samples per run: {ts}\n')
                            f.write('\nPer-run Overall Accuracy:\n')
                            for i, m in enumerate(metrics_runs, start=1):
                                f.write(f'  Run {i}: {m.get("overall_acc", 0):.2f} %\n')
                            # Cross-run swap verification summary
                            if args.conflict_mode in ('audio','both'):
                                ok_runs_a = count_true('all_audio_swapped')
                                f.write(f"\nVerification (audio): All runs swapped different media for 100% of samples: {'YES' if ok_runs_a == len(metrics_runs) else 'NO'} ({ok_runs_a}/{len(metrics_runs)}).\n")
                            if args.conflict_mode in ('video','both'):
                                ok_runs_v = count_true('all_video_swapped')
                                f.write(f"Verification (video): All runs swapped different media for 100% of samples: {'YES' if ok_runs_v == len(metrics_runs) else 'NO'} ({ok_runs_v}/{len(metrics_runs)}).\n")
                            if args.conflict_mode == 'both':
                                ok_runs_b = count_true('all_distinct_targets')
                                f.write(f"Verification (both): All runs used distinct audio/video media targets for 100% of samples: {'YES' if ok_runs_b == len(metrics_runs) else 'NO'} ({ok_runs_b}/{len(metrics_runs)}).\n")
                            # Selected averages with friendly labels
                            oa = mean_of('overall_acc')
                            aa = mean_of('A_acc')
                            va = mean_of('V_acc')
                            ava = mean_of('AV_acc')
                            f.write('\nAverages across runs (selected):\n')
                            if oa is not None:
                                f.write(f'Overall accuracy: {oa:.2f} %\n')
                            if aa is not None:
                                f.write(f'Audio (All) accuracy: {aa:.2f} %\n')
                            if va is not None:
                                f.write(f'Visual (All) accuracy: {va:.2f} %\n')
                            if ava is not None:
                                f.write(f'Audio-Visual (All) accuracy: {ava:.2f} %\n')
                            f.write('\nAverages across runs:\n')
                            for k in agg_keys:
                                val = mean_of(k)
                                if val is not None:
                                    f.write(f'{k}: {val:.2f} %\n')
                            # Cross-run uniqueness audit details
                            if audit_lines:
                                f.write('\nCross-run uniqueness audit:\n')
                                for line in audit_lines:
                                    f.write(line + '\n')
                    except Exception as e:
                        print(f"Warning: failed to write aggregate summary to {summary_path}: {e}")
                else:
                    test(model, test_loader, seed=args.seed, out_path=summary_path, conflict_mode=args.conflict_mode, available_vids=all_vids)
            finally:
                sys.stdout = orig_stdout
        print(f"Saved verbose test log to: {log_path}")
        if args.conflict_mode in ('audio','video','both') and args.num_runs > 1:
            print(f"Saved aggregate summary to: {summary_path}")
            base, ext = os.path.splitext(summary_path)
            for i in range(1, args.num_runs+1):
                print(f"Saved per-run summary to: {base}_run{i}{ext if ext else '.txt'}")
        else:
            print(f"Saved summary results to: {summary_path}")


if __name__ == '__main__':
    main()