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
# from .net_avst import AVQA_Fusion_Net

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/net_avst/'+TIMESTAMP)

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")


sara_abs_path = '/fp/homes01/u01/ec-sarapje/Dataset/Data/data/'
my_source_dir = sara_abs_path


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




def test(model, test_loader, run_nr = 0):
    model.eval()
    total = 0
    correct = 0

    samples = json.load(open("../json/avqa-test.json", 'r'))

    failed_filename = 'test_results/failed_questions_run_' + str(run_nr) + '.csv'
    success_filename = 'test_results/success_questions_run_' + str(run_nr) + '.csv'
    details_filename = 'test_results/details_run_' + str(run_nr) + '.txt'
    final_results_filename = 'test_results/final_results_run_' + str(run_nr) + '.txt'
    
    with open(failed_filename, 'w') as failed_file:
        failed_file.write('idx,question_id,video_id\n')

    with open(success_filename, 'w') as success_file:
        success_file.write('idx,question_id,video_id\n')

    with open(details_filename, 'w') as details_file:
        details_file.write('Details of test run\n\n')

    with open(final_results_filename, 'w') as final_results_file:
        final_results_file.write(f'Final results of test run {run_nr}\n\n')

    A_ext = []
    A_count = []
    A_cmp = []
    A_temp = []
    A_caus = []
    V_ext = []
    V_loc = []
    V_count = []
    V_temp = []
    V_caus = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    AV_caus = []
    AV_purp = []

    with torch.no_grad():

        for batch_idx, sample in enumerate(test_loader):

            audio, visual_posi, visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

            preds_qa,out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            print("Preds shape:", preds.shape)   # should be [batch, num_classes]
            print("Predicted min/max:", predicted.min().item(), predicted.max().item())
            print("Target dtype:", target.dtype, "Target min:", target.min().item(), "Target max:", target.max().item())


            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]

            print('Idx', batch_idx)
            print(x['question_id'])
            print(x['video_id'])
            print('Predicted: ', predicted)
            print('Target: ', target)
            print('Correct until now: ', correct)
            print('Samples tested until now: ', total)
            print('Accuracy until now: %.2f %%' % (100 * correct / total))
            print('\n')

            with open(details_filename, 'a') as details_file:
                details_file.write('Idx ' + str(batch_idx) + '\n')
                details_file.write('Question ID: ' + str(x['question_id']) + '\n')
                details_file.write('Video ID: ' + str(x['video_id']) + '\n')
                details_file.write('Predicted: ' + str(predicted.item()) + '\n')
                details_file.write('Target: ' + str(target.item()) + '\n')
                details_file.write('Correct until now: ' + str(correct) + '\n')
                details_file.write('Samples tested until now: ' + str(total) + '\n')
                details_file.write('Accuracy until now: %.2f %%' % (100 * correct / total) + '\n')
                details_file.write('\n')

            if (predicted == target):
                with open(success_filename, 'a') as success_file:
                    success_file.write(f'{batch_idx},{x["question_id"]},{x["video_id"]}\n')

            else:
                with open(failed_filename, 'a') as failed_file:
                    failed_file.write(f'{batch_idx},{x["question_id"]},{x["video_id"]}\n')


            type =ast.literal_eval(x['type'])

            if type[0] == 'Audio':
                if type[1] == 'Existential':
                    A_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    A_temp.append((predicted == target).sum().item())
                elif type[1] == 'Causal':
                    A_caus.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Existential':
                    V_ext.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    V_temp.append((predicted == target).sum().item())
                elif type[1] == 'Causal':
                    V_caus.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())
                elif type[1] == 'Causal':
                    AV_caus.append((predicted == target).sum().item())
                elif type[1] == 'Purpose':
                    AV_purp.append((predicted == target).sum().item())


    with open(final_results_filename, 'a') as final_results_file:
        final_results_file.write('Audio Existential Accuracy: %.2f %%\n' % (
            100 * sum(A_ext)/len(A_ext)))
        final_results_file.write('Audio Counting Accuracy: %.2f %%\n' % (
            100 * sum(A_count)/len(A_count)))
        final_results_file.write('Audio Cmp Accuracy: %.2f %%\n' % (
            100 * sum(A_cmp) / len(A_cmp)))
        final_results_file.write('Audio Temp Accuracy: %.2f %%\n' % (
            100 * sum(A_temp) / len(A_temp)))
        final_results_file.write('Audio Caus Accuracy: %.2f %%\n' % (
            100 * sum(A_caus) / len(A_caus)))
        final_results_file.write('Audio Accuracy: %.2f %%\n' % (
            100 * (sum(A_ext) + sum(A_count) + sum(A_cmp) + sum(A_temp) + sum(A_caus)) / (len(A_ext)+len(A_count) + len(A_cmp)+len(A_temp)+len(A_caus))))
        final_results_file.write('Visual Ext Accuracy: %.2f %%\n' % (
            100 * sum(V_ext) / len(V_ext)))
        final_results_file.write('Visual Loc Accuracy: %.2f %%\n' % (
            100 * sum(V_loc) / len(V_loc)))
        final_results_file.write('Visual Counting Accuracy: %.2f %%\n' % (
            100 * sum(V_count) / len(V_count)))
        final_results_file.write('Visual Temp Accuracy: %.2f %%\n' % (
            100 * sum(V_temp) / len(V_temp)))
        final_results_file.write('Visual Caus Accuracy: %.2f %%\n' % (
            100 * sum(V_caus) / len(V_caus)))
        final_results_file.write('Visual Accuracy: %.2f %%\n' % (
            100 * (sum(V_ext)+sum(V_loc) + sum(V_count)+sum(V_temp)+sum(V_caus)) / (len(V_ext)+len(V_loc) + len(V_count)+len(V_temp)+len(V_caus))))
        final_results_file.write('AV Ext Accuracy: %.2f %%\n' % (
            100 * sum(AV_ext) / len(AV_ext)))
        final_results_file.write('AV counting Accuracy: %.2f %%\n' % (
            100 * sum(AV_count) / len(AV_count)))
        final_results_file.write('AV Loc Accuracy: %.2f %%\n' % (
            100 * sum(AV_loc) / len(AV_loc)))
        final_results_file.write('AV Cmp Accuracy: %.2f %%\n' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
        final_results_file.write('AV Temporal Accuracy: %.2f %%\n' % (
            100 * sum(AV_temp) / len(AV_temp)))
        final_results_file.write('AV Caus Accuracy: %.2f %%\n' % (
            100 * sum(AV_caus) / len(AV_caus)))
        final_results_file.write('AV Purp Accuracy: %.2f %%\n' % (
            100 * sum(AV_purp) / len(AV_purp))) 
        final_results_file.write('AV Accuracy: %.2f %%\n' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)+sum(AV_caus)+sum(AV_purp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp)+len(AV_caus)+len(AV_purp))))
        final_results_file.write('Overall Accuracy: %.2f %%\n' % (
            100 * correct / total))






    print('Audio Existential Accuracy: %.2f %%' % (
            100 * sum(A_ext)/len(A_ext)))
    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Temp Accuracy: %.2f %%' % (
            100 * sum(A_temp) / len(A_temp)))
    print('Audio Caus Accuracy: %.2f %%' % (
            100 * sum(A_caus) / len(A_caus)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_ext) + sum(A_count) + sum(A_cmp) + sum(A_temp) + sum(A_caus)) / (len(A_ext)+len(A_count) + len(A_cmp)+len(A_temp)+len(A_caus))))
    print('Visual Ext Accuracy: %.2f %%' % (
            100 * sum(V_ext) / len(V_ext)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Temp Accuracy: %.2f %%' % (
            100 * sum(V_temp) / len(V_temp)))
    print('Visual Caus Accuracy: %.2f %%' % (
            100 * sum(V_caus) / len(V_caus)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_ext)+sum(V_loc) + sum(V_count)+sum(V_temp)+sum(V_caus)) / (len(V_ext)+len(V_loc) + len(V_count)+len(V_temp)+len(V_caus))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))
    print('AV Caus Accuracy: %.2f %%' % (
            100 * sum(AV_caus) / len(AV_caus)))
    print('AV Purp Accuracy: %.2f %%' % (
            100 * sum(AV_purp) / len(AV_purp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)+sum(AV_caus)+sum(AV_purp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp)+len(AV_caus)+len(AV_purp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total



def main():

    #Relevant parameters
    mode = 'test'
    sara_abs_path = '/fp/homes01/u01/ec-sarapje/Dataset/Data/data/'
    my_source_dir = sara_abs_path

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default = my_source_dir + 'feats/vggish', help="audio dir")
    # parser.add_argument(
    #     "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps', help="video dir")
    parser.add_argument(
        "--video_res14x14_dir", type=str, default=my_source_dir + 'feats/res18_14x14', help="res14x14 dir")

    parser.add_argument(
        "--label_train", type=str, default="../json/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="../json/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="../json/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default=mode, help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    # parser.add_argument(
    #     "--model_save_dir", type=str, default='net_grd_avst/avst_models/', help="model save dir")
    parser.add_argument(
        "--model_save_dir", type=str, default='avst_models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='avst', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)


    model = AVQA_Fusion_Net()
    model = nn.DataParallel(model)
    print("Sent model to GPU")

    val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='val')
    print("Length of val dataset: ", val_dataset.__len__())

    test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                transform=transforms.Compose([ToTensor()]), mode_flag='test')

    print("Length of test dataset: ", test_dataset.__len__())
    print()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    #model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
    #model.load_state_dict(torch.load('./avst_models/avst.pt'))

    model.load_state_dict(torch.load('./avst_models/avst_newest.pt'))

    model = model.to('cuda') #Try to send the model to GPU after loading the state dict
 
    test(model, test_loader, run_nr = 1)

    print("Testing done.")



if __name__ == '__main__':
    print("Starting test...")
    main()

    #samples = json.load(open("../json/avqa-test.json", 'r'))

    #print(samples[0])

    #print(samples[0]['question_id'])

