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




def test(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open("../json/avqa-test.json", 'r'))
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
            audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

            preds_qa,out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
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
                
                # Now you have the probability distribution
            probability_distribution = F.softmax(preds, dim=1)

            # Find the index of the highest probability answer
            predicted_answer_index = torch.argmax(probability_distribution).item()

            answer_dict = my_source_dir + 'ans_vocab.txt'

            # Map the index to your answer dictionary
            predicted_answer = answer_dict[predicted_answer_index]

            # Print the predicted answer
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Probability Distribution: {probability_distribution.tolist()}")



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



    test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                transform=transforms.Compose([ToTensor()]), mode_flag='test')
    print("Length of dataset: ", test_dataset.__len__())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    #model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
    model.load_state_dict(torch.load('./avst_models/avst.pt'))

    model = model.to('cuda') #Try to send the model to GPU after loading the state dict

    #model.load_state_dict(torch.load('./avst_models/models_grounding_genmain_grounding_gen_2_best.pt'))
    test(model, test_loader)



if __name__ == '__main__':
    print("Starting test...")
    main()