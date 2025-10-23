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
    out_match = torch.zeros(out_match_posi.shape[0] * 2,
                        out_match_posi.shape[1],
                        device=out_match_posi.device,
                        dtype=out_match_posi.dtype)

    batch_labels = torch.zeros(out_match_posi.shape[0] * 2,
                            device=out_match_posi.device,
                            dtype=torch.long)

    for i in range(out_match_posi.shape[0]):

        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion, epoch, progress_epoch_filename):
    model.train()
    total_qa = 0
    correct_qa = 0

    # progress_filename = 'progress/avst_progress.csv'
    # with open(progress_filename, 'w') as f:
    #     f.write('epoch,batch_idx,loss_qa,loss_match,loss_both\n')

    for batch_idx, sample in enumerate(train_loader):
        audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

        optimizer.zero_grad()
        out_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)  
        out_match, match_label = batch_organize(out_match_posi,out_match_nega)  
        out_match ,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()

        preds_qa, out_match_posi, out_match_nega = model(audio, visual_posi,visual_nega, question)

        _, predicted = torch.max(out_qa.data, 1)
        total_qa += out_qa.size(0)
        correct_qa += (predicted == target).sum().item()

        loss_match=criterion(out_match, match_label)
        loss_qa = criterion(out_qa, target)
        loss = loss_qa + 0.5*loss_match

        writer.add_scalar('run/match',loss_match.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/qa_test',loss_qa.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/both',loss.item(), epoch * len(train_loader) + batch_idx)

        # with open(progress_filename, 'a') as f:
        #     f.write(f'{epoch},{batch_idx},{loss_qa.item()},{loss_match.item()},{loss.item()}\n')

        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    accuracy = 100 * correct_qa / total_qa

    with open(progress_epoch_filename, 'a') as f:
        f.write(f'{epoch},{loss_qa.item()},{loss_match.item()},{loss.item()},{accuracy}\n')


def eval(model, val_loader, epoch, eval_filename):
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
    print()

    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    with open(eval_filename, 'a') as f:
        f.write(f'{epoch},{100 * correct_qa / total_qa}\n')

    return 100 * correct_qa / total_qa


def test(model, val_loader):
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
        for batch_idx, sample in enumerate(val_loader):
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
                
            #     # Now you have the probability distribution
            # probability_distribution = F.softmax(preds, dim=1)

            # # Find the index of the highest probability answer
            # predicted_answer_index = torch.argmax(probability_distribution).item()

            # answer_dict = my_source_dir + 'ans_vocab.txt'

            # # Map the index to your answer dictionary
            # predicted_answer = answer_dict[predicted_answer_index]

            # # Print the predicted answer
            # print(f"Question: {question}")
            # print(f"Predicted Answer: {predicted_answer}")
            # print(f"Probability Distribution: {probability_distribution.tolist()}")


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
    # mode = 'train'
    sara_abs_path = '/fp/homes01/u01/ec-sarapje/Dataset/Data/data/'
    my_source_dir = sara_abs_path


    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default = my_source_dir + 'feats/vggish', help="audio dir")
    parser.add_argument(
        "--video_res14x14_dir", type=str, default = my_source_dir + 'feats/res18_14x14', help="res14x14 dir")

    parser.add_argument(
        "--label_train", type=str, default="../json/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="../json/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="../json/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1.46e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="which model to use")
    parser.add_argument(
        "--mode", type=str, default=mode, help="which mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='avst_models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='avst_lr', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1', help='gpu device number')


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
                                    mode_flag='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    mode_flag='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


        # ===================================== load pretrained model ===============================================
        ####### concat model
        pretrained_file = "../grounding_gen/models_grounding_gen/main_grounding_gen_best_new.pt"
        checkpoint = torch.load(pretrained_file)

        print("\n-------------- loading pretrained models --------------")
        model_dict = model.state_dict()
        tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
        tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
        pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in checkpoint.items() if k in tmp2}

        model_dict.update(pretrained_dict1) # Update the model using the parameters of the pre-trained model
        model_dict.update(pretrained_dict2) # Update the model using the parameters of the pre-trained model
        model.load_state_dict(model_dict)

        print("\n-------------- successfully loaded pretrained models --------------")

        # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0

        progress_epoch_filename = 'progress/avst_epoch_progress_lr.csv'
        eval_filename = 'progress/avst_epoch_eval_lr.csv'

        with open(progress_epoch_filename, 'w') as f:
            f.write('epoch,loss_qa,loss_match,loss_both,train_accuracy\n')
        
        with open(eval_filename, 'w') as f:
            f.write('epoch,val_accuracy\n')

        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch, progress_epoch_filename=progress_epoch_filename)

            scheduler.step(epoch)
            F = eval(model, val_loader, epoch, eval_filename)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
                print('Saved model')
                print('Current best epoch:', epoch) 
                print('Current best val acc:', best_F) 
                print()


        # ===================================== load pretrained model ===============================================
    
    else:
        print("\n-------------- Testing: load pretrained models --------------")
        test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                   mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        test(model, test_loader)


if __name__ == '__main__':
    main()