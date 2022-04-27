# =============================================================================
# Baseado no artigo: Creating a Very Simple U-Net Model with PyTorch for Semantic Segmentation of Satellite Images
# Autor: Maurício Cordeiro
# Link: https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
# =============================================================================

# Importa as bibliotecas necessárias
# =============================================================================
import random
from random import shuffle
import time
import os
from pathlib import Path
import datetime

import numpy as np
### import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import models
import torch.nn.functional as F
import torchvision


# Para Dense Net
from torchvision.models import DenseNet
from torchvision.models.densenet import  _Transition, _load_state_dict
from collections import OrderedDict

# Biblioteca que implementa métricas de validação para algoritmos de segmentação.
# --> As métricas implementadas são baseadas no artigo: 
#     "Fully Convolutional Networks for Semantic Segmentation"
# [1] https://github.com/martinkersner/py_img_seg_eval/blob/master/eval_segm.py
from eval_segm import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU

# Configurações para reprodutibilidade dos experimentos.
# --> Ao fixar as sementes para geração de números aleatórios, sempre que repetirmos um mesmo 
#     experimento, garantimos que o resultado é exatamente o mesmo. 
#     É adequado para que os resulatos apresentados em um artigo possam ser reproduzidos.
# =============================================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Configuração da GPU
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\nDevice: {0}'.format(DEVICE))

# ***** Definição do hiperparametros *****
# Reunir os hiperparametros aqui.
# =============================================================================
# Número de épocas que o modelo será treinado.
num_epochs = 50 # 50 epochs (original)
# Taxa de aprendizado
hp_lr = 0.01
# Tamanho do lote
hp_batch_size = 12


# Data e hora do inicio do experimento. Para o relatório.
start_exp = datetime.datetime.now()

# Criando o DataSet
# =============================================================================
class CloudDataset(Dataset):

    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        """
        """
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        """
        """
        files = {'red': r_file, 
                 'green': g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}

        return files
                                       
    def __len__(self):
        """
        """
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False):
        """
        """
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)
    
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    def open_mask(self, idx, add_dims=False):
        """
        """        
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        """
        """        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        ### return x, y
        # Retorna também o caminho para o arquivo. Usado para análise e apresentação dos resultados.
        return (x, y, str(self.files[idx]['red']))
    
    def open_as_pil(self, idx):
        """
        """        
        arr = 256 * self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        """
        """
        s = 'Dataset class with {} files'.format(self.__len__())

        return s


class CloudDatasetTest(Dataset):
    
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, pytorch=True):
        """
        O conjunto de testes do Cloud-38 não possui imagens ground-truth (gt_dir).
        Esta versão do dataset não considera o canal ground-truth.


        """
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir):
        """
        """
        files = {'red': r_file, 
                 'green': g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir')}

        return files
                                       
    def __len__(self):
        """
        """
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False):
        """
        """
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)
    
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    def __getitem__(self, idx):
        """
        """        
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        ### y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        
        ### return x, y
        return (x, str(self.files[idx]['red']))
    
    def open_as_pil(self, idx):
        """
        """        
        arr = 256 * self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        """
        """
        s = 'Dataset class with {} files'.format(self.__len__())

        return s


# The model
# =============================================================================
# Modelo simplificado da UNET. Presente no tutorial:
# Creating a Very Simple U-Net Model with PyTorch for Semantic Segmentation of Satellite Images


class UNetVgg(torch.nn.Module):

    def __init__(self, nClasses):
        super(UNetVgg, self).__init__()

        vgg16pre = torchvision.models.vgg16(pretrained=True)
        self.vgg0 = torch.nn.Sequential(*list(vgg16pre.features.children())[:4])
        self.vgg1 = torch.nn.Sequential(*list(vgg16pre.features.children())[4:9])
        self.vgg2 = torch.nn.Sequential(*list(vgg16pre.features.children())[9:16])
        self.vgg3 = torch.nn.Sequential(*list(vgg16pre.features.children())[16:23])
        self.vgg4 = torch.nn.Sequential(*list(vgg16pre.features.children())[23:30])

        self.bottom = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True)
        )

        self.smooth0 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True)
        )
        self.smooth1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True)
        )
        self.smooth2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True)
        )
        self.smooth3 = torch.nn.Sequential(
            torch.nn.Conv2d(768, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True)
        )
        self.smooth4 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(True)
        )

        self.final = torch.nn.Conv2d(64, nClasses, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat0 = self.vgg0(x)
        feat1 = self.vgg1(feat0)
        feat2 = self.vgg2(feat1)
        feat3 = self.vgg3(feat2)
        feat4 = self.vgg4(feat3)
        feat5 = self.bottom(feat4)

        _, _, H, W = feat4.size()
        up4 = torch.nn.functional.interpolate(feat5, size=(H, W), mode='bilinear', align_corners=False)
        concat4 = torch.cat([feat4, up4], 1)
        end4 = self.smooth4(concat4)

        _, _, H, W = feat3.size()
        up3 = torch.nn.functional.interpolate(end4, size=(H, W), mode='bilinear', align_corners=False)
        concat3 = torch.cat([feat3, up3], 1)
        end3 = self.smooth3(concat3)

        _, _, H, W = feat2.size()
        up2 = torch.nn.functional.interpolate(end3, size=(H, W), mode='bilinear', align_corners=False)
        concat2 = torch.cat([feat2, up2], 1)
        end2 = self.smooth2(concat2)

        _, _, H, W = feat1.size()
        up1 = torch.nn.functional.interpolate(end2, size=(H, W), mode='bilinear', align_corners=False)
        concat1 = torch.cat([feat1, up1], 1)
        end1 = self.smooth1(concat1)

        _, _, H, W = feat0.size()
        up0 = torch.nn.functional.interpolate(end1, size=(H, W), mode='bilinear', align_corners=False)
        concat0 = torch.cat([feat0, up0], 1)
        end0 = self.smooth0(concat0)

        return self.final(end0)


# Implementação simplifada da rede FCN
# Disponível em: https://discuss.pytorch.org/t/problem-with-training-fcn-for-segmentation-of-multi-channel-data/69982/2

# ***** Descomentar a resnet50 para usar na 1080. Utilizei uma resnet18 para poder rodar na 1050 *****

pretrained_net = models.resnet50(pretrained=False)
### pretrained_net = models.resnet18(pretrained=False)

class fcn(nn.Module):
    """
    https://discuss.pytorch.org/t/problem-with-training-fcn-for-segmentation-of-multi-channel-data/69982/2
    """
    def __init__(self, in_channels, num_classes):
        super(fcn, self).__init__()
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        # change input channels to 9
        ### in_channels, out_channels, kernel_size, stride, padding, 
        self.stage1[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]
        #self.scores1 = nn.Conv2d(512, num_classes, 1)
        #self.scores2 = nn.Conv2d(256, num_classes, 1)
        #self.scores3 = nn.Conv2d(128, num_classes, 1)
        self.scores1 = nn.Conv2d(2048, num_classes, 1)
        self.scores2 = nn.Conv2d(1024, num_classes, 1)
        self.scores3 = nn.Conv2d(512, num_classes, 1)
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)

    def forward(self, x):
        print(x.shape)
        x = self.stage1(x)
        s1 = x  # 1/8
        print(s1.shape)
        x = self.stage2(x)
        s2 = x  # 1/16
        print(s2.shape)
        x = self.stage3(x)
        s3 = x  # 1/32
        print(s3.shape)
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s2)
        return s


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=50):
    """
    Função que implementa o laço de treinamento do modelo. 
    Itera ao longo do número de épocas definico (epochs)

    Parameters
    ----------
    model : 
        O modelo da CNN que será treinado (terá os pesos atualizados)
    train_dl : Dataloader
        Dataloader do conjunto de treino.
    valid_dl : Dataloader
        Dataloader do conjunto de validação
    loss_dl : 
        A função de perda utilizada.
    optimizer :
        Otimizador utilizado para treinar a rede.
    acc_fn :
        Função que computa a acurácia entre dois lotes de imagens.
    epochs :
        Número de épocas de treinamento.
    """

    start = time.time()

    # Envia o modelo para a GPU.
    model.cuda()

    # Armazena as perdas de treino e validação para cada época. 
    #   --> Os valores são plotados para analisarmos a evolução do treinamento.
    train_loss, valid_loss = [], []

    # Armazena as acurácias de treino e validação para cada época. 
    #   --> Os valores são plotados para analisarmos a evolução do treinamento.
    train_acc, valid_acc = [], []

    # Melhor acurácia de época
    best_acc = 0.0

    # Itera ao longo das épocas.
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # A cada época realiza uma etapa de treinamento e uma de validação.
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Se a etapa for de treinamento, atualiza os pesos (parametros) do modelo.
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl

            else:
                # Se a etapa for de validação, não atualiza os parametros, apenas avalia a saída.
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            # Inicializa a perda e acurácia desta época
            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # A cada iteração, recebe um lote (batch) de imagens de treino e validação e submete à rede.
            for x, y, _ in dataloader:
                # Alimenta a rede com um lote de imagens (x) e ground-truths (y).

                # Envia os lotes para a GPU.
                x = x.cuda()
                y = y.cuda()

                # Incrementa a contagem de passos desta época.
                step += 1

                # forward pass
                # Propagação adiante dos dados.
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()

                    # Saída da rede para o lote 'x'
                    outputs = model(x)
                    ### outputs = model(x)['out'][0]

                    # Calcula a perda para esta época. Compara a saída da rede (outputs) com a saída deseja (y).
                    loss = loss_fn(outputs, y)

                    # Retropropagação. 
                    # Propaga o erro (gradiente do erro) e atualiza os parametros de acordo com o otimizador.
                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    # Se a etapa é de validação, apenas computa a perda da saída da rede. Sem 
                    with torch.no_grad():
                        outputs = model(x)
                        ### outputs = model(x)['out'][0]
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                # Calcula a acurácia entre o lote gerado pelo modelo e o lote ground-truth.
                acc = acc_fn(outputs, y)

                # Acumula as acurácias e as perdas deste lote para computar a acurácia e perda da época.
                running_acc  += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size 

                # A cada 100 passos, imprime um resumo do treinamento na tela.
                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            # Finaliza o calculo da perda e da acurácia desta época. 
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            # Grava os valores em listas. As listas são retornadas para plotar os relatórios de treinamento.
            # Atualiza a lista de perdas nesta época
            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            # Atualiza a lista de acurácias nesta época
            train_acc.append(epoch_acc) if phase=='train' else valid_acc.append(epoch_acc)

            ### clear_output(wait=True)
            ### print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

    # Calcula o tempo total de treinamento.
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    # Retorna o modelo treinado e as listas de perdas e acurácias, tanto para o treino quanto para a validação.
    ### return model['out'][0], train_loss, valid_loss, train_acc, valid_acc   
    return model, train_loss, valid_loss, train_acc, valid_acc   


def acc_metric(predb, yb):
    """
    Calcula a acurácia entre as imagens preditas e as imagens ground-truth. 
    Calcula para o lote todo (acurácia do lote).
    """
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


def batch_to_img(xb, idx):
    """
    Gera uma imagem RGB para fins de visualização.
    Retorna a imagem com indice 'idx' do lote 'xb'.
    """
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))


def predb_to_mask(predb, idx):
    """
    Gera uma imagem binária a partir da saída da rede.
    Retorna a imagem binária com indice 'idx' do lote 'predb'.
    """
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def predb(predb, idx):

    return torch.functional.F.softmax(predb[idx], 0)


# =============================================================================
# SCRIPT - INÍCIO
# =============================================================================

# Criando os conjuntos de treino e de validação
# -----------------------------------------------------------------------------
# base_path = Path('../input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training')
train_path = Path('../Data/38-Cloud_training')
data = CloudDataset(train_path/'train_red', 
                    train_path/'train_green', 
                    train_path/'train_blue', 
                    train_path/'train_nir',
                    train_path/'train_gt')

# TEST
# --------------------------------------------------
print('Número de imagens no conjunto de dados:')
print(len(data))
# --------------------------------------------------

# Separa 6000 imagens para treino e 2400 para validação
train_ds, valid_ds = torch.utils.data.random_split(data, (6000, 2400))

# Cria os dataloaders para os conjuntos de treino e validação.
train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)


# (!!!!!)
# -----------------------------------------------------------------------------
# (Opção 1) Reutilizar o conjunto de validação para teste
test_dl = DataLoader(valid_ds, batch_size=12, shuffle=False)

# (Opção 2) Usar apenas o conjunto de treino original.
# --> Primeiro, dividir o conjunto de treino, em treino e teste. 
# --> Depois dividir o conjunto de treino, em treino e validação.
# (Não implementado ainda...)

# (Opção 3) Utilizar, para teste, o conjunto de testes oficial do 38-Cloud.
# --> O conjunto de testes oficial não possui o ground-truth.
#     Possui apenas as imagens ground-truth sem recorte e um script Matlab que calcula as métricas 
#     de avaliação. 
test_path = Path('../Data/38-Cloud_test')

# Usamos o Dataset adaptado para o conjunto de testes (ignora as imagens gt).
test_ds = CloudDatasetTest(test_path/'test_red', 
                           test_path/'test_green', 
                           test_path/'test_blue', 
                           test_path/'test_nir')

# O conjunto de testes deve ter shuffle=False
test_dl2 = DataLoader(test_ds, batch_size=hp_batch_size, shuffle=False)


# ***** Escolha o modelo *****
MODEL = 'FCN'  # ou 'FCN'

# Cria uma instância do modelo
# =============================================================================
if MODEL == 'UNET':
    # A imagem de entrada tem 4 canais (red, green, blue, nir). 
    # A imagem de saída tem 2 canais (fundo (0), nuvem (1))
    modelo = UNET(4, 2)

# from my_fcn import MyFCN
# unet = MyFCN(4, 2)
# print(unet)

else:
    modelo = fcn(4, 2)

print(modelo)

# Treina o modelo
# -----------------------------------------------------------------------------
# Define a função de custo (Entropia Cruzada)
loss_fn = nn.CrossEntropyLoss()
# Define o otimizador e a taxa de aprendizado (Adam)
opt = torch.optim.Adam(modelo.parameters(), lr=hp_lr)
# Chama a função que implementa o laço de treinamento.
modelo_trained, train_loss, valid_loss, train_acc, valid_acc = train(modelo, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=num_epochs) 

# DEBUG
print(train_loss)
print(valid_loss)

# Corrige as lista de valores de perda. 
# --> Os elementos da lista estão na GPU. É necessário iterar pela lista e trazer cada elemento 
#     para a CPU.
train_loss_ = []
for loss in train_loss:
    train_loss_.append(loss.item())

valid_loss_ = []
for loss in valid_loss:
    valid_loss_.append(loss.item())

# Acurácias
train_acc_ = []
for acc in train_acc:
    train_acc_.append(acc.item())

valid_acc_ = []
for acc in valid_acc:
    valid_acc_.append(acc.item())

# Plota a o gráfico de treinamento - loss.
plt.figure(figsize=(10,8))
plt.plot(train_loss_, label='Train loss')
plt.plot(valid_loss_, label='Valid loss')
plt.legend()
plt.savefig('training_report_loss.png', bbox_inches='tight')

# Plota a o gráfico de treinamento - acuraccy.
plt.figure(figsize=(10,8))
plt.plot(train_acc_, label='Train acc.')
plt.plot(valid_acc_, label='Valid acc.')
plt.legend()
plt.savefig('training_report_acc.png', bbox_inches='tight')

# Grava um relatório do experimento. 
# --> Armazena algumas informações importantes.
str_rep = f'{start_exp}'
str_rep += f'\nLearning rate: {hp_lr}'
str_rep += f'\nBatch size: {hp_batch_size}'
str_rep += f'\nNumber of epochs: {num_epochs}'
str_rep += f'\nTrain path: {train_path}'
str_rep += f'\nTest path: {test_path}'

exp_file = open('report_exp.txt', 'w')
exp_file.write(str_rep)
exp_file.close()


# TESTANDO O MODELO TREINADO.
# --> Usado para testes quando possuimos as máscaras (imagens ground-truth)
# =============================================================================
print('\nTesting model...')

# Obs.: Pode ser implementado em uma função...
### test_loss = test(modelo, test_dl, loss_fn, opt, acc_metric, epochs=1)

modelo_trained.train(False)  # Set model to evaluate mode

# Lista com as perdas de todos os lotes.
test_loss = []
# Lista com as acurácias de todas as imagens.
acc_list = []

# Listas para armazenas as outras métricas de validação.
pa_list = []
ma_list = []
m_iu_list = []
fe_iu_list = []

# Cria a pasta para os resultados dos experimentos
if not os.path.exists('exp'):
    os.makedirs('exp')

if not os.path.exists('exp_pred'):
    os.makedirs('exp_pred')

# Contador para imagens do conjunto de testes.
img_idx = 0

# Cabeçalho do relatório CSV.
acc_str = '#,image,accuracy,pixel acc.,mean acc.,mean iu, fw. iu, batch acc.'

# Itera por todos os lotes do conjunto de TESTES
for i, (x, y, img_path) in enumerate(test_dl):
    
    with torch.no_grad():
        outputs = modelo_trained(x.cuda())
        ### loss = loss_fn(outputs, y.long())

    # Calcula a acurácia para o lote todo.
    acc = acc_metric(outputs, y)

    print('\n-->')
    print(f'Lote {i}: {acc}')

    # Soma das acurácias individuais deste lote.
    # (para testes)
    acc_batch = 0

    print('==>')
    print(img_path)
   
    # Itera pelas imagens deste lote.
    ### for j in range(hp_batch_size):
    for j, img_path_ in enumerate(img_path):
        # Acurácia desta imagem.
        # (Entre imagem ground-truth e a imagem predita).
        acc = (outputs[j].argmax(dim=0) == y[j].cuda()).float().mean()

        # Atualiza a soma das acurácias
        acc_batch += acc.item()

        print('=====>')
        print(img_path[j])

        # Populate the CSV file (report)
        acc_str += f'\n{img_idx},{img_path[j]},{acc.item()}'

        # Armazena a acurácia em uma lista
        acc_list.append(acc.item())

        # Imprime a acurácia da imagem na tela.
        print(f'{img_idx}: {acc.item():0.4f}')

        # Save images
        plt.imsave(f'exp/{img_idx}_input_image.png', batch_to_img(x, j))
        plt.imsave(f'exp/{img_idx}_mask.png', y[j], cmap='gray')
        plt.imsave(f'exp/{img_idx}_pred_{acc:0.4f}.png', predb_to_mask(outputs, j), cmap='gray')

        # Computar outras métricas de validação
        # ---------------------------------------------------------------------
        # Ground-truth (máscara)
        img_mask = y[j]
        # Imagem predita
        img_pred = predb_to_mask(outputs, j)

        # Nome do arquivo de saída.
        img_filename = str(img_path[j]).split('/')[-1]
        print(img_filename)

        plt.imsave(f'exp_pred/{img_filename}F', img_pred, cmap='gray')

        # IoU
        # Jaccard
        # Dice

        # Computa as demais métricas de validação.
        pa = pixel_accuracy(img_pred, img_mask)
        ma = mean_accuracy(img_pred, img_mask)
        m_iu = mean_IU(img_pred, img_mask)
        fe_iu = frequency_weighted_IU(img_pred, img_mask)

        # Insere as métricas de validação nas respectivas listas.
        pa_list.append(pa)
        ma_list.append(ma)
        m_iu_list.append(m_iu)
        fe_iu_list.append(fe_iu)

        # Grava as métricas de validação no relatório CSV.
        acc_str += f',{pa},{ma},{m_iu},{fe_iu}'

        # Atualiza indice global da imagem
        img_idx += 1

    # Imprime a acurácia média do lote.
    print(f'acc_mean: {(acc_batch / hp_batch_size):0.4f}')

    # Insere a acurácia média do lote no relatório CSV.
    acc_str += f',{(acc_batch / hp_batch_size)}'

    # Armazena a perda do lote.
    ### test_loss.append(loss)
    

# TEST
print(len(acc_list))
print(acc_list)
print(np.mean(acc_list))

# Armazena a acurácia média global no relatório CSV.
acc_str += f'\n,Mean:,{np.mean(acc_list)},{np.mean(pa_list)},{np.mean(ma_list)},{np.mean(m_iu_list)},{np.mean(fe_iu_list)}'

# Grava o relatório CSV em arquivo.
# -----------------------------------------------------------------------------
exp_file = open('results_acc.csv', 'w')
exp_file.write(acc_str)
exp_file.close()


# TESTE 2
# QUando o conjunto de testes não possui as imagens ground truth (máscaras)
# --> Então, apenas geramos a imagem predita...
# =============================================================================
print('\nTesting model 2...')

modelo_trained.train(False)  # Set model to evaluate mode

# Cria a pasta para os resultados dos experimentos
if not os.path.exists('exp_2'):
    os.makedirs('exp_2')

# Cria a pasta para os resultados dos experimentos
if not os.path.exists('exp_2_pred'):
    os.makedirs('exp_2_pred')

# Contador para imagens do conjunto de testes.
img_idx = 0

# Itera por todos os lotes do conunto de TESTES
for i, (x, img_path) in enumerate(test_dl2):
    
    with torch.no_grad():
        outputs = modelo_trained(x.cuda())

    # Itera por cada imagem deste lote, individualmente.
    ### for j in range(hp_batch_size):
    for j, img_path_ in enumerate(img_path):
        print('=====>')
        print(img_path[j])
    
        # Save images
        plt.imsave(f'exp_2/{img_idx}_input_image.png', batch_to_img(x, j))
        ### plt.imsave(f'exp_2/{img_idx}_pred.png', predb_to_mask(outputs, j), cmap='gray')

        # Imagem original - RGB
        img_rgb = batch_to_img(x, j)
        print(img_rgb.shape, img_rgb.min(), img_rgb.max())
        
        # Imagem predita
        img_pred = predb_to_mask(outputs, j)
        print(type(img_pred), img_pred.shape, img_pred.dtype)

        # Nome do arquivo de saída.
        img_filename = str(img_path[j]).split('/')[-1]
        print(img_filename)

        plt.imsave(f'exp_2_pred/{img_filename}F', img_pred, cmap='gray')

        # Atualiza indice global da imagem
        img_idx += 1

print('\nDONE!')