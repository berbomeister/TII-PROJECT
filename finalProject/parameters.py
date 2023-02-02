import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
# device = torch.device("cpu")


# # params for nmtmodel
#began training with 0.3 dropout, 0.001 lr -> 0.2 dropout, 0.0001 lr -> 0.2 dropout, 0.0005 lr -> 0.1 dropout, 0.0003 lr // each step trained for 10 epochs
Nx = 4
n_head = 4
d_model = 128
dropout = 0.3


learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 32

maxEpochs = 10
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
