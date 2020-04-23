from datasets.dataset import KSSDataset, TTSDataLoader
import random
import torch

train_set = KSSDataset(train=True)
val_set = KSSDataset(train=False)
train_loader = TTSDataLoader(train_set, batch_size=32)
val_loader = TTSDataLoader(val_set, batch_size=32)


def test_mel():
    batch = random.choice(train_loader)
    (mels_left, _), mels_right = batch
    assert torch.all(mels_left[:,1:,:] == mels_right[:,:-1,:]).item() == True
    assert torch.all(mels_left[:,0,:] == 0).item() == True