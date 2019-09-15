import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from mixed_loss import MixedLoss
import segmentation_models_pytorch as smp
from trainer import Trainer



# eaxmple
# training on fold 0 and size 512
model_1 = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

lr=5e-4
optimizer=optim.Adam(model_1.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
criterion=MixedLoss(10, 2)
epochs=40
bs=16
acc_steps=32//bs
size=512
fold=0
# data
model_trainer_1 = Trainer(model_1, epochs, lr, acc_steps, optimizer, scheduler, criterion, fold, size, bs)
model_trainer_1.start()

