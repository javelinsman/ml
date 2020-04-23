import os
import tensorflow as tf
from .dataset import KSSDataset, TTSDataLoader
from .model import TTSModel

train_set = KSSDataset(train=True)
val_set = KSSDataset(train=False)
train_loader = TTSDataLoader(train_set, batch_size=64)
val_loader = TTSDataLoader(val_set, batch_size=64)

tts_model = TTSModel()

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=10)

tts_model.model.fit(x=train_loader, epochs=100,
                    validation_data=val_loader, callbacks=[cp_callback])