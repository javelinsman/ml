import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision
import torch
from PIL import Image
import numpy as np
from xtts.experiment import Experiment

def fit_to_width(image_matrix, width):
    scale = width / image_matrix.shape[1]
    image = Image.fromarray(image_matrix)
    resized = np.array(image.resize((
        int(image_matrix.shape[1] * scale),
        int(image_matrix.shape[0] * scale)
    )))
    return resized

def normalize(image_matrix, relative):
    cmap = plt.cm.viridis
    if relative:
        norm = plt.Normalize()
    else:
        norm = plt.Normalize(vmin=-5, vmax=5)
    return cmap(norm(image_matrix))

def rgba2rgb(rgba):
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    return np.stack((
        r * a,
        g * a,
        b * a
    ), axis=2)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, model):
        experiment_id = model.logger.log_dir.replace('.', '').replace('/', '-')
        experiment = Experiment(experiment_id)

        data_passed = trainer.callback_metrics['to_callback']
        batch = data_passed['batch']
        (audio_input, text_input), audio_target = batch
        context = model.forward_with_context([audio_input, text_input])
        def ref(key):
            return context[key].cpu().detach().numpy()
        audio_encoded = ref('audio_encoded')
        text_encoded_att = ref('text_encoded_att')
        text_encoded_chr = ref('text_encoded_chr')
        attention = ref('attention')
        input_to_decoder = ref('input_to_decoder')
        audio_decoded = ref('audio_decoded')
        audio_target = audio_target.cpu().detach().numpy()

        step = trainer.global_step
        experiment.add_tensor('audio_target', step, audio_target[0])
        experiment.add_tensor('audio_encoded', step, audio_encoded[0])
        experiment.add_tensor('text_encoded_att', step, text_encoded_att[0])
        experiment.add_tensor('text_encoded_chr', step, text_encoded_chr[0])
        experiment.add_tensor('attention', step, attention[0])
        experiment.add_tensor('input_to_decoder', step, input_to_decoder[0])
        experiment.add_tensor('audio_decoded', step, audio_decoded[0])