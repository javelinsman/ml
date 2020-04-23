import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision
import torch
from PIL import Image
import numpy as np

def fit_to_width(image_matrix, width):
    scale = width / image_matrix.shape[1]
    image = Image.fromarray(image_matrix)
    resized = np.array(image.resize((
        int(image_matrix.shape[1] * scale),
        int(image_matrix.shape[0] * scale)
    )))
    return resized

def normalize(image_matrix):
    cmap = plt.cm.viridis
    norm = plt.Normalize()
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
        experiment = model.logger.experiment
        data_passed = trainer.callback_metrics['to_callback']
        experiment.add_image(
            'attention',
            self.image_eval_batch(model, data_passed['batch']),
            model.current_epoch
        )

    def image_eval_batch(self, model, batch):
        width = 1080
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

        splitter = np.zeros((15, width, 4))    
        rgba = np.vstack((
            normalize(fit_to_width(audio_target[0].cpu().detach().numpy(), width)),
            splitter,
            normalize(fit_to_width(audio_decoded[0], width)), splitter,
            normalize(fit_to_width(attention[0], width)), splitter,
            normalize(fit_to_width(audio_encoded[0], width)), splitter,
            normalize(fit_to_width(text_encoded_att[0].T, width)), splitter,
            normalize(fit_to_width(text_encoded_chr[0].T, width)), splitter,
            normalize(fit_to_width(input_to_decoder[0], width)), splitter,        
        ))
        rgb = rgba2rgb(rgba)
        return torch.tensor(rgb).permute(2, 0, 1)
