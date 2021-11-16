import os
import numpy as np
import pandas as pd
from utils.analysis.models_utils import *
from tensorflow.keras import models

def load_best_model(gender:str, model:str, trainable:bool, batch_size:int, lr:float, dropout:float):
    name = f'{model}_trainable-{trainable}_batch-{batch_size}_lr-{lr}_dropout-{dropout}'
    models = [x for x in os.listdir(str(Path(checkpoint_path, gender, model))) if name in x]
    model_name = sorted(models)[-1]
    return str(Path(checkpoint_path, gender, model, model_name))

def predict(dir:dict, gender:str, model:str, trainable:bool, batch_size:int, lr:float, dropout:float, train:bool):
    path_to_model = load_best_model(gender=gender, model=model, trainable=trainable, batch_size=batch_size, lr=lr, dropout=dropout)
    #print("Load model: ", path_to_model.rsplit('/', 1)[-1])
    model = models.load_model(path_to_model)
    #os.remove(str(path_to_model))

    test_generator = generators(dir=dir, gender=gender, batch_size=batch_size, train=train, prediction=True)
    predictions = model.predict(test_generator, 1)
    predictions = [pred[0] for pred in predictions]

    labels = test_generator.labels
    images = test_generator.filenames
    images = [x.split('/')[1] for x in images]

    pred_df = pd.DataFrame.from_dict({'image': images, 'probability': predictions,
                                      'class': (len(labels) - sum(labels)) * label0_desc.split() + sum(labels) * label1_desc.split(),
                                      'labels': labels})
    pred_df["prediction"] = np.where(pred_df["probability"] >= .5, label1_desc, label0_desc)
    return pred_df
