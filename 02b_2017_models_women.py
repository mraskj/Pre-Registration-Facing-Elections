import os
import pickle
import warnings

import pandas as pd
import tensorflow as tf

from utils.analysis.create_data_utils import *
from utils.analysis.prediction_utils import *

from sklearn.metrics import zero_one_loss, accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
print(f"Tensorflow version {tf.__version__}")
print(f"{tf.config.list_physical_devices('GPU')}")

# define name of objects
folder = '06_cropped_bw_nobackground'
gender = 'women'
X = 'imagelink'
y = 'elected'


#Remember to make dataframe with model performance and then export as csv to plot and make table in R

transfer = True  # note that this means that we fine-tune rather than use VGG16 as feature extractor
batch_size = 16
lr = 1e-5
dropout = .5

name = f'VGG16_trainable-{transfer}_batch-{batch_size}_lr-{lr}_dropout-{dropout}.hdf5'
print(name)

val_bal_acc_list = list()
val_loss_list = list()
train_bal_acc_list = list()
train_loss_list = list()

for e in range(20):
    # erase existing data directories
    erase_files(folder=folder, verbose=False)

    # create new data directories
    model_dir = make_dirs(X=X, y=y, gender=gender, folder=folder, oversample=True, verbose=False)

    # compute appropriate step sizes (should be the same every time within gender)
    step_size_train = len(model_dir[gender]['data']['train']['image']) // batch_size
    step_size_val = len(model_dir[gender]['data']['val']['image']) // batch_size

    # compute total number of samples and number of training samples (should be the same every time within gender)
    n = len(model_dir[gender]['data']['train']['image']) + len(model_dir[gender]['data']['val']['image'])
    r = len(model_dir[gender]['data']['train']['image'])

    # define image generators and callbacks
    train_generator, val_generator = generators(dir=model_dir, gender=gender, batch_size=batch_size)
    callbacks = define_callbacks(model='VGG16', trainable=transfer, gender=gender, batch_size=batch_size,
                                 lr=lr, dropout=dropout)

    # initialize VGG16 model
    model = initialize_network(model='VGG16', dropout=dropout, lr=lr, trainable=transfer)

    # fit the model
    history = model.fit(train_generator, steps_per_epoch=step_size_train, epochs=epochs,
                        validation_data=val_generator, validation_steps=step_size_val,
                        callbacks=callbacks, verbose=verbose)

    # Compute both the training and validation accuracy and balanced accuracy
    for t in [False, True]:
        predict_df = predict(dir=model_dir, model='VGG16', gender=gender, trainable=transfer,
                             batch_size=batch_size,
                             lr=lr, dropout=dropout, train=t)
        if not t:
            val_loss = zero_one_loss(predict_df['class'], predict_df['prediction'])
            val_bal_acc = balanced_accuracy_score(predict_df['class'], predict_df['prediction'])
            val_acc = accuracy_score(predict_df['class'], predict_df['prediction'])

            val_loss_list.append(val_loss)
            val_bal_acc_list.append(val_bal_acc)
        else:
            train_loss = zero_one_loss(predict_df['class'], predict_df['prediction'])
            train_bal_acc = balanced_accuracy_score(predict_df['class'], predict_df['prediction'])
            train_acc = accuracy_score(predict_df['class'], predict_df['prediction'])

            train_loss_list.append(train_loss)
            train_bal_acc_list.append(train_bal_acc)

    name = f'VGG16_trainable-{transfer}_batch-{batch_size}_lr-{lr}_dropout-{dropout}_epoch'
    models = [x for x in os.listdir(str(Path(checkpoint_path, gender, 'VGG16'))) if name in x]
    for file in sorted(models)[:-1]:
        os.remove(str(Path(checkpoint_path, gender, 'VGG16', file)))

    models = [x for x in os.listdir(str(Path(checkpoint_path, gender, 'VGG16'))) if name in x]
    new_name = f'VGG16_trainable-{transfer}_batch-{batch_size}_lr-{lr}_dropout-{dropout}_iteration{e + 1}.hdf5'
    os.rename(str(Path(checkpoint_path, gender, 'VGG16', models[0])),
              str(Path(checkpoint_path, gender, 'VGG16', new_name)))
    print("Validation balanced accuracy: ", val_bal_acc)
    print("Training balanced accuracy: ", train_bal_acc)


# save model information
fpath = str(Path(checkpoint_path, gender, 'val_bal_acc'))
pickle.dump(val_bal_acc_list, open(fpath, "wb"))
val_bal_acc_list = pickle.load(open(fpath, 'rb'))

fpath = str(Path(checkpoint_path, gender, 'val_loss'))
pickle.dump(val_loss_list, open(fpath, "wb"))
val_loss_list = pickle.load(open(fpath, 'rb'))

fpath = str(Path(checkpoint_path, gender, 'train_bal_acc'))
pickle.dump(train_bal_acc_list, open(fpath, "wb"))
train_bal_acc_list = pickle.load(open(fpath, 'rb'))

fpath = str(Path(checkpoint_path, gender, 'train_loss'))
pickle.dump(train_loss_list, open(fpath, "wb"))
train_loss_list = pickle.load(open(fpath, 'rb'))

df = pd.DataFrame([train_loss_list, train_bal_acc_list, val_loss_list, val_bal_acc_list]).T
df.rename(columns={0:'train_loss', 1:'train_bal_acc', 2:'val_loss', 3:'val_bal_acc'}, inplace=True)
df['iteration'] = np.arange(1, len(df)+1)
df.to_csv('model-performance_women.csv')

## VALIDATION

# single best model
model = sorted(val_bal_acc_list)[len(val_bal_acc_list)-1]
model_number = np.where(val_bal_acc_list==model)[0][0]
#print("100th percentile, the best overall model, iteration: ", model_number)
print("100th percentile, the best overall model, balanced validation accuracy: ", np.round(val_bal_acc_list[model_number],4)*100)

# 80th percentile
model = sorted(val_bal_acc_list)[int(np.ceil(len(val_bal_acc_list)*0.8))]
model_number = np.where(val_bal_acc_list==model)[0][0]
#print("80th percentile, the 18th best model, iteration: ", model_number)
print("80th percentile, the 18th model, balanced validation accuracy: ", np.round(val_bal_acc_list[model_number],4)*100)

# 50th percentile (median)
model = sorted(val_bal_acc_list)[int(np.ceil(len(val_bal_acc_list)*0.5))]
model_number = np.where(val_bal_acc_list==model)[0][0]
#print("50th percentile, the 10th best model (i.e. the median), iteration: ", model_number)
print("50th percentile, the 10th model, balanced validation accuracy: ", np.round(val_bal_acc_list[model_number],5)*100)

# 1st percentile (worst model)
model = sorted(val_bal_acc_list)[0]
model_number = np.where(val_bal_acc_list==model)[0][0]
#print("1st percentile, the worst model, iteration: ", model_number)
print("1st percentile, the worst model, balanced validation accuracy: ", np.round(val_bal_acc_list[model_number],5)*100)

# average balanced validation accuracy
print("Average balanced validation accuracy: ", np.round(np.mean(val_bal_acc_list),4)*100)

# average 0-1 validation loss
print("Average 0-1 validation loss: ", np.round(np.mean(val_loss_list), 4))

## TRAINING

# single best model
model = sorted(train_bal_acc_list)[len(train_bal_acc_list)-1]
model_number = np.where(train_bal_acc_list==model)[0][0]
print("100th percentile, the best overall model, iteration: ", model_number)
print("100th percentile, the best overall model, balanced validation accuracy: ", np.round(train_bal_acc_list[model_number],4)*100)

# 80th percentile
model = sorted(train_bal_acc_list)[int(np.ceil(len(train_bal_acc_list)*0.8))]
model_number = np.where(train_bal_acc_list==model)[0][0]
print("80th percentile, the 18th best model, iteration: ", model_number)
print("80th percentile, the 18th model, balanced validation accuracy: ", np.round(train_bal_acc_list[model_number],4)*100)

# 50th percentile (median)
model = sorted(train_bal_acc_list)[int(np.ceil(len(train_bal_acc_list)*0.5))]
model_number = np.where(train_bal_acc_list==model)[0][0]
print("50th percentile, the 10th best model (i.e. the median), iteration: ", model_number)
print("50th percentile, the 10th model, balanced validation accuracy: ", np.round(train_bal_acc_list[model_number],5)*100)

# 1st percentile (worst model)
model = sorted(train_bal_acc_list)[0]
model_number = np.where(train_bal_acc_list==model)[0][0]
print("1st percentile, the worst model, iteration: ", model_number)
print("1st percentile, the worst model, balanced validation accuracy: ", np.round(train_bal_acc_list[model_number],5)*100)

# average balanced training accuracy
print("Average balanced training accuracy: ", np.round(np.mean(train_bal_acc_list),4)*100)

# average 0-1 training loss
print("Average 0-1 training loss: ", np.round(np.mean(train_loss_list), 4))


