import pickle
import warnings

import tensorflow as tf

from utils.analysis.create_data_utils import *
from utils.analysis.prediction_utils import *

from sklearn.metrics import zero_one_loss, accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
print(f"Tensorflow version {tf.__version__}")
print(f"{tf.config.list_physical_devices('GPU')}")

# define name of objects
folder = '06_cropped_bw_nobackground'
gender = 'men'
X = 'imagelink'
y = 'elected'

# define hyperparameters
transfer = True
batch_size = 16
lr = 1e-06
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


## SAVE MODEL INFORMATION

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
df.to_csv('model-performance_men.csv')