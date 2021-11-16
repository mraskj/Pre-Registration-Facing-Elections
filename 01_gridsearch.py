import re
import tensorflow as tf

from utils.analysis.create_data_utils import *
from utils.analysis.prediction_utils import *

from sklearn.metrics import balanced_accuracy_score

print(f"Tensorflow version {tf.__version__}")
print(f"{tf.config.list_physical_devices('GPU')}")

folder = '06_cropped_bw_nobackground'
gender = 'women'
X = 'imagelink'
y = 'elected'

N_experiments = 5
batch_size_list = [16, 32, 64]
lr_list = [1e-5, 1e-6]
dropout_list = [0.4, 0.5, 0.6]
model_name_list = ['VGG16']
trainable_list = [False, True]

experiment = []

for n in range(N_experiments):
    grid_dict = {}
    for m in model_name_list:
        for t in trainable_list:
            for b in batch_size_list:
                for l in lr_list:
                    for d in dropout_list:
                        grid_name = f'{m}_train-{t}_batch-{b}_lr-{l}_dropout-{d}'
                        if grid_name not in list(grid_dict.keys()):
                            print(f"Iteration {n}: {grid_name}")
                            erase_files(folder=folder, verbose=False)
                            model_dir = make_dirs(X=X, y=y, gender=gender, folder=folder, oversample=True, verbose=False)
                            step_size_train = len(model_dir[gender]['data']['train']['image']) // b
                            step_size_val = (len(model_dir[gender]['data']['val']['image']) // b)
                            train_generator, val_generator = generators(dir=model_dir, gender=gender, batch_size=b)
                            callbacks = define_callbacks(model=m, trainable=t,gender=gender, batch_size=b, lr=l,
                                                         dropout=d)
                            model = initialize_network(model=m, dropout=d, lr=l, trainable=t)
                            history = model.fit(train_generator,
                                                steps_per_epoch=step_size_train,
                                                epochs=epochs,
                                                validation_data=val_generator,
                                                validation_steps=step_size_val,
                                                callbacks=callbacks,
                                                verbose=False)
                            predict_df = predict(dir=model_dir, model=m, trainable=t, gender=gender, batch_size=b, lr=l,
                                                 dropout=d, train=False)
                            bal_acc = balanced_accuracy_score(predict_df['class'], predict_df['prediction'])
                            print(grid_name, ': ', bal_acc)
                            grid_dict[grid_name] = bal_acc
                            models = [x for x in os.listdir(str(Path(checkpoint_path, gender, m))) if grid_name in x]
                            for file in sorted(models):
                                os.remove(str(Path(checkpoint_path, gender, m, file)))
    experiment.append(grid_dict)

cols = ['model', 'trainable', 'batch', 'lr', 'dropout', 'iteration', 'string', 'acc']
df = pd.DataFrame(columns=cols)

for e in range(len(experiment)):
    items = list(experiment[e].items())
    for i, v in enumerate(items):
        hparams = v[0].split('_')
        name = hparams.pop(0)
        hparams = [h.split('-')[-1] for h in hparams]
        for ih, vh in enumerate(hparams):
            if ih == 0:
                hparams[ih] = vh
            elif ih == 1:
                hparams[ih] = int(vh)
            else:
                hparams[ih] = float(vh)
        row = [name] + hparams + [e, v[0], v[1]]
        row = pd.Series(row, index=df.columns)
        row['lr'] = float(re.findall('lr-(.*)_', row.string)[0])
        df = df.append(row, ignore_index=True)


df.to_csv(str(Path(checkpoint_path, gender, 'model/gridsearch-women.csv')), sep='\t') # REPLACE MEN/WOMEN