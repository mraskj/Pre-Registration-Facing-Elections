from pathlib import Path

verbose = True

#base_path = '/home/rask/Desktop/FaceNet'
base_path = '/home/rask/Dropbox/Facing-Politics'
#desk_path = '/home/rask/Desktop/Facing-the-Electorate'
data_path       = str(Path(base_path, 'data'))
img_path        = str(Path(base_path, 'images'))
#df_path         = str(Path(data_path, 'candidates/withpics_reviewed.csv'))
df_path   = str(Path(data_path, 'data.csv'))
landmark_path   = str(Path(base_path, 'utils/shape_predictor_68_face_landmarks.dat'))
checkpoint_path = str(Path(base_path, 'models'))
#bootstrap_path = str(Path(base_path, 'bootstrap'))

train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

men_dir   = 'men'
women_dir = 'women'
all_dir   = 'all'

side_ballot = 'side'
list_ballot = 'list'

checkpoint_string = '_epoch-{epoch:02d}_valacc-{val_acc:.2f}.hdf5'

crop_dim = 224
train_size = 0.7
test_size = 0.5
epochs = 50
target_size = (224,224)
threshold = 0.5
oversampling = .7
patience = 10

label0_desc= 'notelected'
label1_desc = 'elected'
label0_val = 0
label1_val = 1
label0_dir = str(label0_val) + '_' + label0_desc
label1_dir = str(label1_val) + '_' + label1_desc












