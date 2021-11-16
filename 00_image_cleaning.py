import os
import pandas as pd
from utils.preprocessing.image_cleaning_utils import *

""" Loop that code each image and check whether it can be cropped """
start_idx = 0
files = os.listdir(str(Path(img_path, '00_all')))
for file in files[start_idx:]:
    df = pd.read_csv(df_path, index_col=[0])
    df.loc[df['imagelink'] == file, 'reviewed'] = True
    try:
        img_color, img_bw, img_plot = load_image(file, '00_all')  # load image
        plot_image(img_plot)

        validity_check = input("Is the image valid?").lower()  # check if image is valid
        if validity_check == 'no':
            df.loc[df['imagelink'] == file, 'valid_image'] = False

        poster = input('Poster?').lower()
        if poster == 'y':
            df.loc[df['imagelink'] == file, 'poster'] = True

        facial_hair = input("Facial hair?").lower()  # check if image has facial hair
        if facial_hair == 'y':
            df.loc[df['imagelink'] == file, 'facial_hair'] = True

        hat = input("Wears hat?").lower()  # check if image has hat
        if hat == 'y':
            df.loc[df['imagelink'] == file, 'wears_hat'] = True

        glasses = input("Wears glasses?").lower()  # check if image has glasses
        if glasses == 'y':
            df.loc[df['imagelink'] == file, 'glasses'] = True

        ethnic = input("Non-danish?").lower()  # check if image is non-daish
        if ethnic == 'y':
            df.loc[df['imagelink'] == file, 'non_danish'] = True

        save_image(img=img_color, fname=file, model='01_image')
        save_image(img=img_bw, fname=file, model='02_image_bw')
        try:  # cropping
            """ Check if its possible to crop the face """
            for idx, image in enumerate([img_color, img_bw]):
                aligned = crop_face(img=image)
                if idx == 0:
                    save_image(img=aligned, fname=file, model='03_cropped')
                else:
                    bluriness = variance_of_laplacian(aligned)
                    save_image(img=aligned, fname=file, model='04_cropped_bw')
        except:
            df.loc[df['imagelink'] == file, 'crop_face'] = False
        try:  # remove background
            """ Check if its possible to remove background """
            idx = 0
            for img in [img_color, img_bw]:
                nobg_img = remove_background(img=img)
                try:
                    aligned = crop_face(img=nobg_img)
                    if idx == 0:
                        save_image(img=aligned, fname=file, model='05_cropped_nobackground')
                    else:
                        save_image(img=aligned, fname=file, model='06_cropped_bw_nobackground')
                    idx += 1
                except:
                    df.loc[df['imagelink'] == file, 'crop_face'] = False
        except:
            df.loc[df['imagelink'] == file, 'remove_background'] = False
        df.to_csv(df_path)
    except:
        df.loc[df['imagelink'] == file, 'valid_image'] = False
        df.to_csv(df_path)

# keep only images that can be cropped
df = pd.read_csv(df_path, index_col=[0])
df = df.loc[df['crop_face']==True]

""" Loop that computes the blurriness of each image """
files = os.listdir(str(Path(img_path, '06_cropped_bw_nobackground')))

#df['blurriness'] = None
for idx, file in enumerate(files):
    if df.loc[df['imagelink']==files[1]].shape[0] > 0:
        try:
            img_color, img_bw, img_plot = load_image(file, '06_cropped_bw_nobackground') # load image
            df.loc[df['imagelink']==file, 'blurriness'] = variance_of_laplacian(img_bw)
        except:
            df.loc[df['imagelink']==file, 'crop_face'] = False
    else:
        continue

""" Save dataframe with metadata about our base sample"""
df = df.loc[df['crop_face']==True, :]
df.drop(['remove_background', 'crop_face', 'reviewed'], axis=1, inplace=True)
df = df.loc[df['blurriness'] >= 100]
df.to_csv(df_path)



