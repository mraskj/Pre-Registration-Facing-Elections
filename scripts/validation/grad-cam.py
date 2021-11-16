import os
os.chdir("change_to_dir") #change to directory with keras-vis is installed. THis might have changed such that it can simply be imported
from vis.utils import utils

import pandas as pd
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam



#Find prediction layer
layer_idx = utils.find_layer_idx(model, "dense_2") #choose layer

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


#Create separate data frames for classes
both = pd.read_csv("data_on_class_and_probability.csv") #change file name
both["probability"] = both["probability"].astype('float').round(2).astype("str")
red_df = both[both["class"]=="elected"]
blue_df = both[both["class"]=="not_elected"]


#find layer to use
penultimate_layer_idx = utils.find_layer_idx(model, "block5_conv3") #pick layer to do grad cam
red_save_location = "choose_save_location"

#iterate over elected files
for red_file,red_prediction,red_probability, name in zip(
        red_df["file_full_path"],
        red_df["prediction"],
        red_df["probability"],
        red_df["file_name"]):
    f, ax = plt.subplots(1, 1)

    img = utils.load_img(red_file, target_size=(224, 224))
    stacked_img = np.stack((img,)*3, axis=-1)

    grads = visualize_cam(model, layer_idx, filter_indices=0,
                          seed_input=stacked_img, backprop_modifier='guided', penultimate_layer_idx=penultimate_layer_idx)

    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255 )

    ax.set_title(name + "/" + "red" + "/" + red_prediction +"/" + red_probability)
    ax.imshow(overlay(jet_heatmap, stacked_img))
    plt.savefig(red_save_location+name)

#####repeat for not elected#####

#create average
top_50_red = both[(both["prediction"]=="red") & (both["class"]=="red")].sort_values("probability", ascending = False)[1:50]
all_grads_red = []

for red_file,red_prediction,red_probability, name in zip(
        top_50_red["file_full_path"][0:],
        top_50_red["prediction"][0:],
        top_50_red["probability"][0:],
        top_50_red["file_name"][0:]):
    f, ax = plt.subplots(1, 1)
    print(name)
    img = utils.load_img(red_file, target_size=(224, 224))
    stacked_img = np.stack((img,)*3, axis=-1)

    grads = visualize_cam(model, layer_idx, filter_indices=0,
                          seed_input=stacked_img, backprop_modifier='guided', penultimate_layer_idx=penultimate_layer_idx)



    all_grads_red.append(grads)
    #ax.imshow(overlay(jet_heatmap, stacked_img))

#create average grad cam
f, ax = plt.subplots(1, 1)

img = utils.load_img(top_50_blue.iloc[0]["file_full_path"], target_size=(224, 224)) #load example image.
stacked_img = np.stack((img,)*3, axis=-1)

jet_heatmap = np.uint8(cm.jet(np.mean(all_grads, axis = 0))[..., :3] * 255 )
ax.imshow(overlay(jet_heatmap, stacked_img))
plt.savefig(r"save_path\average_blue1.jpg")

#####repeat for blue###