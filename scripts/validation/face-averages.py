import os
import subprocess

#first install http://alyssaq.github.io/face_morpher/

#get top 50 elected and not elected
top_50_blue = both[(both["prediction"]=="elected") & (both["class"]=="elected")].sort_values("probability")[1:50]
top_50_red = both[(both["prediction"]=="not_elected") & (both["class"]=="not_elected")].sort_values("probability", ascending = False)[1:50]

#get path to images
not_cropped_images = "path_to_images"

#get path to images
red_folder_morph = "path1"
blue_folder_morph = "path2"

for redfilename in top_50_red["file_name"]:
    copyfile(os.path.join(not_cropped_images,redfilename), os.path.join(red_folder_morph,redfilename))

for bluefilename in top_50_blue["file_name"]:
    copyfile(os.path.join(not_cropped_images,bluefilename), os.path.join(blue_folder_morph,bluefilename))


#run face averager script
blue = 'python "path_to_averager/averager.py" --images="path_to_images" --out="average.png" --blur --background=transparent'
blue_morph_copy = subprocess.check_output(blue, shell = True )
print(blue_morph_copy)

#####repeat for red######