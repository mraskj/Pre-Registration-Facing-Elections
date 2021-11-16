##based on https://github.com/TiesdeKok/fWHR_calculator
import os
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import face_recognition
import urllib.request
import pandas as pd
os.chdir("correct path") # set path


def load_image(path, url=False):
    if not url:
        return face_recognition.load_image_file(path)
    else:
        if path[-3:] == 'jpg' or  path[-3:] == 'peg':
            urllib.request.urlretrieve(path, 'tmp.jpg')
            return face_recognition.load_image_file('tmp.jpg')
        elif path[-3:] == 'png':
            urllib.request.urlretrieve(path, 'tmp.png')
            return face_recognition.load_image_file('tmp.png')
        else:
            print("Unknown image type")

def get_face_points(points, method='average', top='eyebrow'):
    width_left, width_right = points[0], points[16]

    if top == 'eyebrow':
        top_left = points[18]
        top_right = points[25]

    elif top == 'eyelid':
        top_left = points[37]
        top_right = points[43]

    else:
        raise ValueError('Invalid top point, use either "eyebrow" or "eyelid"')

    bottom_left, bottom_right = points[50], points[52]

    if method == 'left':
        coords = (width_left[0], width_right[0], top_left[1], bottom_left[1])

    elif method == 'right':
        coords = (width_left[0], width_right[0], top_right[1], bottom_right[1])

    else:
        top_average = int((top_left[1] + top_right[1]) / 2)
        bottom_average = int((bottom_left[1] + bottom_right[1]) / 2)
        coords = (width_left[0], width_right[0], top_average, bottom_average)

    ## Move the line just a little above the top of the eye to the eyelid
    if top == 'eyelid':
        coords = (coords[0], coords[1], coords[2] - 4, coords[3])

    return {'top_left' : (coords[0], coords[2]),
            'bottom_left' : (coords[0], coords[3]),
            'top_right' : (coords[1], coords[2]),
            'bottom_right' : (coords[1], coords[3])
            }

def good_picture_check(p, debug=False):
    ## To scale for picture size
    width_im = (p[16][0] - p[0][0]) / 100

    ## Difference in height between eyes
    eye_y_l = (p[37][1] + p[41][1]) / 2.0
    eye_y_r = (p[44][1] + p[46][1]) / 2.0
    eye_dif = (eye_y_r - eye_y_l) / width_im

    ## Difference top / bottom point nose
    nose_dif = (p[30][0] - p[27][0]) / width_im

    ## Space between face-edge to eye, left vs. right
    left_space = p[36][0] - p[0][0]
    right_space = p[16][0] - p[45][0]
    space_ratio = left_space / right_space

    if debug:
        print(eye_dif, nose_dif, space_ratio)

    ## These rules are not perfect, determined by trying a bunch of "bad" pictures
    if eye_dif > 5 or nose_dif > 3.5 or space_ratio > 3:
        return False
    else:
        return True

def FWHR_calc(corners):
    width = corners['top_right'][0] - corners['top_left'][0]
    height = corners['bottom_left'][1] - corners['top_left'][1]
    return float(width) / float(height)

def show_box(image, corners):
    pil_image = Image.fromarray(image)
    w, h = pil_image.size

    ## Automatically determine width of the line depending on size of picture
    line_width = math.ceil(h / 100)

    d = ImageDraw.Draw(pil_image)
    d.line([corners['bottom_left'], corners['top_left']], width = line_width)
    d.line([corners['bottom_left'], corners['bottom_right']], width = line_width)
    d.line([corners['top_left'], corners['top_right']], width = line_width)
    d.line([corners['top_right'], corners['bottom_right']], width = line_width)

    imshow(pil_image)

def get_fwhr(image_path, url=False, show=True, method='average', top='eyelid'):
    image = load_image(image_path, url)
    landmarks = face_recognition.api._raw_face_landmarks(image)
    landmarks_as_tuples = [(p.x, p.y) for p in landmarks[0].parts()]

    if good_picture_check(landmarks_as_tuples):
        corners = get_face_points(landmarks_as_tuples, method=method, top = top)
        fwh_ratio = FWHR_calc(corners)

        if show:
            print('The Facial-Width-Height ratio is: {}'.format(fwh_ratio))
            show_box(image, corners)
        else:
            return fwh_ratio
    else:
        if show:
            print("Picture is not suitable to calculate fwhr.")
            imshow(image)
        else:
            return None

#get list of files

red_path = 'pathtofiles'
blue_path = 'D:/Paper/Ansigter, robert/new_repository/Pictures_ideology/test_training_validation_bw_background_no_beard/train/blue/'

red_files = os.listdir(red_path)
red_files_full_path = [red_path + x for x in red_files]

blue_files = os.listdir(blue_path)
blue_files_full_path = [blue_path + x for x in blue_files]

#iterate over lists
red_masculinity = []

for index,file in enumerate(red_files_full_path):
    try:
        red_masculinity.append(get_fwhr(file, url=False, show = False))
    except:
        pass

blue_masculinity = []

for index,file in enumerate(blue_files_full_path):
    try:
        blue_masculinity.append(get_fwhr(file, url=False, show = False))
    except:
        pass



#create dataframe and save
both = pd.DataFrame.from_dict({'probability': blue_masculinity+red_masculinity,
                               'class':len(red_masculinity)*"blue".split()+len(blue_masculinity)*"red".split(),
                               'file_full_path': red_files_full_path+blue_files_full_path,
                               'file_name':red_files+blue_files},
                              orient='index')

both.to_csv("masculinity_prediction.csv")