import multiprocessing
import os, logging
import PIL
import json
import numpy as np
from PIL import Image as I
from shapely.geometry import Point, Polygon
import cv2
import pickle
import math
from scipy import stats
from tqdm import tqdm
import matplotlib.path as mpltPath

# ====================Read Images and Labels====================

Image_List = []
Image_Dir = None
# Label Dict format - {image_filename:[Polygon Tuples]}
Label = {}
setting = {"original":True, "norm":True}

# Read and Verify the Images
# Input:    Image Directory Path
# Output:   Image Directory Path, Image Lists
def checkImages(path):
    global Image_List, Image_Dir
    Image_Dir = os.path.realpath(path)
    Image_List = []
    for file in os.listdir(Image_Dir):
        i = os.path.join(Image_Dir, file)
        if os.path.isdir(i):
            if not i.endswith(".images_temp"):
                logging.warning("Found a directory ({0}). and has ignored it.".format(i))
            continue
        try:
            I.open(i)
            Image_List.append(file)
        except PIL.UnidentifiedImageError:
            logging.warning("{0} cannot be opened and we has ignored it.".format(i))

    return Image_Dir, Image_List


# Add Polygon into MLMonkey
# Input:    Image Filename
#           Polygon Tuples
# Output:   Current Label Dictionary
def addLabel(image_filename, polygon, types=-1):
    global Label
    if image_filename not in Label.keys():
        Label[image_filename] = {}
        Label[image_filename]["p"] = []
        Label[image_filename]["t"] = []
    Label[image_filename]["p"].append(polygon)
    Label[image_filename]["t"].append(types)
    return Label


# Initialise the Label Dictionary
def initLabel():
    global Label
    Label = {}


# Convert the Split Polygon X and Y to a list of Tuple
# Input:    X coordinates - List or Numpy
#           Y coordinate - List or Numpy
# Output:   Numpy List
def convertPolytoTuple(x, y):
    if len(x) != len(y):
        raise ValueError(
            "The X and Y coordinates do not have same length (X has got {0} & Y has got {1})".format(len(x), len(y)))
    return np.array([(x[i], y[i]) for i in range(len(x))])


# Load VIA Labels
# Input:    JSON File Path
# Output:   Label Dictionary
def readVIALabel(file, init=False, test_gen=False):
    global Label
    if init:
        Label = {}
    with open(os.path.abspath(file)) as f:
        json_label = json.loads(f.read())
        f.close()
    for i in json_label.values():
        if test_gen:
            if i['filename'].split("/")[-1] in Image_List:
                for j in i['regions']:
                    addLabel(i['filename'].split("/")[-1], convertPolytoTuple(j['shape_attributes']['all_points_x'],
                                                                              j['shape_attributes']['all_points_y']))
        else:
            if i['filename'] in Image_List:
                for j in i['regions']:
                    addLabel(i['filename'], convertPolytoTuple(j['shape_attributes']['all_points_x'],
                                                               j['shape_attributes']['all_points_y']))
    return Label

def readLabel(data, init=False):
    global Label
    if init:
        Label = {}

    for i in data:
        if i in Image_List:
            if len(data[i]["polygons"]) < 1:
                logging.warning("Image " + i + " does not have label, skipped.")
                continue
            for j in range(len(data[i]["polygons"])):
                if "type" in data[i]:
                    addLabel(i, data[i]["polygons"][j], data[i]["type"][j])
                else:
                    addLabel(i, data[i]["polygons"][j])
        else:
            logging.warning("Image " + i + " is not found, skipped.")

    return Label

# Check data is correctly loaded
# Process: Logging the error
def check_data():
    global Label
    if len(Label.keys()) <= 0:
        logging.error("Label is not correctly loaded")


# ====================Preprocessing====================

Label_ = {}


# Calculate Bounding Box Area of a polygon in images size
# Input:    List of Tuples - Polygon
# Output:   (Left, Top, Right, Bottom)
def cal_boundary(polygon):
    min_x = min([i[0] for i in polygon])
    min_y = min([i[1] for i in polygon])
    max_x = max([i[0] for i in polygon])
    max_y = max([i[1] for i in polygon])
    return [min_x, min_y, max_x, max_y]


# Crop the defect area and stored into a temp file
# Store the cropped polygon data into dictionary
# Input:    Image ID
#           Crop Padding - Unit: pixels
def crop(did, pad, single=None, poly=None):
    global Label_
    img = None
    if single is not None:
        img = I.open(os.path.realpath(single))
    else:
        img = I.open(os.path.join(Image_Dir, Label_[did]['filename']))

    if poly is None:
        poly = Label_[did]['polygon']

    boundary = cal_boundary(poly)
    padX = int((boundary[2] - boundary[0]) * pad)
    padY = int((boundary[3] - boundary[1]) * pad)
    if padX <= 2:
        padX = 5
    if padY <= 2:
        padY = 5
    boundaryF = [0, 0, 0, 0]
    boundaryF[0] = boundary[0] - padX
    boundaryF[1] = boundary[1] - padY
    boundaryF[2] = boundary[2] + padX
    boundaryF[3] = boundary[3] + padY
    offset = [boundary[0] - padX, boundary[1] - padY]
    if boundaryF[0] < 0:
        offset[0] = 0
        boundaryF[0] = 0
    if boundaryF[1] < 0:
        offset[1] = 0
        boundaryF[1] = 0
    if boundaryF[2] >= img.size[0]:
        boundaryF[2] = img.size[0] - 1
    if boundaryF[3] >= img.size[1]:
        boundaryF[3] = img.size[1] - 1
    cropped_img = img.crop(boundaryF)
    if did is None:
        cropped_img.save(os.path.join("./.images_temp", "-1") + ".jpg")
    else:
        cropped_img.save(os.path.join("./.images_temp", str(did)) + ".jpg")
    if did is not None:
        Label_[did]["crop_poly"] = np.array([(i[0] - offset[0], i[1] - offset[1]) for i in poly])
        Label_[did]["boundary"] = (boundary, [img.width, img.height])
        Label_[did]["width"] = (boundary[3] - boundary[1])
        Label_[did]["height"] = (boundary[2] - boundary[0])
        return None
    else:
        return (boundary, [img.width, img.height]), (boundary[3] - boundary[1]), (boundary[2] - boundary[0]), np.array([(i[0] - offset[0], i[1] - offset[1]) for i in poly])

# Load Image
# Input: Image directory path
# Output: Image array
def loadImage(path):
    img = I.open(path)
    return np.array(img)


# Convert image to map, which determine each pixel is inside of polygon or not
# Input: Image shape,
#        Polygon coordinates list
# Output: Map array
def cal_map(shape, poly):
    p = np.array([[i[1], i[0]] for i in poly])

    # Method 1: 4-5s/img
    # poly = Polygon(p)
    # poly_map = np.array([[poly.contains(Point((w, h))) for w in range(shape[1])] for h in range(shape[0])]).astype(bool)

    # Method 2 : 3-4s/img
    # poly = Polygon(p)
    # w, h = np.meshgrid(range(shape[1]), range(shape[0]))
    # coor_l = list(np.dstack((h, w)).reshape((shape[0] * shape[1], 2)))
    # poly_map = np.array(list(map(lambda x: poly.contains(Point(x)), coor_l))).reshape((shape[0], shape[1]))

    # Method 3: 2.5-3.5s/img
    # poly = Polygon(p)
    # w, h = np.meshgrid(range(shape[1]), range(shape[0]))
    # coor = np.dstack((h, w)).reshape((shape[0] * shape[1], 2))
    # poly_map = np.array(list(map(poly.contains, list(map(Point, coor))))).reshape((shape[0], shape[1]))

    # Method 4: 0.18-0.2s/img
    # poly = mpltPath.Path(p)
    # w, h = np.meshgrid(range(shape[1]), range(shape[0]))
    # coor = np.dstack((h, w)).reshape((shape[0] * shape[1], 2))
    # poly_map = poly.contains_points(coor).reshape((shape[0], shape[1]))

    # Method 5: 0.06-0.08s/img
    poly_map = np.zeros([shape[0], shape[1]])
    cv2.drawContours(poly_map, [poly], -1, 2, -1)
    poly_map = poly_map >= 1

    return poly_map


# Convert RGB to different colour mode
# Input: Image array, mode=(HSV, HLS, LAB)
# Output: Image array
def RGBConvert(img_array, mode="HSV"):
    if mode == "HSV" or mode == "HSB":
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    elif mode == "HLS":
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS)
    elif mode == "LAB":
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    else:
        raise ValueError("Non Supported Mode: {0}".format(mode))


# Calculate bounding box size
# Input: Polygon coordinate list:List
# Output: Bounding box area:Integer
def cal_bb_size(poly):
    boundary = cal_boundary(poly)
    return int((boundary[2] - boundary[0]) * (boundary[3] - boundary[1]))


# Calculate polygon size
# Input: Polygon map: Array
# Output: Polygon size: Integer
def cal_poly_size(poly_map):
    return np.count_nonzero(poly_map)


# Calculate distance between two polygons
# Input: Polygon 1: List
#        Polygon 2: List
# Output: Distance: Integer
def cal_distance(poly1, poly):
    distance = []
    for i in poly:
        if np.all(i != poly1):
            try:
                distance.append(Polygon(poly1).distance(Polygon(i)))
            except:
                continue
    return distance


# Calculate shape complexity
# Input: Polygon coordinate list
# Output: Angle List:Array(Angle),
#         Number of Edge:Integer,
#         Bounding length list:List(Length)
def cal_shape(poly):
    polygon = np.append([poly[-1]], poly, axis=0)
    polygon = np.append(polygon, [poly[0]], axis=0)
    edge = 0
    degree = []
    length = []
    turning = []
    for i in range(1, len(polygon) - 1):
        c = math.dist(polygon[i - 1], polygon[i + 1])
        a = math.dist(polygon[i - 1], polygon[i])
        b = math.dist(polygon[i], polygon[i + 1])
        if a <= 0 or b <= 0:
            continue

        angle = math.degrees(math.acos(round((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), 5)))
        length.append((a, b))
        degree.append(round(angle))
        goin_yd = polygon[i][-1] - polygon[i - 1][-1]
        goin_xd = polygon[i][0] - polygon[i - 1][0]
        if goin_xd == 0:
            # vertical direction
            if goin_yd >= 0:
                # going down
                if polygon[i + 1][0] <= polygon[i][0]:
                    # turn right
                    turning.append(180 - angle)
                else:
                    # turn left
                    turning.append(-180 + angle)
            else:
                # going up
                if polygon[i + 1][0] < polygon[i][0]:
                    # turn left
                    turning.append(-180 + angle)
                else:
                    # turn right
                    turning.append(180 - angle)
        else:
            # non vertical direction
            goin_a = goin_yd / goin_xd
            goin_b = polygon[i][-1] - goin_a * polygon[i][0]
            inter_y = goin_a * polygon[i + 1][0] + goin_b
            if goin_xd >= 0:
                # going right
                if polygon[i + 1][-1] >= inter_y:
                    # turn right
                    turning.append(180 - angle)
                else:
                    # turn left
                    turning.append(-180 + angle)
            else:
                # going left
                if polygon[i + 1][-1] > inter_y:
                    # turn left
                    turning.append(-180 + angle)
                else:
                    # turn right
                    turning.append(180 - angle)
        if angle < 170:
            edge += 1
    turning = np.array(turning).astype(int)
    return np.array(degree), edge, np.array(length), np.array(turning)


# Convert the Labels to ID based Data
# Input:    Colour Mode - Default: HSV/HSB
#           crop_pad (Optional) - Default: 0 - crop extra outside area with percentage
#           save (Optional) - If you want to save the loaded data, please overwrite the saving directory
#           reload (Optional) - Loading the stored data
# Output:   Label_reID Dictionary - Result is stored in Label_reID
def loadData(color_mode="HSV", crop_pad=0, size_percent=False, save=None, reload=None):
    global Label_, Label
    check_data()
    Label_ = {}
    count = 1
    if reload is not None:
        with open(os.path.abspath(reload), 'rb') as f:
            Label_ = pickle.load(f)
            f.close()

        return Label_



    for i in tqdm(Label.keys(), desc="Loading Images"):
        count2 = 1
        for j in Label[i]["p"]:
            did = str(count) + "_" + str(count2)
            Label_[did] = {'filename': i, 'iid': count, 'did': count2, 'polygon': j, "type": Label[i]["t"][count2-1]}
            crop(did, crop_pad)
            img_array = loadImage(os.path.join("./.images_temp", str(did) + ".jpg"))
            Label_[did]["img_arr"] = img_array
            Label_[did]["map"] = cal_map(Label_[did]["img_arr"].shape, Label_[did]["crop_poly"])
            co, hi = cv2.findContours(Label_[did]["map"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_TC89_KCOS)
            if len(co) <= 0:
                Label_[did]["re_poly"] = Label_[did]["crop_poly"]
            else:
                Label_[did]["re_poly"] = co[0].reshape([co[0].shape[0], co[0].shape[-1]])
            if Label_[did]["re_poly"].shape[0] < 3:
                if Label_[did]["crop_poly"].shape[0] >= 3:
                    Label_[did]["re_poly"] = Label_[did]["crop_poly"]
                else:
                    Label_[did]["re_poly"] = np.array([[4,5], [5,6], [6,5], [5,4]])
            Label_[did]["img_arr_mode"] = RGBConvert(Label_[did]["img_arr"], mode=color_mode)
            Label_[did]["bb_size"] = cal_bb_size(Label_[did]["crop_poly"])
            if Label_[did]["bb_size"] <= 0:
                Label_[did]["bb_size"] = 1
            Label_[did]["poly_size"] = cal_poly_size(Label_[did]["map"])
            if size_percent:
                Label_[did]["bb_size"] /= Label_[did]["boundary"][1][0] * Label_[did]["boundary"][1][1]
                Label_[did]["poly_size"] /= Label_[did]["boundary"][1][0] * Label_[did]["boundary"][1][1]
            Label_[did]["hue"] = Label_[did]["img_arr_mode"][:, :, 0]
            Label_[did]["sat"] = Label_[did]["img_arr_mode"][:, :, 1]
            Label_[did]["value"] = Label_[did]["img_arr_mode"][:, :, 2]
            Label_[did]["neighbour_dist"] = cal_distance(Label_[did]["polygon"], Label[i])

            Label_[did]["degree"], Label_[did]["edge"], Label_[did]["edge_len"], Label_[did]["turning"] = \
                cal_shape(Label_[did]["re_poly"])

            count2 += 1

        count += 1

    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(Label_, f)
            f.close()

    return Label_



# ====================Feature Extraction====================


feature_list = []


# Split values into group
# Input: Value,
#        Group_boundary
# Output: Grouped_value:Integer
def cal_group(value, group):
    if value <= group[0]:
        return 1
    elif group[0] < value <= group[1]:
        return 2
    elif group[1] < value <= group[2]:
        return 3
    elif group[2] < value <= group[3]:
        return 4
    else:
        return 5


# Calculate group boundary
# Input: Array:Array(value),
#        Boundary gap:Float
# Output: Group list:List
def group(arr, gap=0.30):
    s = sorted(arr)
    l = [int(len(s) * i / 5) for i in range(1, 5)]
    group = []
    for i in l:
        index = (i, 0)
        for j in range(i - round(gap * l[0] / 2), i + round(gap * l[0] / 2)):
            if s[j + 1] - s[j] >= index[-1]:
                index = (j, s[j + 1] - s[j])
        group.append(index[0])
    return [s[i] for i in group]


# Group the Angles
# Input: Angle_List
# Output: Average angle:Integer
#         Mode angle:Integer
def cal_deg(deg, original, norm):

    if original:
        if norm:
            return np.average(deg) / 180, int(stats.mode(deg, axis=None)[0]) / 180
        else:
            return np.average(deg), int(stats.mode(deg, axis=None)[0])

    d = deg / 30
    d = np.round(d).astype(int)
    avg_d = int(round(np.average(deg) / 30))
    mode_d = int(stats.mode(d, axis=None)[0])
    if avg_d > 6:
        avg_d = 6

    if norm:
        avg_d /= 6
        mode_d /= 6

    return avg_d, mode_d


def cal_shape_comp(turning, deg, edge, original, norm):
    t = np.append([turning[-1]], turning, axis=0)
    total_len = sum(edge[:, 0])
    edge_per = edge / total_len
    edge_per = np.append([edge_per[-1]], edge_per, axis=0)
    follow_turn = 0
    reverse_turn = 0
    small_turn = 0
    edge_ratio = 0
    sc_score = 0
    for i in range(1, len(t)):

        turn_v = abs(t[i] - t[i - 1]) / 360

        if (t[i] > 0 and t[i - 1] < 0) or (t[i] < 0 and t[i - 1] > 0):
            reverse_turn += 1
            follow_turn -= 1 - turn_v

        else:
            follow_turn += 1 - turn_v
        if abs(t[i]) >= 90:
            small_turn += 1
        er = 1 / (max(edge_per[i]) / min(edge_per[i]))
        edge_ratio += er
        sc_score += (abs(t[i]) / 180) * er

    follow_turn /= len(turning)
    edge_ratio /= len(turning)
    sc_score /= len(turning)
    if follow_turn > 1:
        follow_turn = 1
    if follow_turn < 0:
        follow_turn = 0
    reverse_turn /= len(turning)
    small_turn /= len(turning)

    if norm:
        return edge_ratio, follow_turn, reverse_turn, small_turn, sc_score
    else:
        return round(edge_ratio * 10), round(follow_turn * 10), round(reverse_turn * 10), round(small_turn * 10), round(
        sc_score * 10)


# Calculate and group coverage of polygon in bounding box
# Input: Polygon coordinate list
#        Bounding
# Output: Coverage:Float - the percentage of polygon in bounding box
def cal_coverage(poly, bb, original, norm):
    if original:
        if norm:
            return float(poly / bb)
        else:
            return float(poly / bb)
    cvg = round((poly / bb) * 5)

    if norm:
        cvg /= 5
    return cvg


# Calculate and group aspect ratio of bounding box
# Input: Polygon
# Output: Aspect Ratio:Integer - Calculate the ratio between long side and short side by long side / short side,
#         then round it into the closest integer
def cal_asp_ratio(poly, original, norm):
    boundary = cal_boundary(poly)

    if (boundary[3] - boundary[1]) <= 0 or (boundary[2] - boundary[0]) <= 0:
        ratio = 1
    else:
        ratio = 1 - min([(boundary[2] - boundary[0]) / (boundary[3] - boundary[1]),
                         (boundary[3] - boundary[1]) / (boundary[2] - boundary[0])])

    if original:
        if norm:
            return ratio
        else:
            return ratio

    a_ratio = round(ratio * 10)
    if norm:
        a_ratio /= 10

    return a_ratio


# Group neighbour distance
# Input: Neighbour distance list
#        Threshold - short distance threshold
# Output: Grouped Distance - 1: short, 2: long, 3: No neighbour
def cal_dist(dist, threshold, original, norm):
    if threshold == 0:
        if len(dist) == 0:
            if norm:
                return 1
            else:
                return 1
        else:
            if norm:
                return 0
            else:
                return 0
    if len(dist) == 0:
        if norm:
            return 1
        else:
            return 2
    else:
        short = min(dist)
        if short <= threshold:
            if norm:
                return 0
            else:
                return 0
        else:
            if norm:
                return 0.5
            else:
                return 1


# Group hue values - Hue is degree-based value, 0 = 360, so max range is 180
# Input: Hue Array
#        Outside of polygon:Boolean - group polygon inside or outside of hue
#        hue map:Array- polygon map
# Output: Average Hue:Integer
#         Mode Hue:Integer
#         Hue Range:Integer - range of min and max Hue values
#         Unique Hue:Integer - unique number of Hue
def cal_hue(hue, out, hmap, original, norm):
    def hue_range(hu, ori):
        hue_r = 0
        hue_set = list(set(hu))
        for i in hue_set:
            max_r = 0
            if ori:
                max_r = max(list(map(lambda x: min([abs(i - x), abs(i + 179 - x), abs(x + 179 - i)]), hue_set)))
            else:
                max_r = max(list(map(lambda x: min([abs(i - x), abs(i + 12 - x), abs(x + 12 - i)]), hue_set)))

            if max_r > hue_r:
                hue_r = max_r
        return hue_r
    h = None

    if out:
        h = hue[np.logical_not(hmap)]
    else:
        h = hue[hmap]
    h = h.astype(float)
    h = h + 1


    if original:
        distribution = {}
        for i in range(1, 181):
            distribution[i] = 0
        if len(h) <= 0:
            return 0, 0, 0, 0, distribution
        for v, c in list(zip(stats.find_repeats(h).values, stats.find_repeats(h).counts)):
            distribution[int(v)] = round(c / len(h), 2)
        if norm:
            return np.average(h) / 180, int(stats.mode(h, axis=None)[0]) / 180, int(hue_range(h, True)) / 90, \
                   len(np.unique(h)) / 180, distribution

        else:
            return np.average(h), int(stats.mode(h, axis=None)[0]), int(hue_range(h, True)), len(np.unique(h)), \
                   distribution



    avg_h = np.average(h)
    if avg_h >= 172.5:
        avg_h = 1
    else:
        avg_h = int((avg_h + 22.5) / 15)

    h = (h + 22.5) / 15
    h[h >= 13] = 1
    h = h.astype(int)
    distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

    if len(h) <= 0:
        return 0, 0, 0, 0, distribution

    for v, c in list(zip(stats.find_repeats(h).values, stats.find_repeats(h).counts)):
        distribution[int(v)] = c / len(h)

    if norm:
        return avg_h / 12, int(stats.mode(h, axis=None)[0]) / 12, int(hue_range(h, False)) / 6, \
               len(np.unique(h)) / 12, distribution

    return avg_h, int(stats.mode(h, axis=None)[0]), int(hue_range(h, False)), len(np.unique(h)), distribution


# Group saturation or brightness values
# Input: Saturation or brightness Array
#        Outside of polygon:Boolean - group polygon inside or outside of hue
#        hue map:Array- polygon map
# Output: Average Saturation or brightness:Integer
#         Mode Saturation or brightness:Integer
#         Saturation or brightness Range:Integer - range of min and max Saturation or brightness values
#         Unique Saturation or brightness:Integer - unique number of Saturation or brightness
def cal_sat_brt(sat_brt, out, hmap, original, norm):
    if out:
        sb = sat_brt[np.logical_not(hmap)]
    else:
        sb = sat_brt[hmap]

    if original:
        distribution = {}
        for i in range(256):
            distribution[i] = 0

        if len(sb) <= 0:
            return 0, 0, 0, 0, distribution
        for v, c in list(zip(stats.find_repeats(sb).values, stats.find_repeats(sb).counts)):
            distribution[int(v)] = c / len(sb)
        if norm:
            return np.average(sb) / 255, int(stats.mode(sb, axis=None)[0]) / 255, int(max(sb) - min(sb)) / 255, \
                   len(np.unique(sb)) / 255, distribution
        else:
            return np.average(sb), int(stats.mode(sb, axis=None)[0]), int(max(sb) - min(sb)), len(np.unique(sb)), \
                   distribution


    avg_sb = np.average(sb)

    if avg_sb >= 255:
        avg_sb = 5
    else:
        avg_sb = round((avg_sb / 255) * 5)
    sb = sb.astype(float)
    sb = np.round(sb / 255 * 5)
    sb[sb >= 6] = 5
    sb = sb.astype(int)
    distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    if len(sb) <= 0:
        return 0, 0, 0, 0, distribution

    for v, c in list(zip(stats.find_repeats(sb).values, stats.find_repeats(sb).counts)):
        distribution[int(v)] = round(c / len(sb), 2)


    if norm:
        return avg_sb / 5, int(stats.mode(sb, axis=None)[0]) / 5, int(max(sb) - min(sb)) / 5, len(np.unique(sb)) / 5, \
               distribution

    return avg_sb, int(stats.mode(sb, axis=None)[0]), int(max(sb) - min(sb)), len(np.unique(sb)), distribution


# Colour Complexity (Further work)
def cal_comp_colour(hue_dist, sat_dist, brt_dist, out_hue_dist, out_sat_dist, out_brt_dist, original, norm):
    hue_outin = 0
    sat_outin = 0
    brt_outin = 0

    for i in hue_dist:
        hue_outin += abs(hue_dist[i] - out_hue_dist[i])

    for i in sat_dist:
        sat_outin += abs(sat_dist[i] - out_sat_dist[i])

    for i in brt_dist:
        brt_outin += abs(brt_dist[i] - out_brt_dist[i])
    if original:
        if norm:
            hue_outin = hue_outin / 2
            sat_outin = sat_outin / 2
            brt_outin = brt_outin / 2
        else:
            hue_outin = hue_outin
            sat_outin = sat_outin
            brt_outin = brt_outin
    else:
        if norm:
            hue_outin = round(hue_outin / 2)
            sat_outin = round(sat_outin / 2)
            brt_outin = round(brt_outin / 2)
        else:
            hue_outin = round(hue_outin / 2 * 10)
            sat_outin = round(sat_outin / 2 * 10)
            brt_outin = round(brt_outin / 2 * 10)

    return hue_outin, sat_outin, brt_outin


def cal_size_edge(value, original, norm, norm_range=None):

    if original:
        if norm:
            if norm_range is None:
                raise ValueError("norm_range is None, expect a dictionary contain min, max values of the dataset")

            if float((value - norm_range["min"]) / (norm_range["max"] - norm_range["min"])) > 1:
                return 1
            elif float((value - norm_range["min"]) / (norm_range["max"] - norm_range["min"])) < 0:
                return 0
            return float((value - norm_range["min"]) / (norm_range["max"] - norm_range["min"]))
        else:
            return value

    ans = round(10 * (value - norm_range["min"]) / (norm_range["max"] - norm_range["min"]))
    if ans > 10:
        ans = 10

    if norm:
        ans /= 10

    return ans

def norm_data(data):
    return {"min": min(data), "max": max(data)}


# Extract feature Function
# Input:    Label_reID
# Output:   List - Store all features
def featureExtract(outside="mode", distance_threshold=100, shape_detail="full", colour_detail="full", meta_detail=True,
                   norm=True, original=True, save=None, reload=None, size_norm=None, edge_norm=None):
    global Label_, feature_list, setting

    setting["original"] = original
    setting["norm"] = norm
    feature_list = []
    if outside is not None:
        if outside == "average":
            feature_list.extend(["out_hue_avg", "out_sat_avg", "out_brt_avg"])
        elif outside == "mode":
            feature_list.extend(["out_hue_mode", "out_sat_mode", "out_brt_mode"])
        elif colour_detail == "unique":
            feature_list.extend(["out_hue_uni", "out_sat_uni", "out_brt_uni"])
        elif outside == "full":
            feature_list.extend(["out_hue_avg", "out_hue_mode", "out_hue_range", "out_hue_uni",
                                 "out_sat_avg", "out_sat_mode", "out_sat_range", "out_sat_uni",
                                 "out_brt_avg", "out_brt_mode", "out_brt_range", "out_brt_uni"])
        else:
            raise AttributeError('Cannot recognise outside attribute. Expect ("mode", "average", "full", None')
    if meta_detail:
        feature_list.extend(["size", "asp_ratio", "distance"])
    if shape_detail is not None:
        if shape_detail == "basic":
            feature_list.extend(["coverage", "deg_avg", "deg_mode", "edge"])
        if shape_detail == "complex":
            feature_list.extend(["sc_edge_ratio", "sc_follow_turn", "sc_reverse_turn", "sc_small_turn"])
        if shape_detail == "full":
            feature_list.extend(["coverage", "deg_avg", "deg_mode", "edge"])
            feature_list.extend(["sc_edge_ratio", "sc_follow_turn", "sc_reverse_turn", "sc_small_turn"])
    else:
        raise AttributeError('Cannot recognise shape detail attribute. Expect ("basic", "complex", "full")')
    if colour_detail is not None:
        if colour_detail == "average":
            feature_list.extend(["hue_avg", "sat_avg", "brt_avg"])
        if colour_detail == "mode":
            feature_list.extend(["hue_mode", "sat_mode", "brt_mode"])
        if colour_detail == "unique":
            feature_list.extend(["hue_uni", "sat_uni", "brt_uni"])
        if colour_detail == "complex":
            feature_list.extend(["hue_outin", "sat_outin", "brt_outin"])
        if colour_detail == "full":
            feature_list.extend(["hue_avg", "hue_mode", "hue_range", "hue_uni",
                                 "sat_avg", "sat_mode", "sat_range", "sat_uni",
                                 "brt_avg", "brt_mode", "brt_range", "brt_uni", "hue_outin", "sat_outin", "brt_outin"])
    else:
        raise AttributeError(
            'Cannot recognise outside attribute. Expect ("unique", "complex", "mode", "average", "full")')
    if reload is not None:
        with open(os.path.abspath(reload), 'rb') as f:
            Label_ = pickle.load(f)
            f.close()
        return Label_
    norm_size = norm_data([Label_[i]["poly_size"] for i in Label_])
    norm_edge = norm_data([Label_[i]["edge"] for i in Label_])
    if size_norm is not None:
        norm_size = size_norm
    if edge_norm is not None:
        norm_edge = edge_norm
    for did in tqdm(Label_.keys(), desc="Feature Extraction"):
        Label_[did]["norm_size"] = norm_size
        Label_[did]["norm_edge"] = norm_edge
        Label_[did]["size"] = cal_size_edge(Label_[did]["poly_size"], original, norm, norm_size)
        Label_[did]["coverage"] = cal_coverage(Label_[did]["poly_size"], Label_[did]["bb_size"], original, norm)
        Label_[did]["asp_ratio"] = cal_asp_ratio(Label_[did]["re_poly"], original, norm)
        Label_[did]["deg_avg"], Label_[did]["deg_mode"] = cal_deg(Label_[did]["degree"], original, norm)
        Label_[did]["edge"] = cal_size_edge(Label_[did]["edge"], original, norm, norm_edge)
        Label_[did]["sc_edge_ratio"], Label_[did]["sc_follow_turn"], Label_[did]["sc_reverse_turn"], \
        Label_[did]["sc_small_turn"], Label_[did]["sc_score"] = cal_shape_comp(Label_[did]["turning"],
                                                                               Label_[did]["degree"],
                                                                               Label_[did]["edge_len"], original, norm)
        Label_[did]["distance"] = cal_dist(Label_[did]["neighbour_dist"], distance_threshold, original, norm)
        Label_[did]["hue_avg"], Label_[did]["hue_mode"], Label_[did]["hue_range"], Label_[did]["hue_uni"], \
        Label_[did]["hue_dist"] = cal_hue(Label_[did]["hue"], False, Label_[did]["map"], original, norm)
        Label_[did]["sat_avg"], Label_[did]["sat_mode"], Label_[did]["sat_range"], Label_[did]["sat_uni"], \
        Label_[did]["sat_dist"] = cal_sat_brt(Label_[did]["sat"], False, Label_[did]["map"], original, norm)
        Label_[did]["brt_avg"], Label_[did]["brt_mode"], Label_[did]["brt_range"], Label_[did]["brt_uni"], \
        Label_[did]["brt_dist"] = cal_sat_brt(Label_[did]["value"], False, Label_[did]["map"], original, norm)
        Label_[did]["out_hue_avg"], Label_[did]["out_hue_mode"], Label_[did]["out_hue_range"], Label_[did][
            "out_hue_uni"], Label_[did]["out_hue_dist"] = cal_hue(Label_[did]["hue"], True, Label_[did]["map"],
                                                                  original, norm)
        Label_[did]["out_sat_avg"], Label_[did]["out_sat_mode"], Label_[did]["out_sat_range"], Label_[did][
            "out_sat_uni"], Label_[did]["out_sat_dist"] = cal_sat_brt(Label_[did]["sat"], True, Label_[did]["map"],
                                                                      original, norm)
        Label_[did]["out_brt_avg"], Label_[did]["out_brt_mode"], Label_[did]["out_brt_range"], Label_[did][
            "out_brt_uni"], Label_[did]["out_brt_dist"] = cal_sat_brt(Label_[did]["value"], True, Label_[did]["map"],
                                                                      original, norm)
        Label_[did]["hue_outin"], Label_[did]["sat_outin"], Label_[did]["brt_outin"] = cal_comp_colour(
            Label_[did]["hue_dist"], Label_[did]["sat_dist"], Label_[did]["brt_dist"], Label_[did]["out_hue_dist"],
            Label_[did]["out_sat_dist"], Label_[did]["out_brt_dist"], original, norm)

    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(Label_, f)
            f.close()

    return Label_


def addSingleImage(img_path, polygon, norm=True, original=True, size_percent=True, distance_threshold=100):
    global Label_

    norm_size = norm_data([Label_[i]["poly_size"] for i in Label_])
    norm_edge = norm_data([Label_[i]["edge"] for i in Label_])
    new_id = []
    # Load Data
    count = int(list(Label_.keys())[-1].split("_")[0]) + 1
    count2 = 1
    for i in polygon:
        did = str(count) + "_" + str(count2)
        Label_[did] = {'filename': img_path.split("/")[-1], 'iid': count, 'did': count2, 'polygon': i}
        crop(did, 0.25, single=img_path)
        img_array = loadImage(os.path.join("./.images_temp", str(did) + ".jpg"))
        Label_[did]["img_arr"] = img_array
        Label_[did]["map"] = cal_map(Label_[did]["img_arr"].shape, Label_[did]["crop_poly"])
        Label_[did]["img_arr_mode"] = RGBConvert(Label_[did]["img_arr"], mode="HSV")
        if Label_[did]["crop_poly"].shape[0] <= 2:
            Label_[did]["crop_poly"] = np.array([[4, 5], [5, 6], [6, 5], [5, 4]])

        Label_[did]["bb_size"] = cal_bb_size(Label_[did]["crop_poly"])
        if Label_[did]["bb_size"] == 0:
            Label_[did]["bb_size"] = 1
        Label_[did]["poly_size"] = cal_poly_size(Label_[did]["map"])
        if size_percent:
            Label_[did]["bb_size"] /= Label_[did]["boundary"][1][0] * Label_[did]["boundary"][1][1]
            Label_[did]["poly_size"] /= Label_[did]["boundary"][1][0] * Label_[did]["boundary"][1][1]
        Label_[did]["hue"] = Label_[did]["img_arr_mode"][:, :, 0]
        Label_[did]["sat"] = Label_[did]["img_arr_mode"][:, :, 1]
        Label_[did]["value"] = Label_[did]["img_arr_mode"][:, :, 2]
        Label_[did]["neighbour_dist"] = cal_distance(Label_[did]["polygon"], polygon)
        Label_[did]["degree"], Label_[did]["edge"], Label_[did]["edge_len"], Label_[did]["turning"] = \
            cal_shape(Label_[did]["polygon"])

        # Feature Extraction
        Label_[did]["size"] = cal_size_edge(Label_[did]["poly_size"], original, norm, norm_size)
        Label_[did]["coverage"] = cal_coverage(Label_[did]["poly_size"], Label_[did]["bb_size"], original, norm)
        Label_[did]["asp_ratio"] = cal_asp_ratio(Label_[did]["crop_poly"], original, norm)
        Label_[did]["deg_avg"], Label_[did]["deg_mode"] = cal_deg(Label_[did]["degree"], original, norm)
        Label_[did]["edge"] = cal_size_edge(Label_[did]["edge"], original, norm, norm_edge)
        Label_[did]["sc_edge_ratio"], Label_[did]["sc_follow_turn"], Label_[did]["sc_reverse_turn"], \
        Label_[did]["sc_small_turn"], Label_[did]["sc_score"] = cal_shape_comp(Label_[did]["turning"],
                                                                               Label_[did]["degree"],
                                                                               Label_[did]["edge_len"], original, norm)
        Label_[did]["distance"] = cal_dist(Label_[did]["neighbour_dist"], distance_threshold, original, norm)
        Label_[did]["hue_avg"], Label_[did]["hue_mode"], Label_[did]["hue_range"], Label_[did]["hue_uni"], \
        Label_[did]["hue_dist"] = cal_hue(Label_[did]["hue"], False, Label_[did]["map"], original, norm)
        Label_[did]["sat_avg"], Label_[did]["sat_mode"], Label_[did]["sat_range"], Label_[did]["sat_uni"], \
        Label_[did]["sat_dist"] = cal_sat_brt(Label_[did]["sat"], False, Label_[did]["map"], original, norm)
        Label_[did]["brt_avg"], Label_[did]["brt_mode"], Label_[did]["brt_range"], Label_[did]["brt_uni"], \
        Label_[did]["brt_dist"] = cal_sat_brt(Label_[did]["value"], False, Label_[did]["map"], original, norm)
        Label_[did]["out_hue_avg"], Label_[did]["out_hue_mode"], Label_[did]["out_hue_range"], Label_[did][
            "out_hue_uni"], Label_[did]["out_hue_dist"] = cal_hue(Label_[did]["hue"], True, Label_[did]["map"],
                                                                  original, norm)
        Label_[did]["out_sat_avg"], Label_[did]["out_sat_mode"], Label_[did]["out_sat_range"], Label_[did][
            "out_sat_uni"], Label_[did]["out_sat_dist"] = cal_sat_brt(Label_[did]["sat"], True, Label_[did]["map"],
                                                                      original, norm)
        Label_[did]["out_brt_avg"], Label_[did]["out_brt_mode"], Label_[did]["out_brt_range"], Label_[did][
            "out_brt_uni"], Label_[did]["out_brt_dist"] = cal_sat_brt(Label_[did]["value"], True, Label_[did]["map"],
                                                                      original, norm)
        Label_[did]["hue_outin"], Label_[did]["sat_outin"], Label_[did]["brt_outin"] = cal_comp_colour(
            Label_[did]["hue_dist"], Label_[did]["sat_dist"], Label_[did]["brt_dist"], Label_[did]["out_hue_dist"],
            Label_[did]["out_sat_dist"], Label_[did]["out_brt_dist"], original, norm)
        new_id.append(did)
        count2 += 1

    return Label_, new_id


def extractSingleDefect(img, polygon, neighbour, norm=True, original=True, size_percent=True, distance_threshold=100, size_norm=None, edge_norm=None):
    global Label_

    norm_size = norm_data([Label_[i]["poly_size"] for i in Label_])
    norm_edge = norm_data([Label_[i]["edge"] for i in Label_])
    if size_norm is not None:
        norm_size = size_norm
    if edge_norm is not None:
        norm_edge = edge_norm

    Label_single = {"polygon": polygon}
    Label_single["norm_size"] = norm_size
    Label_single["norm_edge"] = norm_edge
    Label_single["boundary"], Label_single["width"], Label_single["height"], Label_single["crop_poly"] = crop(None, 0.25, single=img, poly=polygon)
    Label_single["img_arr"] = loadImage(os.path.join("./.images_temp", "-1" + ".jpg"))
    Label_single["map"] = cal_map(Label_single["img_arr"].shape, Label_single["crop_poly"])
    Label_single["img_arr_mode"] = RGBConvert(Label_single["img_arr"], mode="HSV")
    if Label_single["crop_poly"].shape[0] <= 2:
        Label_single["crop_poly"] = np.array([[4, 5], [5, 6], [6, 5], [5, 4]])
    Label_single["bb_size"] = cal_bb_size(Label_single["crop_poly"])
    if Label_single["bb_size"] == 0:
        Label_single["bb_size"] = 1
    Label_single["poly_size"] = cal_poly_size(Label_single["map"])
    if size_percent:
        Label_single["bb_size"] /= Label_single["boundary"][1][0] * Label_single["boundary"][1][1]
        Label_single["poly_size"] /= Label_single["boundary"][1][0] * Label_single["boundary"][1][1]
    Label_single["hue"] = Label_single["img_arr_mode"][:, :, 0]
    Label_single["sat"] = Label_single["img_arr_mode"][:, :, 1]
    Label_single["value"] = Label_single["img_arr_mode"][:, :, 2]
    Label_single["neighbour_dist"] = cal_distance(Label_single["polygon"], neighbour)
    Label_single["degree"], Label_single["edge"], Label_single["edge_len"], Label_single["turning"] = \
        cal_shape(Label_single["polygon"])

    Label_single["size"] = cal_size_edge(Label_single["poly_size"], original, norm, norm_size)
    Label_single["coverage"] = cal_coverage(Label_single["poly_size"], Label_single["bb_size"], original, norm)
    Label_single["asp_ratio"] = cal_asp_ratio(Label_single["crop_poly"], original, norm)
    Label_single["deg_avg"], Label_single["deg_mode"] = cal_deg(Label_single["degree"], original, norm)
    Label_single["edge"] = cal_size_edge(Label_single["edge"], original, norm, norm_edge)
    Label_single["sc_edge_ratio"], Label_single["sc_follow_turn"], Label_single["sc_reverse_turn"], \
    Label_single["sc_small_turn"], Label_single["sc_score"] = cal_shape_comp(Label_single["turning"],
                                                                           Label_single["degree"],
                                                                           Label_single["edge_len"], original, norm)
    Label_single["distance"] = cal_dist(Label_single["neighbour_dist"], distance_threshold, original, norm)
    Label_single["hue_avg"], Label_single["hue_mode"], Label_single["hue_range"], Label_single["hue_uni"], \
    Label_single["hue_dist"] = cal_hue(Label_single["hue"], False, Label_single["map"], original, norm)
    Label_single["sat_avg"], Label_single["sat_mode"], Label_single["sat_range"], Label_single["sat_uni"], \
    Label_single["sat_dist"] = cal_sat_brt(Label_single["sat"], False, Label_single["map"], original, norm)
    Label_single["brt_avg"], Label_single["brt_mode"], Label_single["brt_range"], Label_single["brt_uni"], \
    Label_single["brt_dist"] = cal_sat_brt(Label_single["value"], False, Label_single["map"], original, norm)
    Label_single["out_hue_avg"], Label_single["out_hue_mode"], Label_single["out_hue_range"], Label_single[
        "out_hue_uni"], Label_single["out_hue_dist"] = cal_hue(Label_single["hue"], True, Label_single["map"],
                                                              original, norm)
    Label_single["out_sat_avg"], Label_single["out_sat_mode"], Label_single["out_sat_range"], Label_single[
        "out_sat_uni"], Label_single["out_sat_dist"] = cal_sat_brt(Label_single["sat"], True, Label_single["map"],
                                                                  original, norm)
    Label_single["out_brt_avg"], Label_single["out_brt_mode"], Label_single["out_brt_range"], Label_single[
        "out_brt_uni"], Label_single["out_brt_dist"] = cal_sat_brt(Label_single["value"], True, Label_single["map"],
                                                                  original, norm)
    Label_single["hue_outin"], Label_single["sat_outin"], Label_single["brt_outin"] = cal_comp_colour(
        Label_single["hue_dist"], Label_single["sat_dist"], Label_single["brt_dist"], Label_single["out_hue_dist"],
        Label_single["out_sat_dist"], Label_single["out_brt_dist"], original, norm)

    return Label_single


def aggregation(data, dist=None):
    if dist is None:
        dist = [1/len(data) for _ in data]
    avg = sum([data[i] * dist[i] for i in range(len(data))])
    return 1 - (sum([abs(i-avg) for i in data]) / len(data) * 2)

def mergeDefect(data, mid, original=True, norm=True):
    label_merge = {"mid": mid, "filename": []}
    total_size = sum([i["size"] for i in data])
    total_defect = len(data)
    distribution = []
    hue = np.array([])
    sat = np.array([])
    brt = np.array([])
    hmap = np.array([])
    deg = np.array([])
    neighbour = []
    for defect in data:
        # Default Information
        if defect["filename"] not in label_merge["filename"]:
            label_merge["filename"].append(defect["filename"])
        distribution.append(defect["size"] / total_size)
        hue = np.append(hue, defect["hue"][defect["map"]])
        hue = np.append(hue, defect["hue"][np.logical_not(defect["map"])])
        sat = np.append(sat, defect["sat"][defect["map"]])
        sat = np.append(sat, defect["sat"][np.logical_not(defect["map"])])
        brt = np.append(brt, defect["value"][defect["map"]])
        brt = np.append(brt, defect["value"][np.logical_not(defect["map"])])
        hmap = np.append(hmap, [True for _ in range(np.count_nonzero(defect["map"]))])
        hmap = np.append(hmap, [False for _ in range(np.count_nonzero(np.logical_not(defect["map"])))])
        deg = np.append(deg, defect["degree"])
        if len(defect["neighbour_dist"]) > 0:
            if (min(defect["neighbour_dist"]) / 1200) >= 0.9:
                neighbour.append(0.9)
            else:
                neighbour.append(min(defect["neighbour_dist"]) / 1200)

    hmap = hmap.astype(bool)
    neighbour = list(set(neighbour))
    neighbours = [i/2400 for i in neighbour]


    # Colour Information (Defect Area)
    label_merge["hue_avg_G"], label_merge["hue_mode_G"], label_merge["hue_range_G"], label_merge["hue_uni_G"], \
    label_merge["hue_dist_G"] = cal_hue(hue, False, hmap, original, norm)
    label_merge["hue_uni_L"] = sum([data[i]["hue_uni"]*distribution[i] for i in range(len(data))])
    label_merge["sat_avg_G"], label_merge["sat_mode_G"], label_merge["sat_range_G"], label_merge["sat_uni_G"], \
    label_merge["sat_dist_G"] = cal_sat_brt(sat, False, hmap, original, norm)
    label_merge["sat_uni_L"] = sum([data[i]["sat_uni"] * distribution[i] for i in range(len(data))])
    label_merge["brt_avg_G"], label_merge["brt_mode_G"], label_merge["brt_range_G"], label_merge["brt_uni_G"], \
    label_merge["brt_dist_G"] = cal_sat_brt(brt, False, hmap, original, norm)
    label_merge["brt_uni_L"] = sum([data[i]["brt_uni"] * distribution[i] for i in range(len(data))])

    label_merge["hue_avg_A"] = aggregation([i["hue_avg"] for i in data], distribution)
    label_merge["hue_mode_A"] = aggregation([i["hue_mode"] for i in data], distribution)
    label_merge["hue_range_A"] = aggregation([i["hue_range"] for i in data], distribution)
    label_merge["hue_uni_A"] = aggregation([i["hue_uni"] for i in data], distribution)

    label_merge["sat_avg_A"] = aggregation([i["sat_avg"] for i in data], distribution)
    label_merge["sat_mode_A"] = aggregation([i["sat_mode"] for i in data], distribution)
    label_merge["sat_range_A"] = aggregation([i["sat_range"] for i in data], distribution)
    label_merge["sat_uni_A"] = aggregation([i["sat_uni"] for i in data], distribution)

    label_merge["brt_avg_A"] = aggregation([i["brt_avg"] for i in data], distribution)
    label_merge["brt_mode_A"] = aggregation([i["brt_mode"] for i in data], distribution)
    label_merge["brt_range_A"] = aggregation([i["brt_range"] for i in data], distribution)
    label_merge["brt_uni_A"] = aggregation([i["brt_uni"] for i in data], distribution)

    # Colour Information (Background)
    label_merge["out_hue_avg_G"], label_merge["out_hue_mode_G"], label_merge["out_hue_range_G"], label_merge["out_hue_uni_G"], \
    label_merge["out_hue_dist_G"] = cal_hue(hue, True, hmap, original, norm)
    label_merge["out_hue_uni_L"] = sum([data[i]["out_hue_uni"] * distribution[i] for i in range(len(data))])
    label_merge["out_sat_avg_G"], label_merge["out_sat_mode_G"], label_merge["out_sat_range_G"], label_merge["out_sat_uni_G"], \
    label_merge["out_sat_dist_G"] = cal_sat_brt(sat, True, hmap, original, norm)
    label_merge["out_sat_uni_L"] = sum([data[i]["out_sat_uni"] * distribution[i] for i in range(len(data))])
    label_merge["out_brt_avg_G"], label_merge["out_brt_mode_G"], label_merge["out_brt_range_G"], label_merge["out_brt_uni_G"], \
    label_merge["out_brt_dist_G"] = cal_sat_brt(brt, True, hmap, original, norm)
    label_merge["out_brt_uni_L"] = sum([data[i]["out_brt_uni"] * distribution[i] for i in range(len(data))])

    label_merge["out_hue_avg_A"] = aggregation([i["out_hue_avg"] for i in data], distribution)
    label_merge["out_hue_mode_A"] = aggregation([i["out_hue_mode"] for i in data], distribution)
    label_merge["out_hue_range_A"] = aggregation([i["out_hue_range"] for i in data], distribution)
    label_merge["out_hue_uni_A"] = aggregation([i["out_hue_uni"] for i in data], distribution)

    label_merge["out_sat_avg_A"] = aggregation([i["out_sat_avg"] for i in data], distribution)
    label_merge["out_sat_mode_A"] = aggregation([i["out_sat_mode"] for i in data], distribution)
    label_merge["out_sat_range_A"] = aggregation([i["out_sat_range"] for i in data], distribution)
    label_merge["out_sat_uni_A"] = aggregation([i["out_sat_uni"] for i in data], distribution)

    label_merge["out_brt_avg_A"] = aggregation([i["out_brt_avg"] for i in data], distribution)
    label_merge["out_brt_mode_A"] = aggregation([i["out_brt_mode"] for i in data], distribution)
    label_merge["out_brt_range_A"] = aggregation([i["out_brt_range"] for i in data], distribution)
    label_merge["out_brt_uni_A"] = aggregation([i["out_brt_uni"] for i in data], distribution)

    # Colour Complexity

    label_merge["hue_outin_G"], label_merge["sat_outin_G"], label_merge["brt_outin_G"] = cal_comp_colour(
        label_merge["hue_dist_G"], label_merge["sat_dist_G"], label_merge["brt_dist_G"], label_merge["out_hue_dist_G"],
        label_merge["out_sat_dist_G"], label_merge["out_brt_dist_G"], original, norm)

    label_merge["hue_outin_L"] = sum([i["hue_outin"] for i in data]) / len(data)
    label_merge["sat_outin_L"] = sum([i["sat_outin"] for i in data]) / len(data)
    label_merge["brt_outin_L"] = sum([i["brt_outin"] for i in data]) / len(data)

    label_merge["hue_outin_A"] = aggregation([i["hue_outin"] for i in data])
    label_merge["sat_outin_A"] = aggregation([i["sat_outin"] for i in data])
    label_merge["brt_outin_A"] = aggregation([i["brt_outin"] for i in data])

    # Shape Information
    label_merge["edge_L"] = sum([data[i]["edge"] * distribution[i] for i in range(len(data))])
    label_merge["edge_A"] = aggregation([i["edge"] for i in data], distribution)
    label_merge["edge_S"] = sum([data[i]["edge"] for i in range(len(data))])

    label_merge["asp_ratio_L"] = sum([data[i]["asp_ratio"] * distribution[i] for i in range(len(data))])
    label_merge["asp_ratio_A"] = aggregation([i["asp_ratio"] for i in data], distribution)

    label_merge["coverage_L"] = sum([data[i]["coverage"] * distribution[i] for i in range(len(data))])
    label_merge["coverage_A"] = aggregation([i["coverage"] for i in data], distribution)

    label_merge["deg_avg_G"], label_merge["deg_mode_G"] = cal_deg(deg, True, True)
    label_merge["deg_avg_L"] = sum([data[i]["deg_avg"] * distribution[i] for i in range(len(data))])
    label_merge["deg_avg_A"] = aggregation([i["deg_avg"] for i in data], distribution)
    label_merge["deg_mode_L"] = sum([data[i]["deg_mode"] * distribution[i] for i in range(len(data))])
    label_merge["deg_mode_A"] = aggregation([i["deg_mode"] for i in data], distribution)

    # Shape Complexity
    label_merge["sc_edge_ratio_L"] = sum([data[i]["sc_edge_ratio"] * distribution[i] for i in range(len(data))])
    label_merge["sc_edge_ratio_A"] = aggregation([i["sc_edge_ratio"] for i in data], distribution)
    label_merge["sc_follow_turn_L"] = sum([data[i]["sc_follow_turn"] * distribution[i] for i in range(len(data))])
    label_merge["sc_follow_turn_A"] = aggregation([i["sc_follow_turn"] for i in data], distribution)
    label_merge["sc_small_turn_L"] = sum([data[i]["sc_small_turn"] * distribution[i] for i in range(len(data))])
    label_merge["sc_small_turn_A"] = aggregation([i["sc_small_turn"] for i in data], distribution)
    label_merge["sc_reverse_turn"] = sum([data[i]["sc_reverse_turn"] * distribution[i] for i in range(len(data))])
    label_merge["sc_reverse_turn_A"] = aggregation([i["sc_reverse_turn"] for i in data], distribution)

    # Meta Information
    label_merge["size_S"] = sum([data[i]["size"] for i in range(len(data))])
    label_merge["size_A"] = aggregation([i["size"] for i in data])
    label_merge["num_defect"] = total_defect
    if len(neighbours) <= 0:
        label_merge["distance_G"] = 1
        label_merge["distance_A"] = 1
    else:
        label_merge["distance_G"] = sum(neighbours)/len(neighbours)
        label_merge["distance_A"] = aggregation(neighbours)

    return label_merge



def mergeDefect_img(data):
    label_group = {}
    img_id = list(set([data[i]["iid"] for i in data]))
    img_id = sorted(img_id)
    for i in tqdm(img_id, desc="Merging Defects"):
        defect_group = [data[d] for d in list(filter(lambda x: data[x]["iid"] == i, data))]
        label_group[i] = mergeDefect(defect_group, i, True, True)

    return label_group

def mergeDefect_single(data, name=""):
    defect_group = [data[d] for d in data]
    label_group = mergeDefect(defect_group, name, True, True)
    return label_group

# Add own features - Add own feature values
# Input: Feature values:Dict(Dict()) - append the own feature values into each defect. -> for each defect_id and for each own feature
# Output: Feature values:Dict(Dict()) - List of feature values after adding own features
def addOwnFeatures(feature, save=None):
    global Label_, feature_list

    feature_list.extend(list(feature[list(feature.keys())[0]].keys()))
    for did in feature.keys():
        for f in feature[did].keys():
            Label_[did][f] = feature[did][f]

    if save is not None:
        with open(os.path.abspath(save), 'wb') as f:
            pickle.dump(Label_, f)
            f.close()

    return Label_


# ====================Get Function Areas====================

# Get Image Directory
# Output:   Path String
def get_ImageDir():
    return Image_Dir


# Get Image Lists
# Output:   List
def get_ImageList():
    return Image_List


# Get Extracted Label
# Output:   Dictionary
def get_Label():
    return Label


# Get Extracted Features
# Output:   Dictionary
def get_Features():
    return Label_


def get_FeatureList():
    return feature_list
 


def get_FeatureRange(own_range=None):
    feature_range = {}
    for i in feature_list:
        if setting["original"]:
            if setting["norm"]:
                feature_range[i] = {"min":0, "max":1}
            else:
                data_list = [Label_[j][i] for j in Label_]
                feature_range[i] = {"min":min(data_list), "max":max(data_list)}
        else:
            if setting["norm"]:
                feature_range[i] = {"min":0, "max":1}
            else:
                if "outin" in i:
                    feature_range[i] = {"min": 0, "max":10}
                elif "sc" in i:
                    feature_range[i] = {"min": 0, "max": 10}
                elif "hue" in i:
                    if "range" in i:
                        feature_range[i] = {"min": 0, "max": 6}
                    else:
                        feature_range[i] = {"min": 1, "max": 12}
                elif "brt" in i or "sat" in i:
                    feature_range[i] = {"min": 0, "max": 5}
                elif "distance" in i:
                    feature_range[i] = {"min": 0, "max": 2}
                elif "asp" in i or "edge" in i or "size" in i:
                    feature_range[i] = {"min": 0, "max": 10}
                elif "coverage" in i:
                    feature_range[i] = {"min": 0, "max": 5}
                elif "deg" in i:
                    feature_range[i] = {"min": 0, "max": 6}
    if own_range is not None:
        for i in own_range.keys():
            feature_range[i] = own_range[i]

    return feature_range