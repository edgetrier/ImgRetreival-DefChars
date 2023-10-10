import os, csv, warnings
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


def get_contours_from_maskfile(filepath, mask_contain_type):
    mask = cv2.imread(filepath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    max_value = int(mask.max())
    mask_group = []
    if mask_contain_type:
        for g in range(1, max_value+1):
            mask_group.append(cv2.inRange(mask, g,g))
    else:
        mask = (mask >= max_value / 2) * 1
        mask = mask.astype(np.uint8)
        mask_group = [mask]

    contours = []
    groups = []
    g_idx = 1
    for i in mask_group:
        co, hi = cv2.findContours(i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        for j in co:
            trim_j = j.reshape(j.shape[0], j.shape[-1])
            contours.append(trim_j)
            groups.append(g_idx)
        g_idx += 1
    return contours, groups

def get_masks_from_contours(img_shape, contour, ignore_thres):
    masks = []
    contours = []
    for i in contour:
        mask = np.zeros(img_shape[:2])
        cv2.drawContours(mask, [i], -1, 2, -1)
        mask = mask >= 1
        if (np.sum(mask) / (img_shape[0] * img_shape[1])) >= ignore_thres and len(i) > 3:
            masks.append(mask)
            contours.append(contour)

    return masks

def clean_contours_with_masks(img_shape, contour, ignore_thres, group):
    masks = []
    contours = []
    groups = []
    idx = 0
    for i in contour:
        mask = np.zeros(img_shape[:2])
        try:
            cv2.drawContours(mask, [i], -1, 2, -1)
        except:
            cv2.drawContours(mask, i, -1, 2, -1)
        mask = mask >= 1
        if (np.sum(mask) / (img_shape[0] * img_shape[1])) >= ignore_thres and len(i) > 3:
            masks.append(mask)
            contours.append(i)
            groups.append(group[idx])
        idx += 1

    return contours, masks, groups

def process_mask(img_path, gt, mask_contain_type, ignore_threshold=0):
    img_shape = cv2.imread(img_path).shape
    contour, group = get_contours_from_maskfile(img_path, mask_contain_type)
    contours = []
    masks = []
    groups = []
    if gt:
        contours, masks, groups = clean_contours_with_masks(img_shape, contour, 0, group)
    else:
        contours, masks, groups = clean_contours_with_masks(img_shape, contour, ignore_threshold, group)

    return contours, masks, groups


def process_bydir(ori_img="", gt_mask="", predict_mask="", mask_contain_type=False, iou_threshold=0.05, iou_smaller_than=[0.5], iou_larger_than=[], extend_on=None, mask_ignore_threshold=0.0005, fp=False, merged=False, separate=False):
    label = {}
    if extend_on is not None:
        if type(extend_on) != type({}):
            raise TypeError("The previous label should ba dictionary")
        label = extend_on

    if type(iou_smaller_than) == type(float()):
        iou_smaller_than = [iou_smaller_than]
    if type(iou_larger_than) == type(float()):
        iou_larger_than = [iou_larger_than]
    if ori_img == "":
        raise FileNotFoundError("Empty Original Image Directory Path")
    if gt_mask == "":
        raise FileNotFoundError("Empty Ground Truth Masks Directory Path")
    if predict_mask == "":
        raise FileNotFoundError("Empty Predicted Masks Directory Path")
    all_img = list(os.listdir(ori_img))
    for i in tqdm(all_img, desc="Processing all outputs"):

        label[i] = {}
        origin = cv2.imread(os.path.join(ori_img, i))
        label[i]["origin_image_path"] = os.path.join(ori_img, i)
        if os.path.exists(os.path.join(gt_mask, i)):
            gt_c, gt_m, gt_t = process_mask(os.path.join(gt_mask, i), True, mask_contain_type)
        else:
            gt_c, gt_m, gt_t = process_mask(os.path.join(gt_mask, i.split(".")[0] + ".png"), True, mask_contain_type)
        label[i]["gt_contours"] = gt_c
        label[i]["gt_masks"] = gt_m
        label[i]["gt_types"] = gt_t
        if os.path.exists(os.path.join(predict_mask, i)):
            pred_c, pred_m, pred_t = process_mask(os.path.join(predict_mask, i), False, mask_contain_type, mask_ignore_threshold)
        else:
            if os.path.exists(os.path.join(predict_mask, i.split(".")[0] + ".png")):
                pred_c, pred_m, pred_t = process_mask(os.path.join(predict_mask, i.split(".")[0] + ".png"), False, mask_contain_type, mask_ignore_threshold)
            else:
                pred_c = []
                pred_m = []
                pred_t = []
        label[i]["predict_contours"] = pred_c
        label[i]["predict_masks"] = pred_m
        label[i]["predict_types"] = pred_t
        label[i]["results"] = []
        label[i]["masks"] = []
        label[i]["polygons"] = []
        label[i]["matches"] = []
        left_pred_idx = list(range(len(pred_m)))
        merge_gt_idx = [[] for p in range(len(pred_m))]
        used_gt = 0
        for g in gt_m:
            matched_pred = []
            result = {"detected": 0, "not-detected": 0, "false-positive": 0, "separate-detected":0, "merged-detected":0}
            if mask_contain_type:
                result = {"detected": 0, "not-detected": 0, "false-positive": 0, "separate-detected": 0,
                          "merged-detected": 0, "correct-type-classified": 0, "wrong-type-classified":0}
            used_pred = 0
            for p in pred_m:
                iou = np.sum(np.logical_and(g, p)) / np.sum(np.logical_or(g, p))
                if iou >= iou_threshold:
                    matched_pred.append([used_pred, p, iou])
                    merge_gt_idx[used_pred].append(used_gt)
                    try:
                        left_pred_idx.remove(used_pred)
                    except:
                        pass
                elif iou > 0:
                    try:
                        left_pred_idx.remove(used_pred)
                    except:
                        pass

                used_pred += 1


            if len(matched_pred) > 0:
                result["detected"] = 1
                if len(matched_pred) > 1 and separate:
                    result["separate-detected"] = 1
                if mask_contain_type:
                    result["correct-type-classified"] = 0
                    result["wrong-type-classified"] = 1
                    for p in matched_pred:
                        if gt_t[used_gt] == pred_t[p[0]]:
                            result["correct-type-classified"] = 1
                            result["wrong-type-classified"] = 0
                            break

            else:
                result["not-detected"] = 1

            sum_iou = sum([i[-1] for i in matched_pred])
            if sum_iou > 1:
                sum_iou = 1

            if iou_smaller_than != []:
                for iout in iou_smaller_than:
                    if sum_iou < iout:
                        result["iou-less-" + str(iout)] = 1
                    else:
                        result["iou-less-" + str(iout)] = 0
            if iou_larger_than != []:
                for iout in iou_larger_than:
                    if sum_iou >= iout:
                        result["iou-more-" + str(iout)] = 1
                    else:
                        result["iou-more-" + str(iout)] = 0

            label[i]["results"].append(result)
            label[i]["masks"].append(g)
            label[i]["polygons"].append(gt_c[used_gt])
            label[i]["matches"].append([p[0] for p in matched_pred])
            used_gt += 1

        for p in merge_gt_idx:
            if len(p) > 1 and merged:
                for g in p:
                    label[i]["results"][g]["merged-detected"] = 1

        if len(left_pred_idx) >= 1 and fp:
            for p in left_pred_idx:
                result = {"detected": 0, "not-detected": 1, "false-positive": 1, "separate-detected": 0, "merged-detected": 0}
                if mask_contain_type:
                    result = {"detected": 0, "not-detected": 1, "false-positive": 1, "separate-detected": 0,
                              "merged-detected": 0, "correct-type-classified": 0, "wrong-type-classified": 1}
                if iou_smaller_than != []:
                    for iout in iou_smaller_than:
                        result["iou-less-" + str(iout)] = 1

                if iou_larger_than != []:
                    for iout in iou_larger_than:
                        result["iou-more-" + str(iout)] = 0
                label[i]["results"].append(result)
                label[i]["masks"].append(pred_m[p])
                label[i]["polygons"].append(pred_c[p])
                label[i]["matches"].append([])


    return label
