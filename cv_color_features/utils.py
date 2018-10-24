import numpy as np
# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2
import pandas as pd
from scipy.spatial.distance import pdist

np.seterr(all='warn')

HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([0, 39, 64]),
            'upper': np.array([20, 255, 255])
        },
        {
            'lower': np.array([161, 39, 64]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([21, 39, 64]),
            'upper': np.array([40, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([101, 39, 64]),
            'upper': np.array([140, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 10% of V
    'black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 11-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([180, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([180, 38, 255])
        }
    ],
    'white_blue': [
        {
            'lower': np.array([101, 0, 160]),
            'upper': np.array([140, 76, 255])
        }
    ],
    'dark_blue': [
        {
            'lower': np.array([101, 77, 0]),
            'upper': np.array([140, 255, 159])
        }
    ]
}


def calc_distance(x1, y1, x2, y2):
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def create_mask(hsv_img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask


def get_hsv(hsv_img, mask=None):
    """
    Returns flattened hue, saturation, and values from given HSV image.
    """
    hue = hsv_img[:, :, 0].flatten()
    sat = hsv_img[:, :, 1].flatten()
    val = hsv_img[:, :, 2].flatten()

    if mask is not None:
        flat_mask = mask.flatten()

        hue = hue[flat_mask > 0]
        sat = sat[flat_mask > 0]
        val = val[flat_mask > 0]

    return hue, sat, val


def get_color_profile(hsv_img, mask=None):
    """
    Finds color profile as pixel counts for color ranges in HSV_RANGES

    Args:
        hsv_img: HSV pixel data (3-D NumPy array)
        mask: optional mask to apply to hsv_img

    Returns:
        Text string for dominant color range (from HSV_RANGES keys)

    Raises:
        tbd
    """
    h, s, v = get_hsv(hsv_img, mask)

    color_profile = {}

    for color, color_ranges in HSV_RANGES.items():
        color_profile[color] = 0

        for color_range in color_ranges:
            pixel_count = np.sum(
                np.logical_and(
                    h >= color_range['lower'][0],
                    h <= color_range['upper'][0]
                ) &
                np.logical_and(
                    s >= color_range['lower'][1],
                    s <= color_range['upper'][1]
                ) &
                np.logical_and(
                    v >= color_range['lower'][2],
                    v <= color_range['upper'][2]
                )
            )

            color_profile[color] += pixel_count

    return color_profile


def get_target_features(hsv_img, mask=None):
    h, s, v = get_hsv(hsv_img, mask)
    s = s / 255.0
    v = v / 255.0
    color_features = get_color_features(hsv_img, mask=mask)

    feature_names = []
    values = []

    # add region features first
    feature_names.append('region_area')
    values.append(len(h))

    feature_names.append('region_saturation_mean')
    values.append(np.mean(s))

    feature_names.append('region_saturation_variance')
    values.append(np.var(s))

    feature_names.append('region_value_mean')
    values.append(np.mean(v))

    feature_names.append('region_value_variance')
    values.append(np.var(v))

    for color, features in color_features.items():
        for feature, value in sorted(features.items()):
            feature_str = '%s (%s)' % (feature, color)
            feature_names.append(feature_str)
            values.append(value)

    target_features = pd.Series(values, index=feature_names)

    return target_features.sort_index()


def get_color_features(hsv_img, mask=None):
    """
    Takes an hsv image and returns a custom set of features useful for machine learning
    :param hsv_img: np.array with color scheme hsv
    :param mask: np.array list that contains the line segments of the thing being described
    :return: dictionary of features for machine learning
    """
    c_prof = get_color_profile(hsv_img, mask)

    if mask is not None:
        tot_px = np.sum(mask > 0)
    else:
        tot_px = hsv_img.shape[0] * hsv_img.shape[1]

    # corner to corner distance used to normalize sub-contour distance metrics
    diag_distance = calc_distance(0, 0, hsv_img.shape[0], hsv_img.shape[1])

    color_features = {}

    for color in HSV_RANGES.keys():
        color_percent = float(c_prof[color]) / tot_px

        # create color mask & apply it
        color_mask = create_mask(hsv_img, [color])

        # apply user specified mask
        if mask is not None:
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)

        ret, thresh = cv2.threshold(color_mask, 1, 255, cv2.THRESH_BINARY)
        new_mask, contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cent_list = []
        area_list = []
        peri_list = []

        if len(contours) == 0:
            color_features[color] = {
                'percent': color_percent,
                'contour_count': len(cent_list),
                'distance_mean': 0.0,
                'distance_variance': 0.0,
                'area_mean': 0.0,
                'area_variance': 0.0,
                'perimeter_mean': 0.0,
                'perimeter_variance': 0.0,
                'largest_contour_area': 0.0,
                'largest_contour_eccentricity': 0.0,
                'largest_contour_circularity': 0.0,
                'largest_contour_convexity': 0.0
            }
            continue

        largest_contour_area = 0.0
        largest_contour_true_area = 0.0
        largest_contour_peri = 0.0
        largest_contour = None

        for c in contours:
            true_area = cv2.contourArea(c)
            area = true_area / float(tot_px)

            # contour may be a single point or a line, which has no area
            # we'll also ignore "noise" of anything 4 pixels or less
            if true_area <= 4.0:
                continue

            peri = cv2.arcLength(c, True)

            if area > largest_contour_area:
                largest_contour_area = area
                largest_contour_true_area = true_area
                largest_contour_peri = peri
                largest_contour = c

            m = cv2.moments(c)
            centroid_x = m['m10'] / m['m00']
            centroid_y = m['m01'] / m['m00']

            cent_list.append((centroid_x, centroid_y))
            area_list.append(area)
            peri_list.append(peri)

        if len(cent_list) <= 1:
            pair_dist = [0.0]
        else:
            pair_dist = pdist(np.array(cent_list)) / diag_distance

        dist_mean = np.mean(pair_dist)
        dist_var = np.var(pair_dist)

        if len(area_list) == 0:
            area_mean = 0.0
            area_var = 0.0
        else:
            area_mean = np.mean(area_list)
            area_var = np.var(area_list)

        if len(peri_list) == 0:
            peri_mean = 0.0
            peri_var = 0.0
        else:
            peri_mean = np.mean(peri_list)
            peri_var = np.var(peri_list)

        largest_contour_eccentricity = 0.0
        largest_contour_circularity = 0.0
        largest_contour_convexity = 0.0

        if largest_contour_true_area >= 0.0 and largest_contour is not None:
            # get smallest bounding rectangle (rotated)
            box = cv2.minAreaRect(largest_contour)
            cnt_w, cnt_h = box[1]

            largest_contour_eccentricity = cnt_w / cnt_h
            if largest_contour_eccentricity > 1:
                largest_contour_eccentricity = 1.0 / largest_contour_eccentricity

            # calculate circularity as (4 * pi * area) / perimeter ^ 2
            largest_contour_circularity = (4 * np.pi * largest_contour_true_area) / float(largest_contour_peri)**2

            # calculate convexity as convex hull perimeter / contour perimeter
            hull = cv2.convexHull(largest_contour)
            largest_contour_convexity = cv2.arcLength(hull, True) / largest_contour_peri

        color_features[color] = {
            'percent': color_percent,
            'contour_count': len(cent_list),
            'distance_mean': dist_mean,
            'distance_variance': dist_var,
            'area_mean': area_mean,
            'area_variance': area_var,
            'perimeter_mean': peri_mean,
            'perimeter_variance': peri_var,
            'largest_contour_area': largest_contour_area,
            'largest_contour_eccentricity': largest_contour_eccentricity,
            'largest_contour_circularity': largest_contour_circularity,
            'largest_contour_convexity': largest_contour_convexity
        }

    return color_features


def generate_features(hsv_img_as_numpy, polygon_points, label=None, region_file_path=None):
    """
    Given an hsv image represented as a numpy array, polygon points which represent a
    target entity, and a label, this function will return a set of important features about
    the entity outlined in the polygons (plus some area blackened to generate a rectangle).
    :param hsv_img_as_numpy: numpy.array
    :param polygon_points: numpy.array
    :param label: str: indicating what the thing is
    :param region_file_path: str: optional file path to save the cropped sub-region as PNG
    :return: a dictionary containing features and a label key
    """
    polygon_points = polygon_points.copy()
    b_rect = cv2.boundingRect(polygon_points)
    x1 = b_rect[0]
    x2 = b_rect[0] + b_rect[2]
    y1 = b_rect[1]
    y2 = b_rect[1] + b_rect[3]

    mask = np.zeros(hsv_img_as_numpy.shape[0:2], dtype=np.uint8)

    cv2.drawContours(mask, [polygon_points], 0, 255, cv2.FILLED)

    mask_img = cv2.bitwise_and(hsv_img_as_numpy, hsv_img_as_numpy, mask=mask)

    # crop region and poly points for efficiency
    this_mask_img = mask_img[y1:y2, x1:x2]
    if len(polygon_points.shape) == 3:
        # dealing with OpenCV type contour
        polygon_points[:, :, 0] = polygon_points[:, :, 0] - x1
        polygon_points[:, :, 1] = polygon_points[:, :, 1] - y1
    else:
        # assume a simple array of x, y coordinates
        polygon_points[:, 0] = polygon_points[:, 0] - x1
        polygon_points[:, 1] = polygon_points[:, 1] - y1

    crop_mask = np.zeros(this_mask_img.shape[0:2], dtype=np.uint8)

    cv2.drawContours(crop_mask, [polygon_points], 0, 255, cv2.FILLED)

    target_features = get_target_features(this_mask_img, mask=crop_mask)

    if region_file_path is not None:
        cv2.imwrite(region_file_path, cv2.cvtColor(this_mask_img, cv2.COLOR_HSV2BGR))

    results = target_features.to_dict()
    results['label'] = label

    return results
