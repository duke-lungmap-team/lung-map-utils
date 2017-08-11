import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import operator
from scipy import optimize
import os
import re
import pandas as pd
import glob
# noinspection PyPackageRequirements
from sklearn.model_selection import GridSearchCV
# noinspection PyPackageRequirements
from sklearn.svm import SVC
# noinspection PyPackageRequirements
from sklearn.linear_model import RidgeClassifier
# noinspection PyPackageRequirements
from sklearn.pipeline import Pipeline
# noinspection PyPackageRequirements
from sklearn.preprocessing import StandardScaler
# noinspection PyPackageRequirements
from sklearn.feature_selection import SelectFdr
from scipy.spatial.distance import pdist
import warnings

np.seterr(all='warn')

alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
ridge_params = {'alpha': alpha}

c_s = [0.01, 0.1, 1.0, 10.0, 100.0]
gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
svc_params = [{'kernel': ['rbf'], 'gamma': gamma, 'C': c_s},
              {'kernel': ['linear'], 'C': c_s}]
clf = GridSearchCV(SVC(probability=True), svc_params, cv=4)
pipe = Pipeline([
        ('scaler', StandardScaler()),
        # ('scaler', RobustScaler()),
        ('feature_selection', SelectFdr()),
        # ('feature_selection', SelectKBest(k=400)),
        ('classification', clf)
    ])

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
    ]
}


def create_mask(hsv_img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            # noinspection PyUnresolvedReferences
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask


def fill_holes(mask):
    """
    Fills holes in a given binary mask.
    """
    # noinspection PyUnresolvedReferences
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # noinspection PyUnresolvedReferences
    new_mask, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        # noinspection PyUnresolvedReferences
        cv2.drawContours(new_mask, [cnt], 0, 255, -1)

    return new_mask


def filter_contours_by_size(mask, min_size=1024, max_size=None):
    # noinspection PyUnresolvedReferences
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # noinspection PyUnresolvedReferences
    new_mask, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if max_size is None:
        max_size = int(mask.shape[0] * mask.shape[1] * 0.20)
    min_size = min_size

    good_contours = []

    for c in contours:
        # noinspection PyUnresolvedReferences
        rect = cv2.boundingRect(c)
        rect_area = rect[2] * rect[3]

        if max_size >= rect_area >= min_size:
            good_contours.append(c)

    return good_contours


def gaussian(x, height, center, width):
    return height * np.exp(-(x - center) ** 2 / (2 * width ** 2))


def two_gaussian(x, h1, c1, w1, h2, c2, w2):
    return (
        gaussian(x, h1, c1, w1) +
        gaussian(x, h2, c2, w2)
    )


def error_function(p, x, y):
    return (two_gaussian(x, *p) - y) ** 2


def determine_hist_mode(sat_channel):
    cnt, bins = np.histogram(sat_channel.flatten(), bins=256, range=(0, 256))

    maximas = {}

    for i, c in enumerate(cnt[:-1]):
        if cnt[i + 1] < c:
            maximas[i] = c

    maximas = sorted(maximas.items(), key=operator.itemgetter(1))

    if len(maximas) > 1:
        maximas = maximas[-2:]
    else:
        return None

    guess = []

    for m in maximas:
        guess.extend([m[1], m[0], 10])

    optim, success = optimize.leastsq(error_function, guess[:], args=(bins[:-1], cnt))

    min_height = int(sat_channel.shape[0] * sat_channel.shape[1] * 0.01)

    if optim[2] >= optim[-1] and optim[0] > min_height:
        center = optim[1],
        width = optim[2]
    else:
        center = optim[4]
        width = optim[5]

    lower_bound = int(center - width / 2.0)
    upper_bound = int(center + width / 2.0)

    return lower_bound, upper_bound


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

    # TODO: take h, s, v as separate args to avoid multiple calls to get_hsv()...also eliminates mask arg
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


def plot_rectangles(hsv_img, fig_size=(16, 16), rectangles=None, lw=1, alt_rectangles=None, lw2=1):
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    # noinspection PyUnresolvedReferences
    plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))

    if rectangles is not None:
        for rect in rectangles:
            ax.add_patch(
                patches.Rectangle(
                    xy=(rect[0], rect[1]),
                    width=rect[2],
                    height=rect[3],
                    fill=False,
                    edgecolor='#00ff00',
                    lw=lw
                )
            )
    if alt_rectangles is not None:
        for rect in alt_rectangles:
            ax.add_patch(
                patches.Rectangle(
                    xy=(rect[0], rect[1]),
                    width=rect[2],
                    height=rect[3],
                    fill=False,
                    edgecolor='#ff0000',
                    lw=lw2
                )
            )


def find_rect_union(rect_list):
    for i, r in enumerate(rect_list):
        if i == 0:
            min_x = r[0]
            min_y = r[1]
            max_x = r[0] + r[2]
            max_y = r[1] + r[3]
            continue

        if r[0] < min_x:
            min_x = r[0]
        if r[1] < min_y:
            min_y = r[1]
        if r[0] + r[2] > max_x:
            max_x = r[0] + r[2]
        if r[1] + r[3] > max_y:
            max_y = r[1] + r[3]

    return min_x, min_y, max_x - min_x, max_y - min_y


def find_contour_union(contour_list, img_shape):
    union_mask = np.zeros(img_shape, dtype=np.uint8)

    for c in contour_list:
        c_mask = np.zeros(img_shape, dtype=np.uint8)
        # noinspection PyUnresolvedReferences
        cv2.drawContours(c_mask, [c], 0, 255, cv2.FILLED)
        # noinspection PyUnresolvedReferences
        union_mask = cv2.bitwise_or(union_mask, c_mask)

    return union_mask


def get_class_map(train_dir):
    class_paths = []

    for path in sorted(glob.glob("/".join([train_dir, "*"]))):
        if os.path.isdir(path):
            class_paths.append(path)

    class_map = {}

    for i, class_path in enumerate(sorted(class_paths)):
        folder, class_name = os.path.split(class_path)

        class_id = i + 1
        class_map[class_id] = class_name

    return class_map


def get_image_metadata(metadata, source_file_name):
    matches = metadata[metadata.img_file == source_file_name]

    first = matches[matches.index == matches.first_valid_index()]

    species = first.organism_label.get_values()[0]
    if species == 'mus musculus':
        species = 'mouse'
    else:
        species = 'human'

    development = first.age_label.get_values()[0]
    magnification = first.magnification.get_values()[0]

    probes = matches.probe_label.unique()
    probes = [p.lower() for p in probes]

    return species, development, magnification, probes


def load_training_data(train_dir):
    class_paths = []

    for path in sorted(glob.glob("/".join([train_dir, "*"]))):
        if os.path.isdir(path):
            class_paths.append(path)

    class_map = {}

    ss = []

    for i, class_path in enumerate(sorted(class_paths)):
        folder, class_name = os.path.split(class_path)

        class_id = i + 1
        class_map[class_id] = class_name

        train_files = glob.glob("/".join([class_path, "*.sig"]))

        for f in train_files:
            with open(f) as f_in:
                lines = f_in.readlines()

                values = []
                features = []

                for line in lines[2:]:
                    val, feature = line.split('\t')
                    values.append(val)
                    features.append(feature.strip())

                features.extend(['class_id', 'class', 'Path'])
                values.extend([class_id, class_name, f])

                s = pd.Series(values, index=features)
                ss.append(s)

    training_data = pd.concat(ss, axis=1).T

    # need to convert values to numeric else they will all be 'object' dtypes
    training_data = training_data.convert_objects(convert_dates=False, convert_numeric=True)

    return training_data, class_map


def build_trained_model(training_data, classifier='svc'):
    alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ridge_params = {'alpha': alpha}

    c_s = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    svc_params = [{'kernel': ['rbf'], 'gamma': gamma, 'C': c_s},
                  {'kernel': ['linear'], 'C': c_s}]

    if classifier == 'svc':
        clf = GridSearchCV(SVC(probability=True), svc_params, cv=5)
        # clf = GridSearchCV(SVC(probability=True, class_weight='balanced'), svc_params, cv=5)
    elif classifier == 'ridge':
        clf = GridSearchCV(RidgeClassifier(), ridge_params, cv=5)
    else:
        raise NotImplementedError("Only 'svc' (default) and 'ridge' classifiers are supported")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        # ('scaler', RobustScaler()),
        ('feature_selection', SelectFdr()),
        # ('feature_selection', SelectKBest(k=400)),
        ('classification', clf)
    ])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        pipe.fit(
            training_data.ix[:, :-3],
            training_data.ix[:, -3].astype('int')
        )

    return pipe


def get_custom_target_features(hsv_img, mask=None, show_plots=False):
    h, s, v = get_hsv(hsv_img, mask)
    color_features = get_custom_color_features(hsv_img, mask=mask, show_plots=show_plots)

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


def get_custom_color_features(hsv_img, mask=None, show_plots=False):
    """
    Takes an hsv image and returns a custom set of features useful for machine learning
    :param hsv_img: np.array with color scheme hsv
    :param mask: np.array list that contains the line segments of the thing being described
    :param show_plots: boolean to show plots
    :return: dictionary of features for machine learning
    """
    c_prof = get_color_profile(hsv_img, mask)

    if mask is not None:
        tot_px = np.sum(mask > 0)
    else:
        tot_px = hsv_img.shape[0] * hsv_img.shape[1]

    color_features = {}

    for color in HSV_RANGES.keys():
        color_percent = float(c_prof[color]) / tot_px

        # create color mask & apply it
        color_mask = create_mask(hsv_img, [color])
        # noinspection PyUnresolvedReferences
        mask_img = cv2.bitwise_and(hsv_img, hsv_img, mask=color_mask)

        # apply user specified mask
        if mask is not None:
            # noinspection PyUnresolvedReferences
            mask_img = cv2.bitwise_and(mask_img, mask_img, mask=mask)
            # noinspection PyUnresolvedReferences
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)

        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.title(color)
            # noinspection PyUnresolvedReferences
            ax1.imshow(cv2.cvtColor(mask_img, cv2.COLOR_HSV2RGB))
            # noinspection PyUnresolvedReferences
            ax2.imshow(cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB))
        # noinspection PyUnresolvedReferences
        ret, thresh = cv2.threshold(color_mask, 1, 255, cv2.THRESH_BINARY)
        # noinspection PyUnresolvedReferences
        new_mask, contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
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
        largest_contour_peri = 0.0
        largest_contour = None

        for c in contours:
            # noinspection PyUnresolvedReferences
            area = cv2.contourArea(c)

            if area <= 0.0:
                continue
            # noinspection PyUnresolvedReferences
            peri = cv2.arcLength(c, True)

            if area > largest_contour_area:
                largest_contour_area = area
                largest_contour_peri = peri
                largest_contour = c
            # noinspection PyUnresolvedReferences
            m = cv2.moments(c)
            centroid_x = m['m10'] / m['m00']
            centroid_y = m['m01'] / m['m00']

            cent_list.append((centroid_x, centroid_y))
            area_list.append(area)
            peri_list.append(peri)

        if len(cent_list) <= 1:
            pair_dist = [0.0]
        else:
            pair_dist = pdist(cent_list)

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

        if largest_contour_area >= 0.02 * tot_px and largest_contour is not None:
            # get smallest bounding rectangle (rotated)
            # noinspection PyUnresolvedReferences
            box = cv2.minAreaRect(largest_contour)
            cnt_w, cnt_h = box[1]

            largest_contour_eccentricity = cnt_w / cnt_h
            if largest_contour_eccentricity < 1:
                largest_contour_eccentricity = 1.0 / largest_contour_eccentricity

            # calculate inverse circularity as 1 / (area / perimeter ^ 2)
            largest_contour_circularity = largest_contour_peri / np.sqrt(largest_contour_area)

            # calculate convexity as convex hull perimeter / contour perimeter
            # noinspection PyUnresolvedReferences
            hull = cv2.convexHull(largest_contour)
            # noinspection PyUnresolvedReferences
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


def write_custom_sig_file(img_path, target_features, suffix=None):
    m = re.search("(.*)\.(tif)$", img_path)

    g = m.groups()

    if len(g) > 0:
        if suffix is None:
            sig_file = '{0}.sig'.format(g[0])
        else:
            sig_file = '{0}_{1}.sig'.format(g[0], suffix)

        with open(sig_file, 'w') as f:
            f.write('custom features\n')
            f.write(img_path + '\n')
            for feature, value in sorted(target_features.iteritems()):
                line_str = '%f\t%s\n' % (value, feature)
                f.write(line_str)


def find_border_contours(contours, img_h, img_w):
    """
    Given a list of contours, splits them into 2 lists: the border contours and non-border contours

    Args:
        contours: list of contours to separate
        img_h: original image height
        img_w: original image width

    Returns:
        2 lists, the first being the border contours

    Raises:
        tbd
    """

    min_y = 0
    min_x = 0

    max_y = img_h - 1
    max_x = img_w - 1

    mins = {min_x, min_y}
    maxs = {max_x, max_y}

    border_contours = []
    non_border_contours = []

    for c in contours:
        # noinspection PyUnresolvedReferences
        rect = cv2.boundingRect(c)

        c_min_x = rect[0]
        c_min_y = rect[1]
        c_max_x = rect[0] + rect[2] - 1
        c_max_y = rect[1] + rect[3] - 1

        c_mins = {c_min_x, c_min_y}
        c_maxs = {c_max_x, c_max_y}

        if len(mins.intersection(c_mins)) > 0 or len(maxs.intersection(c_maxs)) > 0:
            border_contours.append(c)
        else:
            non_border_contours.append(c)

    return border_contours, non_border_contours


def generate_custom_features(hsv_img_as_numpy, polygon_points, label):
    """
    Given an hsv image represented as a numpy arrary, polygon points which represent a
    target entity, and a label, this function will return a set of important features about
    the entity outlined in the polygons (plus some area blackened to generate a rectangle).
    :param hsv_img_as_numpy: numpy.array
    :param polygon_points: numpy.array
    :param label: str: indicating what the thing is
    :return: a dictionary containg features and a label key
    """
    # noinspection PyUnresolvedReferences
    b_rect = cv2.boundingRect(polygon_points)
    x1 = b_rect[0]
    x2 = b_rect[0] + b_rect[2]
    y1 = b_rect[1]
    y2 = b_rect[1] + b_rect[3]
    mask = np.zeros(hsv_img_as_numpy.shape[0:2], dtype=np.uint8)
    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    # noinspection PyUnresolvedReferences
    cv2.drawContours(mask, [polygon_points], 0, 255, cv2.FILLED)
    # noinspection PyUnresolvedReferences
    mask_img = cv2.bitwise_and(hsv_img_as_numpy, hsv_img_as_numpy, mask=mask)
    # plt.imshow(cv2.cvtColor(mask_img, cv2.COLOR_HSV2RGB))
    this_mask_img = mask_img[y1:y2, x1:x2]
    # plt.imshow(cv2.cvtColor(this_mask_img, cv2.COLOR_HSV2RGB))
    target_features = get_custom_target_features(this_mask_img)
    results = target_features.to_dict()
    results['label'] = label
    return results

def fill_border_contour(contour, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    # noinspection PyUnresolvedReferences
    cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

    # Extract the perimeter pixels, leaving out the last pixel
    # of each side as it is included in the next side (going clockwise).
    # This makes all the side arrays the same length.
    # We also flip the bottom and left side, as we want to "unwrap" the
    # perimeter pixels in a clockwise fashion.
    top = mask[0, :-1]
    right = mask[:-1, -1]
    bottom = np.flipud(mask[-1, 1:])
    left = np.flipud(mask[1:, 0])

    # combine the perimeter sides into one continuous array
    perimeter_pixels = np.concatenate([top, right, bottom, left])

    region_boundary_locs = np.where(perimeter_pixels == 255)[0]

    # the perimeter here is not a geometric perimeter but the number of pixels around the image
    img_h = img_shape[0]
    img_w = img_shape[1]
    perimeter = (img_h - 1) * 2 + (img_w - 1) * 2

    # account for the wrap around from the last contour pixel to the end,
    # i.e. back at the start at (0, 0)
    wrap_distance = region_boundary_locs.max() - perimeter

    # insert the wrap distance in front of the region boundary locations
    region_boundary_locs = np.concatenate([[wrap_distance], region_boundary_locs])

    # calculate the gap size between boundary pixel locations
    gaps = np.diff(region_boundary_locs)

    # if there's only one gap, the contour is already filled
    if not np.sum(gaps > 1) > 1:
        return mask

    # add one to the results because of the diff offset
    max_gap_idx = np.where(gaps == gaps.max())[0] + 1

    # there should only be one, else we have a tie and should probably ignore that case
    if max_gap_idx.size != 1:
        return None

    start_perim_loc_of_flood_entry = region_boundary_locs[max_gap_idx[0]]

    # see if subsequent perimeter locations were also part of the contour,
    # adding one to the last one we found
    subsequent_region_border_locs = region_boundary_locs[
        region_boundary_locs > start_perim_loc_of_flood_entry
    ]

    flood_fill_entry_point = start_perim_loc_of_flood_entry

    for loc in subsequent_region_border_locs:
        if loc == flood_fill_entry_point + 1:
            flood_fill_entry_point = loc

    # we should hit the first interior empty point of the contour by moving forward one pixel around the perimeter
    flood_fill_entry_point += 1

    # now to convert our perimeter location back to an image coordinate
    if flood_fill_entry_point < img_w:
        print('top')
        flood_fill_entry_coords = (flood_fill_entry_point, 0)
    elif flood_fill_entry_point < img_w + img_h:
        print('right')
        flood_fill_entry_coords = (img_w - 1, flood_fill_entry_point - img_w + 1)
    elif flood_fill_entry_point < img_w * 2 + img_h:
        print('bottom')
        flood_fill_entry_coords = (img_h + (2 * img_w) - 3 - flood_fill_entry_point, img_h - 1)
    else:
        print('left')
        flood_fill_entry_coords = (0, perimeter - flood_fill_entry_point)

    flood_fill_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)

    print(flood_fill_entry_point)
    print(flood_fill_entry_coords)
    # noinspection PyUnresolvedReferences
    cv2.floodFill(mask, flood_fill_mask, tuple(flood_fill_entry_coords), 255)

    return mask
