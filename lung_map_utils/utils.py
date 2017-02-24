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
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFdr
from wndcharm.FeatureVector import FeatureVector
from scipy.spatial.distance import pdist
import warnings

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
    ]
}


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


def fill_holes(mask):
    """
    Fills holes in a given binary mask.
    """
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    new_mask, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        cv2.drawContours(new_mask, [cnt], 0, 255, -1)

    return new_mask


def filter_contours_by_size(mask, min_size=1024, max_size=None):
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
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


def get_hsv(hsv_img):
    """
    Returns flattened hue, saturation, and values from given HSV image.
    """
    hue = hsv_img[:, :, 0].flatten()
    sat = hsv_img[:, :, 1].flatten()
    val = hsv_img[:, :, 2].flatten()

    return hue, sat, val


def get_color_profile(hsv_img):
    """
    Finds color profile as pixel counts for color ranges in HSV_RANGES

    Args:
        hsv_img: HSV pixel data (3-D NumPy array)

    Returns:
        Text string for dominant color range (from HSV_RANGES keys)

    Raises:
        tbd
    """
    h, s, v = get_hsv(hsv_img)

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


def get_target_features(region, rect_index):
    tmp_tiff = 'rect_%d.tif' % rect_index

    region.save(tmp_tiff)

    target_fv = FeatureVector(name='FromTiff', long=True, color=True, source_filepath=tmp_tiff)
    target_fv.GenerateFeatures(quiet=True, write_to_disk=False)

    target_features = pd.Series(target_fv.values, index=target_fv.feature_names)

    os.remove(tmp_tiff)

    return target_features


def get_custom_target_features(hsv_img):
    color_features = get_custom_color_features(hsv_img)

    feature_names = []
    values = []

    # add region features first
    feature_names.append('region_area')
    values.append(hsv_img.shape[0] * hsv_img.shape[1])

    feature_names.append('region_saturation_mean')
    values.append(np.mean(hsv_img[:, :, 1]))

    feature_names.append('region_saturation_variance')
    values.append(np.var(hsv_img[:, :, 1]))

    feature_names.append('region_value_mean')
    values.append(np.mean(hsv_img[:, :, 2]))

    feature_names.append('region_value_variance')
    values.append(np.var(hsv_img[:, :, 2]))

    for color, features in color_features.iteritems():
        for feature, value in sorted(features.iteritems()):
            feature_str = '%s (%s)' % (feature, color)
            feature_names.append(feature_str)
            values.append(value)

    target_features = pd.Series(values, index=feature_names)

    return target_features.sort_index()


def predict(input_dict):
    rect_index = input_dict['rect_index']
    region = input_dict['region']
    trained_model = input_dict['model']
    class_map = input_dict['class_map']
    custom = input_dict['custom']

    if custom:
        target_features = get_custom_target_features(region)
    else:
        # extract wnd-charm features from region
        target_features = get_target_features(region, rect_index)

    # classify target features using training data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        target_prediction = trained_model.predict(target_features)

        if class_map[target_prediction[0]] == 'background':
            return rect_index, class_map[target_prediction[0]], None
        try:
            target_prob = trained_model.predict_proba(target_features)
        except Exception:  # TODO: check which exception is raised
            target_prob = None

    if target_prob is not None:
        probabilities = {}
        for i, prob in enumerate(target_prob[0]):
            probabilities[class_map[i + 1]] = prob

        sorted_probabilities = sorted(probabilities.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_probabilities = None

    return rect_index, class_map[target_prediction[0]], sorted_probabilities


def get_custom_color_features(hsv_img, show_plots=False):
    c_prof = get_color_profile(hsv_img)

    tot_px = hsv_img.shape[0] * hsv_img.shape[1]

    color_features = {}

    for color in HSV_RANGES.keys():
        color_percent = float(c_prof[color]) / tot_px

        # create color mask
        mask = create_mask(hsv_img, [color])

        mask_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.title(color)
            ax1.imshow(cv2.cvtColor(mask_img, cv2.COLOR_HSV2RGB))
            ax2.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
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
            area = cv2.contourArea(c)

            if area <= 0.0:
                continue

            peri = cv2.arcLength(c, True)

            if area > largest_contour_area:
                largest_contour_area = area
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
            box = cv2.minAreaRect(largest_contour)
            cnt_w, cnt_h = box[1]

            largest_contour_eccentricity = cnt_w / cnt_h
            if largest_contour_eccentricity < 1:
                largest_contour_eccentricity = 1.0 / largest_contour_eccentricity

            # calculate inverse circularity as 1 / (area / perimeter ^ 2)
            largest_contour_circularity = largest_contour_peri / np.sqrt(largest_contour_area)

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
