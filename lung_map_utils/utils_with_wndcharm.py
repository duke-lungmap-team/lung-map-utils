from wndcharm.FeatureVector import FeatureVector
import warnings
from .utils import get_custom_target_features
import pandas as pd
import operator
import os


def get_target_features(region, rect_index):
    tmp_tiff = 'rect_%d.tif' % rect_index

    region.save(tmp_tiff)

    target_fv = FeatureVector(name='FromTiff', long=True, color=True, source_filepath=tmp_tiff)
    target_fv.GenerateFeatures(quiet=True, write_to_disk=False)

    target_features = pd.Series(target_fv.values, index=target_fv.feature_names)

    os.remove(tmp_tiff)

    return target_features


def get_target_features_from_tif(tif_file):
    target_fv = FeatureVector(name='FromTiff', long=True, color=True, source_filepath=tif_file)
    target_fv.GenerateFeatures(quiet=True, write_to_disk=False)

    target_features = pd.Series(target_fv.values, index=target_fv.feature_names)

    return target_features


def predict(input_dict):
    rect_index = input_dict['rect_index']
    region = input_dict['region']
    trained_model = input_dict['model']
    class_map = input_dict['class_map']
    custom = input_dict['custom']
    contour_mask = input_dict['contour_mask']
    hsv_img = input_dict['hsv_img']

    if custom and contour_mask is None:
        target_features = get_custom_target_features(region)
    elif custom and contour_mask is not None:
        target_features = get_custom_target_features(hsv_img, mask=contour_mask)
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
