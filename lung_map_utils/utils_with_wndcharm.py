from wndcharm.FeatureVector import FeatureVector
import pandas as pd
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