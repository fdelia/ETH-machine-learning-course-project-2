from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

class RandomBinsExtraction(BaseEstimator, TransformerMixin):
    """Build n bins with mean from values"""
    def __init__(self, splits=1000, hist_bins=None,
        images_x_from=None, images_x_to=None,
        images_y_from=None, images_y_to=None):

        self.splits = splits
        self.hist_bins = hist_bins


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = []
        if self.hist_bins is None:
            self.hist_bins = [[1, 298.8218117717887, 415.59953383322301, 529.00853569772312, 637.92188924633831, 741.69761514701986, 840.16357827183447, 936.05487410467663, 1034.8715039519698, 1139.5417907370197, 1246.5076326626559, 1350.2613661802782, 1451.0932380428474, 1700]]

        first = True
        for row in X:
            row = row[600000:5700000] # has no real influence since those are all 0s, just for performance
            # This is feature selection
            # if self.images_x_from is not None and self.images_x_to is not None:
            #     images = np.split(row, 176)[self.images_x_from : self.images_x_to]

                # x needs to be set for this, but don't mind at the moment
                # if self.images_y_from is not None and self.images_y_to is not None:
                #     images_new = []
                #     for image in images:
                #         images_new.append(np.split(image, 208)[self.images_y_from : self.images_y_to])
                #     images = np.array(images_new)
                #
                # row = np.array(images).flatten()

            splits = np.array_split(row, int(self.splits))

            features = []
            for j, split in enumerate(splits):
                i = int(j / len(splits) * len(self.hist_bins))

                features.append(np.histogram(split, bins=self.hist_bins[i])[0])

            X_new.append(np.array(features).flatten())

        if first:
            #print("features: "+str(len(features)))
            first = False

        return X_new

class Run(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipe = None

    def sampleIt(self, X, y):
        X_new = []; y_new = []; sample_w = []
        for i, row_y in enumerate(y):
            for cl, p_i in enumerate(row_y):
                X_new.append(X[i])
                y_new.append(cl)
                sample_w.append(p_i)
        return X_new, y_new, sample_w

    def fit(self, X, y=None):
        if y is not None:
            X, y, weights = self.sampleIt(X, y)

        pipe = Pipeline([
            ('BinsExtraction', RandomBinsExtraction(splits=1000)),
            #('vct', VarianceThreshold(threshold=5.0)),
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(C=1.0, solver='newton-cg', n_jobs=-1))
            #('linearSVC', LinearSVC(C=1.0, max_iter=1000))
        ])
        pipe.fit(X, y, **{'logreg__sample_weight': weights})
        self.pipe = pipe
        return self

    def transform(self, X, y=None):
        return X

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)
    def predict(self, X):
        return self.pipe.predict(X)
