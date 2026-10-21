import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class CappingTransformer(BaseEstimator, TransformerMixin):
    """Caps selected columns at a percentile learned from the training data."""

    def __init__(self, cols, percentile=0.99):
        self.cols = cols
        self.percentile = percentile

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.caps_ = {}
        for c in self.cols:
            self.caps_[c] = X[c].quantile(self.percentile)
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].clip(upper=self.caps_[c])
        return X


def build_preprocessor(
    cols_to_cap: list[str], numeric_cols: list[str], percentile: float = 0.99
) -> ColumnTransformer:
    """Returns a preprocessor that caps selected columns at a percentile and imputes missing values in numeric columns."""

    numeric_transformer = Pipeline(
        steps=[
            ("capping", CappingTransformer(cols=cols_to_cap, percentile=percentile)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num_preprocessing", numeric_transformer, numeric_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preprocessor
