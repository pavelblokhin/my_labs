import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        index = np.random.choice(X.shape[0], size=(1000000, 2), replace=True)
        distance_squares = np.sum((X[index[:, 0]] - X[index[:, 1]]) ** 2, axis=1)
        sigma = np.median(distance_squares)
        self.w = 1 / np.sqrt(sigma) * np.random.normal(0, 1, size=(self.n_features, self.new_dim)) # я не знаю почему не работает np.random.normal(0, 1 / sigma) это бред
        self.b = np.random.uniform(-np.pi, np.pi, size=(self.n_features, 1))
        return self

    def transform(self, X, y=None):
        transform_X = self.func(self.w @ X.T + self.b).T
        return transform_X


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        index = np.random.choice(X.shape[0], size=(1000000, 2), replace=True)
        distance_squares = np.sum((X[index[:, 0]] - X[index[:, 1]]) ** 2, axis=1)
        sigma = np.sqrt(np.median(distance_squares))
        
        copies = int(np.ceil(self.n_features / self.new_dim))
        blocks = []
        for _ in range(copies):
            G = np.random.normal(0, 1, size=(self.new_dim, self.new_dim))
            Q, R = np.linalg.qr(G)
            diag = np.sqrt(np.random.chisquare(self.new_dim, self.new_dim))
            S = np.diag(diag)
            blocks.append(1 / sigma * (S @ Q))

        self.w = np.vstack(blocks)[:self.n_features, :]
        self.b = np.random.uniform(-np.pi, np.pi, (self.n_features, 1))
        return self


class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        if not self.use_PCA: # как бы надо self чтобы достать инфу
            self.new_dim = X.shape[1]
            self.feature_creator.new_dim = X.shape[1]
        pipeline_steps: list[tuple] = [
            ('standart scaler', StandardScaler()), # для метода опорных векторов нужен
            ('PCA', PCA(n_components=self.new_dim)),
            ('RFF', self.feature_creator),
            ('classifier', self.classifier)
        ]
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
