import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn.base import TransformerMixin

from .equations import Equations


class EquationsScaler:
    """Wrapper around preprocessing methods of `sklearn`.

    Parameters
    ----------
    x_scaler : str
    x_scaler : str
    skip_x : str or list-like

    """
    def __init__(self, x_scaler=None, y_scaler=None, skip_x=None):

        if skip_x is None:
            skip_x = []
        elif not hasattr(skip_x, "__iter__"):
            skip_x = [skip_x, ]

        def switch_scaler(scaler_id):
            if scaler_id is None:
                aux_res = CustomScaler
            elif isinstance(scaler_id, str):
                if scaler_id == "rank":
                    aux_res = CustomRankScaler
                elif scaler_id == "zscore":
                    aux_res = CustomStandardScaler
                elif scaler_id == "zscore_xs":
                    aux_res = CustomStandardScalerXS
                elif scaler_id == "above_median":
                    aux_res = CustomThresholdScaler
                else:
                    raise NotImplementedError
            else:
                aux_res = scaler_id

            return aux_res

        # these will be applied to regressors and responses
        self._x_scaler_one = switch_scaler(x_scaler)
        self._y_scaler_one = switch_scaler(y_scaler)

        self._x_scaler = None
        self._y_scaler = None
        self._skip_x = skip_x

    @property
    def x_scaler(self):
        return self._x_scaler

    @x_scaler.setter
    def x_scaler(self, value):
        assert isinstance(value, dict)
        self._x_scaler = value

    @property
    def y_scaler(self):
        return self._y_scaler

    @y_scaler.setter
    def y_scaler(self, value):
        self._y_scaler = value

    def fit(self, equations):
        """Fit scalers to x and y.

        Parameters
        ----------
        equations : Equations

        Returns
        -------
        EquationsScaler

        """
        # apply within each characteristic
        x_scl = dict()

        for k, v in equations.x.groupby(axis=1, level=equations.reg_level):
            x_scl[k] = self._x_scaler_one().fit(v)

        y_scl = self._y_scaler_one().fit(equations.y)

        self.x_scaler = x_scl
        self.y_scaler = y_scl

    def fit_transform(self, x, y=None):
        """Fit to x, transform it, plus transform another Equations if asked.

        Parameters
        ----------
        x : Equations
        y : Equations

        Returns
        -------
        Equations

        """
        self.fit(x)

        if y is None:
            return self.transform(x)

        else:
            return self.transform(x), self.transform(y)

    def transform(self, equations):
        """Transform both y and x.

        Parameters
        ----------
        equations : Equations

        Returns
        -------
        Equations

        """
        # apply within each characteristic
        x_scl = list()

        for k, v in equations.x.groupby(axis=1, level=equations.reg_level):
            if k not in self._skip_x:
                x_scl.append(pd.DataFrame(self.x_scaler[k].transform(v),
                                          index=v.index,
                                          columns=v.columns))
            else:
                x_scl.append(v)

        x_scl = pd.concat(x_scl, axis=1)

        # simply apply to y
        y_scl = pd.DataFrame(self.y_scaler.transform(equations.y.copy()),
                             index=equations.y.index,
                             columns=equations.y.columns)

        res = Equations(y=y_scl, x=x_scl, weight=equations.weight)

        return res

    def inverse_transform(self, equations):
        """Inverse transform y.

        Parameters
        ----------
        equations : Equations

        Returns
        -------
        res : Equations

        """
        y_it = pd.DataFrame(self.y_scaler.inverse_transform(equations.y),
                            index=equations.y.index,
                            columns=equations.y.columns)

        return y_it


class EWMAScaler:
    """
    """
    def __init__(self, estimate_mu=True, estimate_sd=True, **kwargs):
        """
        """
        self.ewm_par = kwargs
        self.estimate_mu = estimate_mu
        self.estimate_sd = estimate_sd
        self.data = None

        self.mu = 0.0
        self.sd = 1.0

    def fit(self, data, shift=False):
        """

        Parameters
        ----------
        data : pandas.DataFrame
        shift : bool

        Returns
        -------
        None

        """
        if isinstance(shift, bool):
            lag = 1 if shift else 0
        elif isinstance(shift, (int, np.int)):
            lag = shift
        else:
            raise ValueError("Wrong value type for 'shift'.")

        if self.estimate_mu:
            self.mu = data.ewm(**self.ewm_par).mean().shift(lag)

        if self.estimate_sd:
            self.sd = data.ewm(**self.ewm_par).std().shift(lag)

        self.data = data

    def transform(self, data_new):
        """

        Parameters
        ----------
        data_new

        Returns
        -------

        """
        # copy mu, sd
        mu = self.mu + 0.0
        sd = self.sd + 0.0

        # merge `data_new` with the data used to fit
        data_merged = pd.concat((self.data, data_new), axis=0)

        # re-estimate mu, sd
        if self.estimate_mu:
            mu = data_merged.ewm(**self.ewm_par).mean()
        if self.estimate_sd:
            sd = data_merged.ewm(**self.ewm_par).std()

        # transform the new data
        data_transf = (data_new - mu) / sd

        return data_transf

    def fit_transform(self, data, shift=False):
        """

        Parameters
        ----------
        data
        shift : bool or int

        Returns
        -------

        """
        # fit first
        self.fit(data, shift)

        # transform using the whole period-by-period frames, not just the
        # last row
        data_transf = (data - self.mu) / self.sd

        return data_transf

    def __str__(self):
        """

        Returns
        -------

        """
        res = "\n".join((
                "EWMScaler", "---------",
                "mu", "--", str(self.mu),
                "sd", "--", str(self.sd)
            ))

        return res


class RollingQuantileScaler:
    """
    Parameters
    ----------
    n_quantiles
    kwargs
    """
    def __init__(self, n_quantiles, **kwargs):
        """
        """
        self.n_quantiles = n_quantiles
        self.labels = list(range(n_quantiles))
        self.rolling_kwargs = kwargs

    def transform(self, data):
        """

        Parameters
        ----------
        data : pandas.DataFrame

        Returns
        -------

        """
        def roll_q(df):
            qs = np.percentile(df, np.linspace(0, 1, self.n_quantiles)*100)
            aux_res = np.digitize(df[-1], bins=qs, right=True)
            return aux_res

        res = data.rolling(**self.rolling_kwargs).apply(roll_q)

        return res


class CustomScaler(TransformerMixin):

    def fit(self, *args, **kwargs):
        return self

    def transform(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        return x

    def inverse_transform(self, x):
        return x


class CustomRankScaler(CustomScaler):
    """Rank observations

    Parameters
    ----------
    time_consistent : bool
        True to ensure the highest/lowest ranks are the same irrespective
        of the number of complete columns
    """
    def __init__(self, time_consistent=True):
        super().__init__()
        self.time_consistent = time_consistent

    def transform(self, x):
        """Rank observations.

        Parameters
        ----------
        x : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame

        """
        if self.time_consistent:
            # ensure ranks are consistent in the time history
            med = pd.concat([x.median(axis=1), ] * x.shape[1],
                            axis=1,
                            keys=x.columns)
            x_to_rank = x.fillna(med)
        else:
            x_to_rank = x

        res = x_to_rank \
            .rank(axis=1, pct=True) \
            .where(x.notnull())

        return res


class CustomThresholdScaler(CustomScaler):
    """Binary whether variable is higher than the x-sectional quantile.

    Parameters
    ----------
    q : float
        quantile to use as threshold
    """
    def __init__(self, q=0.5):
        super(CustomThresholdScaler, self).__init__()
        self.q = q

    def transform(self, x):
        """Map observations to quantiles.

        Parameters
        ----------
        x : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame

        """
        res = x \
            .gt(x.quantile(self.q, axis=1), axis=0).astype(float) \
            .where(x.notnull())

        return res


class CustomStandardScaler(pp.StandardScaler):
    pass


class CustomStandardScalerXS(pp.StandardScaler):

    def fit(self, x, y=None):
        return super().fit(x.T, y=(None if y is None else y.T))

    def transform(self, x, copy=None):
        # disregard the previously calculated values
        self.fit(x)
        return super().transform(x.T).T

    def fit_transform(self, x, y=None, **fit_params):
        # disregard the previously calculated values
        self.fit(x)
        return super().transform(x.T).T

    def inverse_transform(self, x, copy=None):
        return x


class CustomRollingStandardScaler(CustomScaler):
    pass
