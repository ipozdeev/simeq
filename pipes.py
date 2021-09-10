import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .equations import Equations


def to_values(func):
    """Extract values from dataframes."""
    def wrapper(*args, **kwargs) -> tuple:
        return tuple(x_.values for x_ in func(*args, **kwargs))
    return wrapper


class CrossSectionDataset(Dataset):
    """
    Parameters
    ----------
    equations : Equations
    """

    def __init__(self, equations):

        self.x, self.y, self.w = equations.stack(dropna=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_ = self.x[item]
        y_ = self.y[item]
        w_ = self.w[item]

        return x_, y_, w_


class BalancedPanelDataset(Dataset):
    """Dataset representing y_t ~ x_t.

    `torch.utils.data.Dataset` implementation to regress whole rows in y onto
    multiple rows in x.

    Parameters
    ----------
    exog : pandas.DataFrame
        dimensions TxN
    endog : pandas.DataFrame
        dimensions Tx(NK), where K is an integer
    lookback_exog : int
    lookfwd_endog : int
    map_exog : callable
    """
    def __init__(self, exog, endog, lookback_exog, lookfwd_endog,
                 map_exog=None, zero_na=False):
        """
        """
        assert exog.index.equals(endog.index)

        self.exog = exog
        self.endog = endog
        self.lookback_exog = lookback_exog
        self.lookfwd_endog = lookfwd_endog
        self.map_exog = (lambda x: x) if map_exog is None else map_exog
        self.zero_na = zero_na

    def __len__(self):
        return len(self.endog)

    def __getitem__(self, item):
        if (item < (self.lookback_exog - 1)) | \
                (item > len(self) - self.lookfwd_endog):
            return self.__getitem__(
                np.random.randint(self.lookback_exog,
                                  len(self) - self.lookfwd_endog)
            )
        x_ = self.exog.iloc[:(item+1)].tail(self.lookback_exog)\
            .pipe(self.map_exog)\
            .replace(np.nan, 0.0 if self.zero_na else np.nan)\
            .values.reshape(1, -1)
        y_ = self.endog.iloc[item:].head(self.lookfwd_endog) \
            .replace(np.nan, 0.0 if self.zero_na else np.nan) \
            .values

        return x_, y_


class CarryDataset(Dataset):
    """Sorted portfolio stuff.

    Parameters
    ----------
    pfsort : pandas.DataFrame
        of dim T x (NM) where M is the number of portfolios
    rx : pandas.DataFrame
        forward-looking rx, of maturity `lookforward`
    rs : pandas.DataFrame
        forward-looking rs, daily
    exog
    lookforward : int
        investment horizon, in units of frequency of the index
    lookback : int
    """
    def __init__(self, pfsort, rx, rs, exog, lookforward, lookback=1):
        self.pfsort = pfsort
        self.rx = rx
        self.rs = rs
        self.exog = exog
        self.lookforward = lookforward
        self.lookback = lookback

    def __len__(self):
        return len(self.exog)

    def __getitem__(self, item):
        if (item < self.lookback) | \
                (item > len(self) - self.lookforward):
            return self.__getitem__(
                np.random.randint(self.lookback,
                                  len(self) - self.lookforward)
            )
        x_ = self.exog.iloc[:(item+1)].tail(self.lookback)\
            .values.reshape(1, -1)

        # sorted stuff
        w_ = self.pfsort.iloc[item]
        rs_ = self.rs.iloc[item:].head(self.lookforward) \
            .mul(w_, axis=1).sum(axis=1, level=0) \
            .values
        rx_ = self.rx.iloc[[item]] \
            .mul(w_, axis=1).sum(axis=1, level=0) \
            .values

        return x_, rs_, rx_


class TermStructureGranularDataSet(Dataset):
    """Term structure for ANN, with transformations.

    Parameters
    ----------
    ts : DataFrame
        with time for the index, (asset, maturity) for the columns

    """
    def __init__(self, ts: pd.DataFrame, lookback: int):
        self.ts = ts
        self.lookback = lookback
        self.assets = ts.columns.unique(0)
        self.n_periods = len(ts)
        self.n_assets = len(self.assets)

    def __len__(self):
        return (self.n_periods - self.lookback + 1) * self.n_assets

    @to_values
    def __getitem__(self, item) -> tuple:
        t, n = item // self.n_assets, item % self.n_assets
        x = self.ts.iloc[t:(t+self.lookback)][self.assets[n]]
        x = x.sub(x.iloc[-1, 0])

        return (x, )


class ExcessReturnsDataset(Dataset):
    """

    Parameters
    ----------
    rx : DataFrame
    rs : DataFrame
    lookforward : int
    """

    def __init__(self, rx: pd.DataFrame, rs: pd.DataFrame, lookforward: int):

        if not all([rx.index.equals(rs.index), rx.columns.equals(rs.columns)]):
            raise ValueError("`rx` and `rs` are not aligned.")

        self.rx = rx
        self.rs = rs
        self.lookforward = lookforward

    def __len__(self):
        return len(self.rx) - self.lookforward + 1

    @to_values
    def __getitem__(self, item):
        rx_ = self.rx.iloc[[item]]
        rs_ = self.rs.iloc[item:].head(self.lookforward)

        return rx_, rs_


class BackwardForwardDataset(Dataset):

    def __init__(self, backward_ds, forward_ds):
        self.backward_ds = backward_ds
        self.forward_ds = forward_ds

    def __len__(self):
        return len(self.backward_ds) - self.forward_ds.lookforward

    def __getitem__(self, item) -> tuple:
        res = (*self.backward_ds.__getitem__(item),
               *self.forward_ds.__getitem__(item+self.backward_ds.lookback-1))

        return res


class BackwardDataset(Dataset):

    def __init__(self, ds: pd.DataFrame, lookback: int):

        self.ds = ds
        self.lookback = lookback

    def __len__(self):
        return len(self.ds) - self.lookback + 1

    @to_values
    def __getitem__(self, item) -> tuple:
        x = self.ds.iloc[:(item + self.lookback)].tail(self.lookback)
        # x = x.sub(x.groupby(axis=1, level=0).apply(lambda z_: z_.iloc[-1, 0]),
        #           axis=1, level=0)

        return (x,)


class TermStructureDataSet(BackwardDataset):
    """Term structure for CNN, with transformations."""
    def __init__(self, ds, lookback: int, transform: callable = None):
        if transform is None:
            transform = lambda x: x

        super(TermStructureDataSet, self).__init__(ds, lookback)

        self.transform = transform

    @to_values
    def __getitem__(self, item) -> tuple:
        x = self.ds.iloc[:(item + self.lookback)].tail(self.lookback)\
            .pipe(self.transform)
        # x = x.sub(x.groupby(axis=1, level=0).apply(lambda z_: z_.iloc[-1, 0]),
        #           axis=1, level=0)

        return (x, )


class ForwardDataset(Dataset):

    def __init__(self, ds: pd.DataFrame, lookforward: int):

        self.ds = ds
        self.lookforward = lookforward

    def __len__(self):
        return len(self.ds) - self.lookforward + 1

    @to_values
    def __getitem__(self, item) -> tuple:
        x = self.ds.iloc[item:].head(self.lookforward)

        return (x, )


if __name__ == '__main__':
    pass
