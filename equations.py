import pandas as pd
import numpy as np
from collections import OrderedDict


class Equations:
    """Equations system.

    The system has N equations, each featuring 1 dependent variable and
    several (possibly a different number for each equation) regressors.

    Dependent variables are collected into a `pandas.DataFrame` with column
    names corresponding to equation names; regressors are collected into a
    `pandas.DataFrame` columned by a `pandas.MultiIndex` of (equation_name,
    regressor), such that each individual equation can be written as:
        y.loc[:, c] ~ x.loc[:, (c, slice(None))]

    Parameters
    ----------
    y : pandas.DataFrame
        with equation names for columns
    x : pandas.DataFrame
        with multiindex (equation_name, regressor_name) for columns
    weight : numpy.array
        of sample weights, passed on as `sample_weights` to `keras.model.fit`

    """
    def __init__(self, y, x, weight=None):
        """
        """
        if isinstance(x, dict):
            x = pd.concat(x, axis=1, names=["equation", "regressor"])

        if isinstance(x, pd.Series):
            x = pd.concat([x] * y.shape[1], keys=y.columns, axis=1)

        if x.columns.nlevels < 2:
            x = pd.concat([x], keys=["regressor"], axis=1).swaplevel(axis=1)

        # makes sure all ys have regressors ---------------------------------
        assert all(y.columns.isin(x.columns.droplevel(-1)))

        # handle multilevels ------------------------------------------------
        # TODO: handle case when y.columns.nlevels = 2, but 1 for x
        if y.columns.nlevels > 1:
            y.columns = ["_".join(c) for c in y.columns]
            x.columns = pd.MultiIndex.from_arrays(
                [["_".join(c) for c in x.columns.droplevel("regressor")],
                 x.columns.get_level_values("regressor")],
                names=["equation", "regressor"]
            )

        # weight ------------------------------------------------------------
        if isinstance(weight, (list, np.ndarray)):
            weight = pd.Series(index=x.index, data=weight)

        # store original values ---------------------------------------------
        y_orig = y.copy()
        x_orig = x.copy()

        # set names on levels -----------------------------------------------
        y_orig, x_orig = self.set_index_names(y_orig, x_orig)

        # align on the index axis -------------------------------------------
        y_aln, x_aln = y_orig.align(x_orig, axis=0, join="inner")
        x_aln = x_aln.reindex(columns=y_aln.columns, level=0)

        # sort indices to ease conversion to numpy later
        y_aln = y_aln.sort_index(axis=1)
        x_aln = x_aln.sort_index(axis=1)

        self.y_orig = y_orig
        self.x_orig = x_orig
        self.y = y_aln
        self.x = x_aln
        self.weight = weight.loc[x_aln.index] if weight is not None else None
        self.weigh_flag = weight is not None
        self.eq_level = y_orig.columns.name
        self.reg_level = x_orig.columns.names[1]

        self.constraints = None

    @property
    def equations(self):
        return self.y.columns

    @property
    def regressors(self):
        return self.x.columns.unique(1)

    @staticmethod
    def set_index_names(y, x):
        """Set level names.

        Parameters
        ----------
        y : pandas.DataFrame
        x : pandas.DataFrame

        Returns
        -------
        y : pandas.DataFrame
        x : pandas.DataFrame

        """
        if y.columns.name is None:
            if x.columns.names[0] is None:
                y.columns.name = "equation"
            else:
                y.columns.name = x.columns.names[0]

        if x.columns.names[1] is None:
            x.columns.names = [y.columns.name, "regressor"]
        else:
            x.columns.names = [y.columns.name, x.columns.names[1]]

        return y, x

    def insert_constant(self):
        """Append a vector of ones to each set of regressors.

        Returns
        -------
        Equations

        """
        return self.insert_into_x(const=pd.Series(1, index=self.x.index))

    def insert_into_x(self, **regressors):
        """Append new variables to each set of regressors.

        Parameters
        ----------
        regressors : pandas.DataFrame or pandas.Series

        Returns
        -------
        None

        """
        for x_name, x_ in regressors.items():
            if isinstance(x_, pd.Series):
                x_df = pd.concat([x_] * len(self.equations), axis=1,
                                 keys=self.equations)
                return self.insert_into_x(**{x_name: x_df})

            elif isinstance(x_, pd.DataFrame):
                x_df = pd.concat([x_, ], axis=1, keys=[x_name, ])\
                    .swaplevel(axis=1)
                x_new = self.x.join(x_df, how="left").sort_index(axis=1)
                return Equations(self.y, x_new, self.weight)

            else:
                raise NotImplementedError

    def __iter__(self):
        """
        """
        # concat x and y to be able to efficiently loop over rows
        xy = pd.concat(
            (self.x.stack(level=self.eq_level),
             self.y.stack(level=self.eq_level).rename("y")),
            axis=1)

        # loop
        for row in xy.values:
            yield row[:-1], row[-1]

    @staticmethod
    def complete_windows(x, y, window):
        """

        Parameters
        ----------
        x : pandas.DataFrame
        y : pandas.DataFrame
        window : int

        Returns
        -------
        pandas.DataFrame
`
        """
        date_asset = pd.concat(
            (x,
             pd.concat([y, ], axis=1, keys=["response"])
                .swaplevel(axis=1)
             ),
            axis=1)

        # count cases
        cnt = date_asset.rolling(window, min_periods=window).count()

        complete_x = cnt.drop("response", axis=1, level=1)\
            .eq(window).all(axis=1, level=0)
        complete_y = date_asset.xs("response", axis=1, level=1).notnull()
        res = complete_x & complete_y

        res = res.where(res).stack(dropna=True)

        res.index.names = ["date", "equation"]

        res = res.reset_index(level=1).loc[:, "equation"]

        return res

    def stack(self, dropna=False):
        """

        Parameters
        ----------
        dropna : bool
            similar to `pandas.DataFrame.stack(*, dropna)`

        Returns
        -------
        x : numpy.ndarray
        y : numpy.ndarray
        w : numpy.ndarray
            weights
        """
        x = self.x.stack(level=self.eq_level, dropna=False)
        y = self.y.stack(level=self.eq_level, dropna=False).to_frame()

        if dropna:
            x, y = x.dropna().align(y.dropna(), axis=0, join="inner")
        else:
            x, y = x.align(y, axis=0, join="inner")

        if self.weight is not None:
            w = self.weight.reindex(index=x.index, level=0).values
            # w = w / w[-1]
        else:
            w = None

        return x.values, y.values, w

    def to_generator(self, min_batch_len=None):
        """

        Returns
        -------

        """
        if min_batch_len is None:
            min_batch_len = len(self.equations)

        # stack y, x, and weights together to loop through rows of this
        # dataframe
        date_asset = pd.concat(
            (self.x,
             pd.concat([self.y, ], axis=1, keys=["response"])
                .swaplevel(axis=1),
             # pd.concat([self.weight.to_frame() if self.weight is not None
             #            else pd.Series(1, index=self.x.index).to_frame(), ],
             #           axis=1,
             #           keys=["wght"])
             ),
            axis=1)
        batches = date_asset.stack(level=self.eq_level).dropna()

        # count complete cases
        batches_count = batches.groupby(axis=0, level=0).count().iloc[:, 0]

        # select rows where at least `min_batch_len` of assets have all
        #   features in place
        idx_good = batches_count.index[batches_count.ge(min_batch_len)]

        while True:
            dt = np.random.choice(idx_good)
            ys = batches.loc[dt]["response"].values
            xs = batches.loc[dt].drop(["response"], axis=1).values
            ws = [self.weight.loc[dt] if self.weight is not None else 1] *\
                len(ys)

            yield xs, ys, ws

    def unstack(self, y_new, dropna=False):
        """Reshape from the squeezed form w/o NA back to a panel.

        `y_new` would most likely be result of a `predict()` method on
        `self.x`, and hence a 1D array. This reshapes it back to the shape
        of `self.y`.

        Parameters
        ----------
        y_new : numpy.ndarray
        dropna : bool

        Returns
        -------
        pandas.DataFrame
            of the same shape as `self.y`

        """
        x = self.x.stack(level=self.eq_level, dropna=False)
        y = self.y.stack(level=self.eq_level, dropna=False)

        if dropna:
            _, y = x.dropna().align(y.dropna(), axis=0, join="inner")
        else:
            _, y = x.align(y, axis=0, join="inner")

        y_new = pd.Series(index=y.index, data=y_new.squeeze())
        # y.loc[:] = y_new.squeeze()
        y_unstk = y_new.unstack()

        res = y_unstk.reindex(index=self.y.index)

        return res

    def split(self, ratio=1, loc=None):
        """Split into train and test samples.

        Parameters
        ----------
        ratio : float or array-like
            can be negative
        loc
            optional, index to split on - becomes the last index of the
            training sample, so the next valid index starts off the test sample

        Returns
        -------
        Equations
        Equations

        """
        if loc is not None:
            y_new_train = self.y.loc[:loc]
            x_new_train = self.x.loc[:loc]
            w_new_train = (self.weight.loc[:loc] if self.weight is not None
                           else None)
            y_new_test = self.y.loc[loc:]
            x_new_test = self.x.loc[loc:]
            w_new_test = (self.weight.loc[loc:] if self.weight is not None
                          else None)

        else:
            if ratio < 0:
                return self.split(1 + ratio)[::-1]

            # determine split location
            idx = int(np.ceil(len(self.y) * ratio))

            y_new_train = self.y.iloc[:idx]
            x_new_train = self.x.iloc[:idx]
            w_new_train = (self.weight.iloc[:idx] if self.weight is not None
                           else None)
            y_new_test = self.y.iloc[idx:]
            x_new_test = self.x.iloc[idx:]
            w_new_test = (self.weight.iloc[idx:] if self.weight is not None
                          else None)

        res = (
            Equations(y_new_train, x_new_train, weight=w_new_train),
            Equations(y_new_test, x_new_test, weight=w_new_test)
        )

        return res

    def add_constraints(self, r=None, q=None):
        """Add constraints of the form rB = q

        Parameters
        ----------
        r : pandas.DataFrame
        q : pandas.Series

        Returns
        -------

        """
        r = r.reindex(columns=self.x.columns).fillna(0.0)

        if q is None:
            q = pd.Series(0.0, index=r.index)

        self.constraints = {"r": r, "q": q}

    def make_equal_coef_constraints(self, column_mapper=None, grouper=None):
        """Make DataFrame of constraints imposing equality of coefficients.

        Those regressors which belong to the same group, as dictated by
        `grouper` of `column_mapper` will have their coefficients imposed to
        be equal.

        Parameters
        ----------
        column_mapper : callable (optional)
            many-to-fewer transformation to arrive at a grouper
        grouper : pandas.Index (optional)
            grouper

        Returns
        -------
        None

        """
        if grouper is None:
            if not isinstance(column_mapper, (list, tuple)):
                column_mapper = [column_mapper, ]

            reg_level_grouped = self.x.columns.get_level_values("regressor")

            for func in column_mapper:
                reg_level_grouped = reg_level_grouped.map(func)

            grouper = pd.Series(index=self.x.columns, data=reg_level_grouped)

        # make constraints
        constraints = list()

        # the idea is to fill the columns corresponding to the coefficients
        #   which are imposed to be equal with +1 and -1, and to fill the
        #   rest with zeros
        for _, grp in self.x.groupby(axis=1, by=grouper):
            n = grp.shape[1]
            constraints_tmp = pd.DataFrame(
                data=np.hstack((np.ones((n - 1, 1)), -1 * np.eye(n - 1))),
                columns=grp.columns)
            constraints.append(constraints_tmp)

        # concat
        constraints = pd.concat(constraints, axis=0) \
            .reindex(columns=self.x.columns) \
            .fillna(0.0) \
            .reset_index(drop=True)

        # add to self
        self.add_constraints(r=constraints)

    def for_linearmodels(self, estimator, **kwargs):
        """Represent in a form suitable for Sheppard's linearmodels.

        Parameters
        ----------
        estimator : callable
            one of linearmodels' estimators, e.g. PanelOLS, IVSystemGMM or SUR
        kwargs
            arguments to `estimator`, e.g. weights, entity_effects etc.

        Returns
        -------
        Union[lm.panel.model._LSSystemModelBase,
              lm.panel.model._PanelModelBase]

        """
        assert callable(estimator)

        # differentiate between system (e.g. SUR) and panel (e.g. PanelOLS)
        #   estimators
        if estimator.__module__ == "linearmodels.system.model":
            # e.g. SUR
            y_, x_ = self.y.stack().dropna()\
                .align(self.x.stack(level=0).dropna(), join="inner")

            y_ = y_.unstack()
            x_ = x_.unstack(level=1).swaplevel(axis=1)

            equations = OrderedDict()
            for eq in self.equations:
                equations[eq] = {
                    "dependent": y_[eq],
                    "exog": x_[eq]
                }
                if self.weight is not None:
                    equations[eq]["weights"] = self.weight.loc[y_[eq].index]

            model = estimator(equations=equations, **kwargs)

        elif estimator.__module__ == "linearmodels.panel.model":
            # e.g. PanelOLS
            y = self.y.stack(dropna=False).swaplevel()
            if not isinstance(y.name, str):
                y.name = "y"
            x = self.x.stack(level=0, dropna=False).swaplevel(axis=0)

            if self.weight is not None:
                w = self.weight.reindex(x.index, level=1)
            else:
                w = None

            model = estimator(dependent=y, exog=x, weights=w, **kwargs)

        else:
            raise NotImplementedError

        # add constraints
        if self.constraints is not None:
            model.add_constraints(r=self.constraints["r"],
                                  q=self.constraints["q"])

        return model

    def for_gretl(self, path2data):
        """

        Returns
        -------

        """
        # prepare data for export and import to gretl
        self.y.to_csv(path2data + "y.csv", index_label="date")
        x = self.x.copy()
        x.columns = ["_".join(cc) for cc in self.x.columns]
        x.to_csv(path2data + "x.csv", index_label="date")

        # data import statements
        data_cmd = "\n".join((
            "open " + path2data + "y.csv",
            "append " + path2data + "x.csv"
        ))

        # function to write one equation
        def equation_maker(eq_name):
            aux_res = "equation {} ".format(eq_name) + " ".join(
                [eq_name + "_" + x_name for x_name in self.x[eq_name].columns]
            )
            return aux_res

        # create command for system estimation
        system_cmd = "\n".join(
            ["new_sys <- system", ] +
            ["\t{}".format(equation_maker(eq)) for eq in self.equations] +
            ["end system"]
        )

        # constraints
        # eq_name_to_num = {v: k for k, v in enumerate(self.equations)}
        # x_name_to_num = {k: k for k, v in enumerate(self.equations)}
        # restr_enum = pd.concat({
        #     eq_name_to_num[eq_name]: grp.T.reset_index(drop=True).T
        #     for eq_name, grp in self.constraints["r"].groupby(
        #         axis=1, level="equation")
        # }, axis=1)
        #
        # def restriction_maker(restr_row):
        #     # row_idx is pythonic, i.e. 0 indexes the first item
        #     row_no_zeros = restr_row.where(restr_row != 0).dropna()
        #     aux_res = \
        #         " + ".join([
        #             "{mult} * b[{idx_1},{idx_2}]".format(mult=v,
        #                                                  idx_1=eq_num + 1,
        #                                                  idx_2=coef_num + 1)
        #             for (eq_num, coef_num), v in row_no_zeros.iteritems()
        #         ]) +\
        #         " = {}".format(self.constraints["q"].iloc[restr_row.name])
        #     return aux_res
        #
        # restr_list = list()
        # for _, row in restr_enum.iterrows():
        #     restr_list.append("\t{}".format(restriction_maker(row)))

        # for eq_num, eq_name in enumerate(self.equations):
        #     this_grp = self.constraints["r"][eq_name]\
        #         .where(self.constraints["r"][eq_name] != 0.0)\
        #         .dropna(how="all")
        #     for r_idx, row in this_grp.iterrows():
        #         restr_list.append(restriction_maker(eq_num, row))

        # constraints: alternative
        r = ";\\\n\t".join(
            [", ".join(row.astype(str))
             for row in self.constraints["r"].values]
        )
        q = ";\\\n\t".join([row.astype(str) for row in
                            self.constraints["q"].values])

        rmat_str = "matrix Rmat = {{\\\n\t{}\\\n}}"
        qvec_str = "matrix Qvec = {{\\\n\t{}\\\n}}"
        r_mat_str = "\n".join((rmat_str, qvec_str)).format(r, q)

        restr_list = ["\tR = Rmat\n\tq = Qvec", ]

        if self.constraints is None:
            restriction_cmd = ""
        else:
            restriction_cmd = "\n".join(
                [r_mat_str, ] +
                ["restrict new_sys", ] +
                restr_list +
                ["end restrict"]
            )

        # estimate
        estimate_cmd = "estimate new_sys method=sur --iterate"

        # commands together
        commands = "\n\n".join((
            data_cmd, system_cmd, restriction_cmd, estimate_cmd
        ))

        with open(path2data + "script.inp", mode="w") as f:
            f.write(commands)

    def get_effects(self, grouper):
        """Create a df of effects

        Parameters
        ----------
        grouper : pandas.Series
            indexed with columns of `self.y` (equation names)

        Returns
        -------
        pandas.DataFrame

        """
        fx_df = grouper.to_frame(self.y.index[0]).T\
            .reindex(index=self.y.index, method="ffill")

        fx_ser = fx_df.stack().swaplevel()

        # fx = pd.get_dummies(fx_ser)

        return fx_ser

    def ols(self):
        """

        Returns
        -------

        """
        pass
