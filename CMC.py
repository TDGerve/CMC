from scipy.optimize import Bounds, basinhopping, root_scalar
import numpy as np
import pandas as pd

from typing import Union
from numpy.typing import ArrayLike

"""
Depleted morb mantle - melt bulk partition coefficients.
CO2 from Rostenthal et al. (2015), H2O from Hirschmann et al. (2009) and all other from Workman and Hart (2005)
"""
Kds = pd.Series(
    {
        "Rb": 1e-5,
        "Ba": 1.2e-4,
        "Th": 1e-3,
        "U": 1.1e-3,
        "Nb": 3.4e-3,
        "Ta": 3.4e-3,
        "La": 1e-2,
        "Ce": 2.2e-2,
        "Pb": 1.4e-2,
        "Pr": 2.7e-2,
        "Nd": 3.1e-2,
        "Sr": 2.5e-2,
        "Zr": 2.5e-2,
        "Hf": 3.5e-2,
        "Sm": 4.5e-2,
        "Eu": 5e-2,
        "Ti": 5.8e-2,
        "Gd": 5.6e-2,
        "Tb": 6.8e-2,
        "Dy": 7.9e-2,
        "Ho": 8.4e-2,
        "Y": 8.8e-2,
        "Er": 9.7e-2,
        "Yb": 0.115,
        "Lu": 1.2e-1,
        "CO2": 5.5e-4,
        "H2O": 6.5e-3,
    }
)


def rayleigh_fractionation(
    init_c: Union[float, ArrayLike], X: Union[float, ArrayLike], element: str
) -> Union[float, ArrayLike]:
    """
    Parameters
    ----------
    init_c : float, numpy.array or pandas.Series
        initial concentrations
    X : float, numpy.array or pandas.Series
        crystallisation extent as mass fraction
    element : string
        element name

    Returns
    -------
    float, numpy.array or pandas.Series
        differentiated element concentrations
    """

    Kd = Kds[element]

    return init_c * ((1 - X) ** (Kd - 1))


class CMC:
    """
    Model two component concurrent mixing and crystallisation according to  and Shorttle et al. (2016).
    Equation numbers refer to Shorttle et al.
    """

    def __init__(self, source1: pd.Series, source2: pd.Series, **kwargs):

        self.source1 = source1
        self.source2 = source2
        # Enriched component fraction
        self.w_enriched = kwargs.get("w_enriched", 0.4)
        # Maximum fractionation
        self.F_max = kwargs.get("F_max", 0.7)
        # Miminum glass MgO
        self.MgO_min = kwargs.get("MgO_min", 0.7)
        # Model parameters
        # Mixing extent
        self.N_min, self.N_max = kwargs.get("N_limits", [1, 108])
        # Mixing rate
        self.k_mixing = kwargs.get("k", 1.2)
        self.n_samples = kwargs.get("n_samples", 1e3)

    def calculate(self, **kwargs):

        parents_mixed = self._mix(**kwargs)
        return self._crystallise(parents_mixed, **kwargs)

    def fit_w_enriched(self, data, elements, **kwargs):

        bounds = Bounds((1e-2), (1 - 1e-2))
        x0 = kwargs.pop("x0", self.w_enriched)
        niter = kwargs.pop("n_iter", 500)
        niter_success = kwargs.pop("niter_success", 50)

        n_samples = kwargs.pop("n_samples", 3e3)
        kwargs["n_samples"] = n_samples

        data = data[elements].copy()

        solution = basinhopping(
            self._root_w_cmc,
            x0,
            minimizer_kwargs={"args": (elements, data, kwargs), "bounds": bounds},
            T=0.1,
            stepsize=0.2,
            niter=niter,
            niter_success=niter_success,
        )
        self.w_enriched = float(solution.x)
        return solution

    def _generate_parameters(self, **kwargs):

        w_enriched = kwargs.get("w_enriched", self.w_enriched)
        mixing_rate = kwargs.get("mixing_rate", self.k_mixing)
        n_samples = kwargs.get("n_samples", self.n_samples)

        x_max_enriched = self.F_max
        x_max_depleted = x_max_enriched

        w_fractions = np.array([w_enriched, (1 - w_enriched)])

        x_sources = np.array([x_max_enriched, x_max_depleted])
        # Random number generator
        rng = np.random.default_rng(12345)
        # Degree of differentiation
        U_differentiation = rng.uniform(0, 1, int(n_samples))
        # Degree of mixing, equation (5)
        N_mixing = (
            self.N_min * (self.N_max / self.N_min) ** U_differentiation**mixing_rate
        )
        # generate mixing weights from Dirichlet distribution, equation (11)
        alphas = [(N - 1) * w_fractions for N in N_mixing]
        r_mixing = np.array([rng.dirichlet(a) for a in alphas])
        # Maximum degree of fractionation, equation (4)
        X_max = np.array([(r * x_sources).sum() for r in r_mixing])
        # equation (3)
        X = X_max * U_differentiation

        self.parameters = {
            "N_mixing": N_mixing,
            "r_mixing": r_mixing,
            "X_max": X_max,
            "X": X,
        }

    def _mix(self, MgO=False, generate_parameters=True, **kwargs):

        source1 = kwargs.get("source1", self.source1)
        source2 = kwargs.get("source2", self.source2)
        n_samples = kwargs.get("n_samples", self.n_samples)

        self._generate_parameters(**kwargs)
        element_names = Kds.index.intersection(source1.index)
        # Mix parent melts
        parents_mixed = pd.DataFrame(
            dtype=float, columns=element_names, index=np.arange(0, int(n_samples), 1)
        )
        if MgO:
            parents_mixed["MgO"] = np.nan

        for element in parents_mixed.columns:
            # equation (6)
            parents_mixed[element] = (
                source1[element] * self.parameters["r_mixing"][:, 0]
                + source2[element] * self.parameters["r_mixing"][:, 1]
            )

        return parents_mixed

    def _crystallise(self, starting_compositions, MgO=False, **kwargs):

        element_names = Kds.index.intersection(starting_compositions.columns)

        parents_mixed = starting_compositions.copy()

        fractionated = parents_mixed.copy()
        for element in element_names:
            # equation (13)
            fractionated[element] = rayleigh_fractionation(
                parents_mixed[element], self.parameters["X"], element
            )

            # parents_mixed[element].mul(
            #     (1 - self.parameters["X"]) ** (Kds[element] - 1)
            # )
        if MgO:
            # equation (14)
            fractionated["MgO"] = (
                fractionated["MgO"]
                - (fractionated["MgO"] - self.MgO_min)
                / self.parameters["X_max"]
                * self.parameters["X"]
            )

        # Fractionation extent
        fractionated["X"] = self.parameters["X"]
        # Mixing degree
        fractionated["mixed"] = 1 - 1 / self.parameters["N_mixing"]

        return fractionated

    def _compare_2d_histograms(self, data, model, bins=16, hists_out=False):

        xmin, xmax = model.iloc[:, 0].min(), model.iloc[:, 0].max()
        xmin, xmax = np.floor(xmin / 10) * 10, np.ceil(xmax / 10) * 10

        ymin, ymax = model.iloc[:, 1].min(), model.iloc[:, 1].max()
        ymin, ymax = np.floor(ymin / 10) * 10, np.ceil(ymax / 10) * 10

        d_range = [[xmin, xmax], [ymin, ymax]]

        hist_data = data.loc[
            data.iloc[:, 0].between(xmin, xmax) & data.iloc[:, 1].between(ymin, ymax)
        ]
        model_data = model.loc[
            model.iloc[:, 0].between(xmin, xmax) & model.iloc[:, 1].between(ymin, ymax)
        ]

        density_model = np.histogram2d(
            model_data.iloc[:, 0], model_data.iloc[:, 1], bins=bins, range=d_range
        )
        density_observed = np.histogram2d(
            hist_data.iloc[:, 0], hist_data.iloc[:, 1], bins=bins, range=d_range
        )

        normalise = density_model[0].sum() / density_observed[0].sum()

        difference = abs(density_model[0] / normalise - density_observed[0]).mean()

        if hists_out:
            return density_model, density_observed
        else:
            return difference

    def _root_w_cmc(self, w_enriched, elements, data, kwargs):

        w_enriched = float(w_enriched)

        bins = kwargs.pop("bins", 16)
        averages = kwargs.pop("averages", 1)

        source1 = self.source1[elements]
        source2 = self.source2[elements]

        results = []

        for _ in range(averages):
            model = self.calculate(
                w_enriched=w_enriched, source1=source1, source2=source2, **kwargs
            )

            difference = self._compare_2d_histograms(
                data[elements], model[elements], bins=bins
            )

            results.append(difference)

        return np.mean(difference)


def calculate_wX(
    composition: pd.DataFrame,
    enriched_component: pd.Series,
    depleted_component: pd.Series,
) -> pd.DataFrame:
    """
    Calculate crystallisation extent and parental melt composition from sample trace element composition.
    Based on a two component concurrent mixing and crystallisation model (Rudge et al., 2013).

    Parameters
    ----------
    composition :   pd.DataFrame
        Melt compositions for two trace elements

    enriched_component  :   pd.Series
        Trace element composition of an enriched primary melt

    depleted_component  :   pd.Series
        Trace element composition of a depleted primary melt

    Returns
    _______
    pd.DataFrame
        Crystallised  mass fraction (F) and mass fraction of the enriched component in the parental melt (w_enriched) for each sample in 'composition'

    """

    results = pd.DataFrame(
        dtype=float, columns=["w_enriched", "X"], index=composition.index
    )

    for i, row in composition.iterrows():
        try:
            results.loc[i, "w_enriched"] = root_scalar(
                _root_w,
                args=(row, enriched_component, depleted_component),
                x0=0.1,
                x1=0.4,
            ).root
        except ValueError:
            results.loc[i, "w_enriched"] = np.nan

    element, _ = composition.columns
    element_0 = (
        results["w_enriched"] * enriched_component[element]
        + (1 - results["w_enriched"]) * depleted_component[element]
    )
    # e2_0 = results["w"] * components.loc[index[0], e2] + (1 - results["w"]) * components.loc[index[1], e2]

    Kd = Kds[element]

    results["X"] = 1 - (composition[element] / element_0) ** (1 / (Kd - 1))

    return results


def _root_w(
    w_enriched,
    fractionated_concentrations: pd.Series,
    enriched_component: pd.Series,
    depleted_component: pd.Series,
):

    w_depleted = 1 - w_enriched

    # Kd2 has to be smaller than Kd1 otherwise e1_0 ** D will potentiall result in (negative number) ** (< 1)
    Kd_selection = pd.Series(
        {e: c for e, c in Kds.items() if e in fractionated_concentrations.index}
    ).sort_values(ascending=False)

    e1, e2 = Kd_selection.index
    e1_F, e2_F = fractionated_concentrations[[e1, e2]]
    kd1 = Kds[e1]
    kd2 = Kds[e2]

    e1_e, e2_e = enriched_component[[e1, e2]]
    e1_d, e2_d = depleted_component[[e1, e2]]

    D = (kd2 - 1) / (kd1 - 1)

    e1_0 = (
        w_depleted * e1_d + w_enriched * e1_e + 1e-3
    )  # prevent e0 from becoming exactly 0.
    e2_0 = w_depleted * e2_d + w_enriched * e2_e + 1e-3

    F_ratio = _calculate_F_ratio([e1_F, e2_F], [e1, e2])

    return ((e1_0**D) / e2_0) - F_ratio


def _calculate_F_ratio(concentrations, elements):

    e1, e2 = elements
    e1_F, e2_F = concentrations
    kd1, kd2 = Kds[e1], Kds[e2]

    D = (kd2 - 1) / (kd1 - 1)

    return (e1_F**D) / e2_F
