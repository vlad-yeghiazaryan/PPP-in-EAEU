import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import add_constant, OLS, QuantReg
from statsmodels.tsa.stattools import adfuller, lagmat, add_trend
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.distributions.empirical_distribution import ECDF
import numba


class QTAR:
    r"""Quantile unit root test

    Based on the Koenker, R., Xiao, Z., 2004. methodology.

    Parameters
    ----------
    endog   -  Nx1 matrix. Preferably pandas series or a numpy array.
    threshold_dummy - Nx1 matrix. A series containing 1s for t>=t* (t* is the threshold date) and 0s otherwise.
    model   -   c  = Constant (Default in Koenker & Xiao (2004)).
                ct = Constant and trend.
    pmax    -  Maximum number of lags for dy (serially correlated terms).
    ic      -  Information Criterion: Default: AIC
                "AIC"    = Akaike
                "BIC"    = Schwarz
                "t-stat" = t-stat significance
    exog    -  Additional exogenous variables to be included in the estimation. Default: None

    Returns
    -------
    τ (quantile)  - the quantile for which the regression was estimated for.
    Lags          - the effective number of lags determined through the information criterion.
    α₀(τ)         - the magnitude of the shock.
    ρ₁(τ)         - the estimated rho parameter for a given quantile. The speed of mean reversion.
    α₁(τ)         - The change in the speed of mean reversion before and after the threshold.
    δ²            - A nuisance parameter used to estimate critical values. See equation (10) at page 778.
    tₙ(τ)          - the quantile unit root statistic (t-ratio for a given quantile) before the threshold.

    """
    def __init__(self, endog, threshold_dummy, model='c', pmax=5, ic='AIC', exog=None):
        # setup
        self.setup(endog, threshold_dummy, model, pmax, ic, exog)
        
    def setup(self, endog, threshold_dummy=None, model=None, pmax=None, ic=None, exog=None):
        # Erase previous results
        self.results = None
        
        # Custom instance changes
        if type(threshold_dummy) != type(None):
            self.threshold_dummy = threshold_dummy
        if model != None:
            self.model = model
        if pmax != None:
            self.pmax = pmax
        if ic != None:
            self.ic = ic
        
        if type(endog) != pd.Series:
            self.endog = pd.Series(endog, name='y')
            self.endog_original = pd.Series(endog, name='y')
        else:
            self.endog_original = endog
            self.endog = endog

        # Creating endog and exog
        
        # creating lags
        y1 = pd.DataFrame(self.endog.shift(1)[1:]).add_suffix('.L1')
        y1_pre = pd.DataFrame(y1.squeeze(axis=1)*(1 - np.array(self.threshold_dummy[1:]))
                             ).add_prefix('pre-'+self.threshold_dummy.name+'_')
        y1_post = pd.DataFrame(y1.squeeze(axis=1)*np.array(self.threshold_dummy[1:])
                              ).add_prefix('post-'+self.threshold_dummy.name+'_')
        dy = self.endog.diff(1)[1:]  # first difference
        X = pd.concat([y1_pre, y1_post], axis=1)

        resultsADF = adfuller(self.endog, maxlag=self.pmax,
                              regression=model, autolag=self.ic)
        ADFt, p, cvADF = resultsADF[0], resultsADF[2], resultsADF[4]

        # Defining series in case of possible legs
        if p > 0:
            dyl = pd.DataFrame(lagmat(dy, maxlag=p), index=dy.index,
                               columns=['Diff.L' + str(i) for i in range(1, 1 + p)])[p:]
            X = pd.concat([y1[p:], y1_post[p:], dyl], axis=1)
        # If the ic decides not to include lags
        else:
            X = X[p:]

        # Removing tails
        self.lags = p
        self.extraExog = exog[p + 1:] if type(exog) != type(None) else None
        self.L1 = y1[p:]
        self.L1_pre = y1_pre[p:]
        self.L1_post = y1_post[p:]
        self.diff = dy[p:]
        self.diffLags = dyl if self.lags > 0 else None
        self.endog = self.endog[p + 1:]
        if self.model == 'c':
            self.exog = add_constant(X)
        elif self.model == 'ct':
            self.exog = add_trend(add_constant(X), 'ct')
        else:
            raise ValueError(
                'Model type is not recognized: ' + str(self.model))

        # Adding additional exog if present
        self.exog = pd.concat([self.exog, self.extraExog], axis=1)
        return self

    # Fitting data for different quantiles
    def fitForQuantiles(self, quantiles):
        r"""quantiles -  A list like array of the quantiles to perform the fitting.
            example: np.arange(0.1, 1, 0.1)
        """
        results = []
        for tau in quantiles:
            res = self.fit(tau)
            results.append(res.results)
        df = pd.DataFrame(results)

        # Adding the QKS statistic
        df['QKS'] = max(map(abs, df['tₙ(τ)']))
        df['name'] = self.endog.name
        df.set_index('quantile', inplace=True)
        return df

    def fit(self, tau):
        r"""tau -  quantile (ranges: 0.1,...,1)
        """
        # Running the quantile regression
        n = len(self.endog)
        qOut1 = QuantReg(self.endog, self.exog).fit(q=tau)
        
        # calculating alpha and rho
        alpha0_tau = qOut1.params[0]
        rho_tau = qOut1.params[1]
        alpha1_tau = qOut1.params[2]

        # calculating delta2 using covariance
        ind = qOut1.resid < 0
        phi = tau - ind
        w = self.diff
        cov = np.cov(phi, w)[0, 1]
        delta2 = (cov / (np.std(w, ddof=1) * np.sqrt(tau * (1 - tau)))) ** 2

        # Calculating quantile bandwidth
        h = self.bandwidth(tau, n, True)
        if tau <= 0.5 and h > tau:
            h = self.bandwidth(tau, n, False)
            if h > tau:
                h = tau / 1.5

        if tau > 0.5 and h > 1 - tau:
            h = self.bandwidth(tau, n, False)
            if h > (1 - tau):
                h = (1 - tau) / 1.5

        if self.lags == 0:
            # Defining some inputs
            if self.model == 'c':
                X = add_constant(pd.concat([self.L1, self.L1_post], axis=1))
            elif self.model == 'ct':
                X = add_trend(add_constant(pd.concat([self.L1, self.L1_post], axis=1)), 'ct')
            else:
                raise ValueError(
                    'Model type is not recognized: ' + str(self.model))
            X = pd.concat([X, self.extraExog], axis=1)

        # The common case
        elif self.lags > 0:
            X = self.exog
        
        # Running the other 2 QuantRegs
        qOut2 = QuantReg(self.endog, X).fit(q=tau + h)
        qOut3 = QuantReg(self.endog, X).fit(q=tau - h)

        # Extracting betas
        rq1 = qOut2.params
        rq2 = qOut3.params

        # Setting inputs for the unit root test
        z = X
        mz = z.mean(axis=0)

        q1 = np.matmul(mz, rq1)
        q2 = np.matmul(mz, rq2)
        fz = 2 * h / (q1 - q2)
        if fz < 0:
            fz = 0.01

        xx = pd.Series(np.ones(len(X)), name='const')
        if self.lags > 0:
            if self.model == 'c':
                xx = add_constant(self.diffLags)
            elif self.model == 'ct':
                xx = add_trend(add_constant(self.diffLags), 'ct')
            else:
                raise ValueError(
                    'Model type is not recognized: ' + str(self.model))

        # Adding extra exog
        xx = pd.concat([xx, self.extraExog], axis=1)

        # aligning matrices
        xx = np.array(xx)
        y1 = np.array(self.L1)
        y1_post = np.array(self.L1_post)

        # Constructing a NxN matrix
        PX = np.eye(len(xx)) - \
            xx.dot(np.linalg.inv(np.dot(xx.T, xx))).dot(xx.T)
        fzCrt = fz / np.sqrt(tau * (1 - tau))
        
        # We will preform a single unit root test for the series before the threshold
        eqPX = np.sqrt((y1.T.dot(PX).dot(y1))[0][0])

        # QADF statistic
        QURadf = fzCrt * eqPX * (rho_tau - 1)

        # Exposing variables
        self.tau = tau
        self.alpha0_tau = alpha0_tau
        self.rho_tau = rho_tau
        self.alpha1_tau = alpha1_tau
        self.delta2 = delta2
        self.QURadf = QURadf
        self.regression = qOut1
        
        # Exposing final results
        self.results = {
            'quantile': round(self.tau, 2),
            'Lags': self.lags,
            'α\u2080(τ)': round(self.alpha0_tau, 4),
            'ρ\u2081(τ)': round(self.rho_tau, 4),
            'α\u2081(τ)': round(self.alpha1_tau, 5),
            'δ\u00B2': round(self.delta2, 3),
            't\u2099(τ)': round(self.QURadf, 4)
        }
        return self

    @staticmethod
    @numba.jit(forceobj=True, parallel=True)
    def bandwidth(tau, n, is_hs, alpha=0.05):
        x0 = norm.ppf(tau)  # inverse of cdf
        f0 = norm.pdf(x0)  # Probability density function

        if is_hs:
            a = n**(-1 / 3)
            b = norm.ppf(1 - alpha / 2)**(2 / 3)
            c = ((1.5 * f0**2) / (2 * x0**2 + 1))**(1 / 3)

            h = a * b * c
        else:
            h = n**(-0.2) * ((4.5 * f0**4) / (2 * x0**2 + 1)**2)**0.2

        return h

    def __repr__(self):
        try:
            self.regression
        except AttributeError:
            return str(AutoReg)
        else:
            return object.__repr__(self.regression)

    def summary(self):
        if self.results != None:
            rmv_chars = {'}': '', '{': '', "'": ''}
            rmv_out = str(self.results).translate(str.maketrans(rmv_chars))
            out = rmv_out.replace(',', '\n').replace('\n ', '\n')
            print(out)
        else:
            return object.__repr__(self)
        

# Code for reporting values


def oneTailUpper(value, ecdfValue, significanceLevels, rounding=2):
    pValue = round(1 - ecdfValue, 3)
    starValue = str(round(value, rounding)) + f'({pValue})' + ''.join(
        ['*' for t in significanceLevels if pValue <= t])
    return starValue

def oneTailLower(value, ecdfValue, significanceLevels, rounding=2):
    pValue = round(ecdfValue, 3)
    starValue = str(round(value, rounding)) + f'({pValue})' + ''.join(
        ['*' for t in significanceLevels if pValue <= t])
    return starValue

def twoTail(value, ecdfValue, significanceLevels, rounding=2):
    pValue = round(min(ecdfValue, 1 - ecdfValue), 3) 
    starValue = str(round(value, rounding)) + f'({pValue})' + ''.join(
        ['*' for t in significanceLevels if pValue <= t/2])
    return starValue

def addStars(row, test, significanceLevels, rounding=2):
    boots = row[1]
    value = row[0]
    ecdfFunc = ECDF(boots, side='left')
    ecdfValue = ecdfFunc(value)
    starValue = test(value, ecdfValue, significanceLevels, rounding)
    return starValue

def QTAR_CustomRport(CountryQADF, results, significanceLevels, dropColumns=None):
    values =  ['α₀(τ)', 'ρ₁(τ)', 'α₁(τ)', 'tₙ(τ)']
    for value in values:
        test = twoTail if value == 'α₀(τ)' else oneTailLower
        rounding = 2 if value != 'α₁(τ)' else 4
        CountryQADF[value] = pd.concat([CountryQADF, results.groupby('quantile')[value].apply(lambda x:np.array(x))], axis=1)[value].apply(addStars, axis=1, args=[test, significanceLevels, rounding])
    CountryQADF['QKS'] = addStars([CountryQADF['QKS'].mean(), results.groupby('name')['QKS'].mean()], oneTailUpper, significanceLevels)
    CountryQADF.loc[0.2:, 'QKS'] = ''
    CountryQADF.drop(columns=dropColumns, inplace=True) if dropColumns != None else None

    # final cleanup
    report = CountryQADF.T.drop('name')
    report['name']  = CountryQADF['name'][0.1]
    report.set_index(['name',report.index], inplace=True)
    
    return report
