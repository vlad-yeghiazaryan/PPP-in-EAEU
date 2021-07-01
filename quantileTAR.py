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
    ρ₁(OLS)       - the estimated rho parameter for the OLS estimation.
    δ²            - A nuisance parameter used to estimate critical values. See equation (10) at page 778.
    Half-lives    - ln(0.5)/ln(abs(ρ₁(τ))).
    tₙ(τ)          - the quantile unit root statistic (t-ratio for a given quantile).
    cv            - 1%, 5%, 10% critical values for the estimated δ².

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
            X = pd.concat([y1_pre[p:], y1_post[p:], dyl], axis=1)
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
        df['pre-QKS'] = max(map(abs, df['pre-tₙ(τ)']))
        df['post-QKS'] = max(map(abs, df['post-tₙ(τ)']))
        df['name'] = self.endog.name
        df.set_index('quantile', inplace=True)
        return df

    def fit(self, tau):
        r"""tau -  quantile (ranges: 0.1,...,1)
        """
        # Running the quantile regression
        n = len(self.endog)
        qOut1 = QuantReg(self.endog, self.exog).fit(q=tau)
        olsFit = OLS(self.endog, self.exog).fit()
        
        # calculating alpha and rho
        alpha_tau = qOut1.params[0]
        pre_rho_tau = qOut1.params[1]
        post_rho_tau = qOut1.params[2]
        pre_rho_ols = olsFit.params[1]
        post_rho_ols = olsFit.params[2]

        # Estimating Half-lifes pre_rho_tau
        hl = np.log(0.5) / np.log(np.abs(post_rho_tau))
        hl = '∞' if hl < 0 else round(hl, 3)

        # calculating delta2 using covariance
        ind = qOut1.resid < 0
        phi = tau - ind
        w = self.diff
        cov = np.cov(phi, w)[0, 1]
        delta2 = (cov / (np.std(w, ddof=1) * np.sqrt(tau * (1 - tau)))) ** 2

        # calculating critical values associate with our delta2
        crv = self.crit_QRadf(delta2, self.model)

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

        # Defining some inputs
        if self.model == 'c':
            X = add_constant(pd.concat([self.L1_pre, self.L1_post], axis=1))
        elif self.model == 'ct':
            X = add_trend(add_constant(pd.concat([self.L1_pre, self.L1_post], axis=1)), 'ct')
        else:
            raise ValueError(
                'Model type is not recognized: ' + str(self.model))

        X = pd.concat([X, self.extraExog], axis=1)

        # The common case
        if self.lags > 0:
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
        y1_pre = np.array(self.L1_pre)
        y1_post = np.array(self.L1_post)

        # Constructing a NxN matrix
        PX = np.eye(len(xx)) - \
            xx.dot(np.linalg.inv(np.dot(xx.T, xx))).dot(xx.T)
        fzCrt = fz / np.sqrt(tau * (1 - tau))
        
        # We will preform two unit root test for the series before 
        # and after the threshold
        eqPX_pre = np.sqrt((y1_pre.T.dot(PX).dot(y1_pre))[0][0])
        eqPX_post = np.sqrt((y1_post.T.dot(PX).dot(y1_post))[0][0])

        # QADF statistics for y1_pre and y1_post
        QURtadf_pre = fzCrt * eqPX_pre * (pre_rho_tau - 1)
        QURtadf_post = fzCrt * eqPX_post * (post_rho_tau - 1)
        cv = self.crit_QRadf(delta2, self.model)

        # Exposing variables
        self.tau = tau
        self.alpha_tau = alpha_tau
        self.pre_rho_tau = pre_rho_tau
        self.post_rho_tau = post_rho_tau
        self.pre_rho_ols = pre_rho_ols
        self.post_rho_ols = post_rho_ols
        self.delta2 = delta2
        self.QURtadf_pre = QURtadf_pre
        self.QURtadf_post = QURtadf_post
        self.hl = hl
        self.regression = qOut1
        self.cvs = {
            'CV10%': cv['10%'],
            'CV5%': cv['5%'],
            'CV1%': cv['1%']
        }
        self.results = {
            'quantile': round(self.tau, 2),
            'Lags': self.lags,
            'α\u2080(τ)': round(self.alpha_tau, 4),
            'pre-ρ\u2081(τ)': round(self.pre_rho_tau, 4),
            'post-ρ\u2082(τ)': round(self.post_rho_tau, 4),
            'pre-ρ\u2081(OLS)': round(self.pre_rho_ols, 4),
            'post-ρ\u2082(OLS)': round(self.post_rho_ols, 4),
            'ρ\u2081(τ)-ρ\u2082(τ)':round(self.pre_rho_tau-self.post_rho_tau, 5),
            'δ\u00B2': round(self.delta2, 3),
            'Half-lives': self.hl,
            'pre-t\u2099(τ)': round(self.QURtadf_pre, 4),
            'post-t\u2099(τ)': round(self.QURtadf_post, 4),
            'CV10%': round(self.cvs['CV10%'], 4),
            'CV5%': round(self.cvs['CV5%'], 4),
            'CV1%': round(self.cvs['CV1%'], 4)
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

    @staticmethod
    @numba.jit(forceobj=True, parallel=True)
    def crit_QRadf(r2, model):
        ncCV = [[-2.4611512, -1.783209, -1.4189957],
                [-2.494341, -1.8184897, -1.4589747],
                [-2.5152783, -1.8516957, -1.5071775],
                [-2.5509773, -1.895772, -1.5323511],
                [-2.5520784, -1.8949965, -1.541883],
                [-2.5490848, -1.8981677, -1.5625462],
                [-2.5547456, -1.934318, -1.5889045],
                [-2.5761273, -1.9387996, -1.602021],
                [-2.5511921, -1.9328373, -1.612821],
                [-2.5658, -1.9393, -1.6156]]

        cCV = [[-2.7844267, -2.115829, -1.7525193],
               [-2.9138762, -2.2790427, -1.9172046],
               [-3.0628184, -2.3994711, -2.057307],
               [-3.1376157, -2.5070473, -2.168052],
               [-3.191466, -2.5841611, -2.2520173],
               [-3.2437157, -2.639956, -2.316327],
               [-3.2951006, -2.7180169, -2.408564],
               [-3.3627161, -2.7536756, -2.4577709],
               [-3.3896556, -2.8074982, -2.5037759],
               [-3.4336, -2.8621, -2.5671]]

        ctCV = [[-2.9657928, -2.3081543, -1.9519926],
                [-3.1929596, -2.5482619, -2.1991651],
                [-3.3727717, -2.7283918, -2.3806008],
                [-3.4904849, -2.8669056, -2.5315918],
                [-3.6003166, -2.9853079, -2.6672416],
                [-3.6819803, -3.095476, -2.7815263],
                [-3.7551759, -3.178355, -2.8728146],
                [-3.8348596, -3.2674954, -2.973555],
                [-3.8800989, -3.3316415, -3.0364171],
                [-3.9638, -3.4126, -3.1279]]

        # Selecting the critical values set based on model type
        cvs = {'nc': ncCV, 'c': cCV, 'ct': ctCV}
        cv = cvs[model]

        delta2 = pd.Series(np.arange(0.1, 1.1, 0.1), name='delta2')
        crt = pd.DataFrame(cv, index=delta2, columns=['1%', '5%', '10%'])

        if r2 < 0.1:
            ct = crt.iloc[0, :]
        else:
            r210 = r2 * 10
            if (r210) >= 10:
                ct = crt.iloc[9, :]
            else:
                #  Main logic goes here
                r2a = int(np.floor(r210))
                r2b = int(np.ceil(r210))
                wa = r2b - r210
                ct = wa * crt.iloc[(r2a - 1), :] + (1 - wa) * \
                    crt.iloc[(r2b - 1), :]
        return ct

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
    values =  ['α₀(τ)', 'pre-ρ₁(τ)', 'post-ρ₂(τ)', 'pre-ρ₁(OLS)', 'post-ρ₂(OLS)', 
               'pre-tₙ(τ)', 'post-tₙ(τ)', 'ρ₁(τ)-ρ₂(τ)']
    for value in values:
        test = twoTail if value == 'α₀(τ)' else oneTailLower
        rounding = 2 if value != 'ρ₁(τ)-ρ₂(τ)' else 4
        CountryQADF[value] = pd.concat([CountryQADF, results.groupby('quantile')[value].apply(lambda x:np.array(x))], axis=1)[value].apply(addStars, axis=1, args=[test, significanceLevels, rounding])
    CountryQADF['pre-QKS'] = addStars([CountryQADF['pre-QKS'].mean(), results.groupby('name')['pre-QKS'].mean()], oneTailUpper, significanceLevels)
    CountryQADF.loc[0.2:, 'pre-QKS'] = ''
    CountryQADF['post-QKS'] = addStars([CountryQADF['post-QKS'].mean(), results.groupby('name')['post-QKS'].mean()], oneTailUpper, significanceLevels)
    CountryQADF.loc[0.2:, 'post-QKS'] = ''
    CountryQADF['Half-lives'] = CountryQADF['Half-lives'].apply(lambda x: round(x/12) if not isinstance(x, str) else x)
    CountryQADF.rename(columns={'Half-lives':'Half-lives (years)'}, inplace=True)
    CountryQADF.drop(columns=dropColumns, inplace=True) if dropColumns != None else None

    # final cleanup
    report = CountryQADF.T.drop('name')
    report['name']  = CountryQADF['name'][0.1]
    report.set_index(['name',report.index], inplace=True)
    
    return report
