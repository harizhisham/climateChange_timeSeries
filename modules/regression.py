'''
It does stuff
'''
def regional_auto_ols(data=None, view=None, show=False, diagnostics=False, lags=None, fourier=False, monthly=False):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import statsmodels.stats.api as sms
    from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox
    import testing

    assert data is not None
    assert view is not None

    perf_in = {}
    perf_out = {}
    acorr_stat = {}
    r2 = {}
    pval = {}

    if view == 'regional':
        for u in data['region'].unique():
            region = str(u)

            '''
            References a global variable. Grouping is done here.
            '''
            filtered = data[data['region']==region].groupby(['timestamp']).max().drop(columns=['region','year','month'])

            # Enumerates the index in the time series and calculates the FFT based
            #   off the index number. Returns a list
            if fourier:
                X_f = []
                for n, a in enumerate(filtered.index):
                    x = [1, n, np.sin(2 * n * np.pi / 12), np.cos(2 * n * np.pi / 12)] # Dropped n
                    X_f += [x]
                X_f = pd.DataFrame(X_f, columns=['const','Idx','sin','cos'], index=filtered.index)
            elif monthly:
                X_f = []
                for n,(d,p) in enumerate(filtered['min_temp'].items()):
                    m = [1, n] + [0] * 12
                    m[d.month+1] = 1
                    X_f += [m]
                X_f = pd.DataFrame(X_f,
                                    columns=['1',
                                            'Idx',
                                            'j','f','m','a','m','j','j','a','s','o','n','d'],
                                    index=filtered.index)
            else:
                X_f = []
                for n, a in enumerate(filtered.index):
                    X_f += [n]
                 # Convert the list created into a pandas DF, using the initial time series index
                X_f = pd.DataFrame(X_f, columns=['Idx'], index=filtered.index)


            # Concatenate the FFT with the initial endog. time series
            filtered_f = pd.concat([filtered, X_f],axis=1)
            # Delete the FFT to save memory
            del X_f

            last = 0
            if lags is not None:
                assert type(lags) == list
                for l in lags:
                    filtered_f['L_'+str(l)] = pd.DataFrame(filtered_f['min_temp']).shift(l)

                last=lags[-1]

            y = list(filtered_f['min_temp'])
            X = filtered_f.iloc[:,3:]

            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                shuffle=False)

            reg = sm.OLS(y_train[last:],
                         X_train[last:])
            results = reg.fit()

            y_hat = results.predict(X_train[last:])
            y_pred = results.predict(X_test)

            if show:
                print('---------------\n%s\n---------------' % ('Time Series Regression Results\n'+region))
                print(results.summary())

                lb, p_values=ljungbox(results.resid, lags=36)

                if diagnostics:
                    testing.ts_plot(results.resid, lags=36)
                    idx = list(range(len(y_train[last:])))
                    print('---------------\n%s\n---------------' % ('Monthly Min Temp\n'+region))
                    testing.eval_plot(idx, y_train[last:], y_hat)
                    testing.measure_error(y_train[last:], y_hat, label=region+' OLS OOS', show=True)
                    testing.measure_error(y_test, y_pred, label=region+' OLS OOS',show=True)
                    print('Durbin Watson: %s' % str(sms.durbin_watson(results.resid)))
                print('\n')

            perf_in[region] = testing.measure_error(y_train[last:], y_hat, label=region+' OLS OOS', show=False)
            perf_out[region] = testing.measure_error(y_test, y_pred, label=region+' OLS OOS',show=False)
            acorr_stat[region] = sms.durbin_watson(results.resid)
            r2[region] = results.rsquared
            pval[region] = results.pvalues['Idx']

    elif view == 'coord':
        for u in data['coord'].unique():
            coord = str(u)

            '''
            References a global variable. Grouping is done here.
            '''
            filtered = data[data['coord']==coord].groupby(['timestamp']).mean().drop(columns=['year','month'])

            # Enumerates the index in the time series and calculates the FFT based
            #   off the index number. Returns a list
            if fourier:
                X_f = []
                for n, a in enumerate(filtered.index):
                    x = [1, n, np.sin(2 * n * np.pi / 12), np.cos(2 * n * np.pi / 12)] # Dropped n
                    X_f += [x]
                X_f = pd.DataFrame(X_f, columns=['const','Idx','sin','cos'], index=filtered.index)
            elif monthly:
                X_f = []
                for n,(d,p) in enumerate(filtered['min_temp'].items()):
                    m = [1, n] + [0] * 12
                    m[d.month+1] = 1
                    X_f += [m]
                X_f = pd.DataFrame(X_f,
                                    columns=['1',
                                            'Idx',
                                            'j','f','m','a','m','j','j','a','s','o','n','d'],
                                    index=filtered.index)
            else:
                X_f = []
                for n, a in enumerate(filtered.index):
                    X_f += [n]
                 # Convert the list created into a pandas DF, using the initial time series index
                X_f = pd.DataFrame(X_f, columns=['Idx'], index=filtered.index)


            # Concatenate the FFT with the initial endog. time series
            filtered_f = pd.concat([filtered, X_f],axis=1)
            # Delete the FFT to save memory
            del X_f

            last = 0
            if lags is not None:
                assert type(lags) == list
                for l in lags:
                    filtered_f['L_'+str(l)] = pd.DataFrame(filtered_f['min_temp']).shift(l)

                last=lags[-1]

            y = list(filtered_f['min_temp'])
            X = filtered_f.iloc[:,3:]

            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                shuffle=False)

            reg = sm.OLS(y_train[last:],
                         X_train[last:])
            results = reg.fit()

            dw_stat = sms.durbin_watson(results.resid)

            y_hat = results.predict(X_train[last:])
            y_pred = results.predict(X_test)

            if show:
                print('---------------\n%s\n---------------' % ('Time Series Regression Results\n'+coord))
                print(results.summary())

                lb, p_values=ljungbox(results.resid, lags=36)

                if diagnostics:
                    testing.ts_plot(results.resid, lags=36)
                    idx = list(range(len(y_train[last:])))
                    print('---------------\n%s\n---------------' % ('Monthly Min Temp\n'+coord))
                    testing.eval_plot(idx, y_train[last:], y_hat)
                    testing.measure_error(y_train[last:], y_hat, label=coord+' OLS OOS', show=True)
                    testing.measure_error(y_test, y_pred, label=coord+' OLS OOS',show=True)
                    print('Durbin Watson: %s' % str(sms.durbin_watson(results.resid)))
                print('\n')

            perf_in[coord] = testing.measure_error(y_train[last:], y_hat, label=coord+' OLS OOS', show=False)
            perf_out[coord] = testing.measure_error(y_test, y_pred, label=coord+' OLS OOS',show=False)
            acorr_stat[coord] = sms.durbin_watson(results.resid)
            r2[coord] = results.rsquared
            pval[coord] = results.pvalues['Idx']

    return perf_in, perf_out, acorr_stat, pval
