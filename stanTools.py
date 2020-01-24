import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def plot_corr_reps(df_fit_1, df_fit_2):
    '''
    expects 2 stan fit dataframes.
    plots the correlation of the mean of differential growth rates (diff_r) among posterior samples
    '''
    diff_cols = [c for c in df_fit_1.columns if c.startswith('diff_r')]
    mean_fits_1 = np.mean(df_fit_1[diff_cols])
    mean_fits_2 = np.mean(df_fit_2[diff_cols])
    print('pearson corr:', pearsonr(mean_fits_1, mean_fits_2))
    plt.figure()
    plt.scatter(mean_fits_1, mean_fits_2)
    plt.xlabel('differential growth rate 1')
    plt.ylabel('differential growth rate 2')

    plt.show()
    
def plot_unscaled_reps(df_rm, df_rm2):
    plt.figure()
    x= df_rm.mean_fit.values
    y = df_rm2.mean_fit.values
    plt.scatter(x,y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.plot([0, 1.4], [0, 1.4], ls="--", c=".3")
    plt.title('unscaled replicates')
    plt.show()

    print(np.poly1d(np.polyfit(x, y, 1)))

def get_range(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)

def plot_all_ppcs(df_fit, var_plot, obs_data):
    '''
    Plot posterior predictive distribution for mean, stdev, max and range of count data
    
    Parameters
    ----------
    df_fit : stan df_fit object
    var_plot : str, 
        such as 'c_pre'
    obs_data : dict
        dictionary fed into stan model for in sm.sampling() containing the observed data var_plot as key
    '''

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2) #, height_ratios=[3, 1], width_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    plot_ppc(ax0, df_fit, var_plot, obs_data, np.mean,'mean count of wt variants before selection', 'mean' )
    ax1 = plt.subplot(gs[1])

    plot_ppc(ax1, df_fit, var_plot, obs_data, np.std,'stdev count of wt variants before selection', 'stdev' )

    ax2 = plt.subplot(gs[2])
    plot_ppc(ax2, df_fit, var_plot, obs_data, np.max,'max count of wt variants before selection', 'max' )
    
    ax3= plt.subplot(gs[3])
    plot_ppc(ax3, df_fit, var_plot, obs_data, get_range,'range count of wt variants before selection', 'range' )
    plt.show()
   
def plot_ppc(ax, df_fit, var_plot, obs_data, t_func,p_title, xlabel):
    '''
    Plot posterior predictive checks for a particular summary statistic of observed and 
    
    Parameters
    ----------
    ax : plt.axes subclass
        to plot into
    df_fit : stan df_fit object
    var_plot : str, 
        such as 'c_pre'
    obs_data : dict
        dictionary fed into stan model for in sm.sampling() containing the observed data var_plot as key
    t_func : function(), needs to be able to accept axis argument
        ie. np.mean
    p_title : str
        plot title
    xlabel : str
        xlabel for plot
    
    '''
    c_pre_rep = df_fit1[[c for c in df_fit.columns if c.startswith(var_plot+'_rep')]]
    c_pre_rep_t = t_func(c_pre_rep, axis=1)
    
    obs_t = t_func(obs_data[var_plot])
    
    #plot
    ax.hist(c_pre_rep_t, bins=20, alpha=0.5, label='MCMC samples')
    ax.axvline(obs_t, label='observed data')
    ax.legend()
    ax.set(title= var_plot +' '+p_title, xlabel=xlabel, ylabel='Frequency');


## plotting replicate error bars
def create_aamut_col_df(d, df_fit, col_pre = 'diff_r'):
    ''' create df with aamut and column name to index aamut
    '''
    df_mut = pd.DataFrame.from_dict(list(d.items()), orient='columns')
    df_mut = df_mut.rename(columns={0:'aa_mut', 1:'num'})

    #get diff_r columns
    diff_cols = [c for c in df_fit.columns if c.startswith(col_pre)]
    #make dictionary of aamut number to diff_r
    num_to_col = dict(zip(map(int,[c.lstrip(col_pre)[1:-1] for c in diff_cols]), diff_cols))
    # convert to dataframe
    df_col = pd.DataFrame.from_dict(list(num_to_col.items()))
    df_col = df_col.rename(columns={0:'num', 1: 'col'})
    df_merge = df_mut.merge(df_col, left_on='num', right_on='num')
    return df_merge


def compute_HDI_upper(chain, interval=.95):
    return compute_HDI(chain, 'Upper', interval=.95)
def compute_HDI_lower(chain, interval=.95):
    return compute_HDI(chain, 'Lower', interval=.95)

def compute_HDI(chain,bound, interval = .95):
    '''
    computes HDI based on the shorted interval that can be found that spans 95% of samples
    ok for unimodal distributions, but will fail for multi-modal distributions if some mass in the middle needs to be excluded
    bound ['Upper', 'Lower']
    '''
    # sort chain using the first axis which is the chain
    chain.sort()
    # how many samples did you generate?
    nSample = chain.size    
    # how many samples must go in the HDI?
    nSampleCred = int(np.ceil(nSample * interval))
    # number of intervals to be compared
    nCI = nSample - nSampleCred
    # width of every proposed interval
    width = np.array([chain[i+nSampleCred] - chain[i] for  i in range(nCI)])
    # index of lower bound of shortest interval (which is the HDI) 
    best  = width.argmin()
    # put it in a dictionary
    HDI   = {'Lower': chain[best], 'Upper': chain[best + nSampleCred], 'Width': width.min()}
    return HDI[bound]

def create_df_mean_hdi(dic_aakey, df_fit, col_pre = 'diff_r'):
    '''
    takes a df_fit from stan, and returns a dataframe of the mean and 95% Highest density interval of the posterior
    for each growth rate
    returns df with mean_fit, hdi_lower, hdi_upper
    '''
    df = create_aamut_col_df(d, df_fit, col_pre=col_pre)
    samples = df_fit[df.col.values]

    df['mean_fit'] = np.mean(samples, axis=0).values

    df['hdi_lower'] = (samples.apply(compute_HDI_lower, axis=0, raw=True)).values
    df['hdi_upper'] = (samples.apply(compute_HDI_upper, axis=0, raw=True)).values
    return df

def plot_corr_reps_errs(df_fit1, df_fit2):
    # to plot the errorbar from poisson modeling
    df1 = create_df_mean_hdi(d, df_fit1)
    df2 = create_df_mean_hdi(d, df_fit2)
    plt.figure(figsize=(10,10))
    plt.errorbar(df1.mean_fit, df2.mean_fit, 
                 
                 xerr = [(df1.mean_fit- df1.hdi_lower).values, (df1.hdi_upper - df1.mean_fit).values], 
                 yerr = [(df2.mean_fit - df2.hdi_lower).values, (df2.hdi_upper - df2.mean_fit).values],
                 alpha=0.2, 
                 fmt='o',
                 markersize = 3,
                 elinewidth = 0.5
                )
    plt.xlabel('rep1')
    plt.ylabel('rep2')
    plt.show()