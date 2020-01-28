import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from copy import deepcopy
from constants import WT_PARE_DNA_AA, WT_PARD_DNA_AA

AA_LIST_ALPHABETICAL = "ACDEFGHIKLMNPQRSTVWY_*"
AA_LIST_ALPHABETICAL = "_NQAILVMFWYCGPDERKHST*"


################################################################
# Posterior predictive check functions
################################################################
def get_range(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def plot_all_ppcs(df_fit, var_plot, obs_data):
    '''
    Plot posterior predictive distribution for mean, stdev, max and range of count data

    Parameters
    ----------
    df_fit : stan df_fit object
    var_plot : str, ie. 'c_pre'
    obs_data : dict
        dictionary fed into stan model for in sm.sampling() containing the observed data var_plot as key
    '''

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)  # , height_ratios=[3, 1], width_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    plot_ppc(ax0, df_fit, var_plot, obs_data, np.mean, 'mean count of wt variants before selection', 'mean')
    ax1 = plt.subplot(gs[1])

    plot_ppc(ax1, df_fit, var_plot, obs_data, np.std, 'stdev count of wt variants before selection', 'stdev')

    ax2 = plt.subplot(gs[2])
    plot_ppc(ax2, df_fit, var_plot, obs_data, np.max, 'max count of wt variants before selection', 'max')

    ax3 = plt.subplot(gs[3])
    plot_ppc(ax3, df_fit, var_plot, obs_data, get_range, 'range count of wt variants before selection', 'range')
    plt.show()


def plot_ppc(ax, df_fit, var_plot, obs_data, t_func, p_title, xlabel):
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
    c_pre_rep = df_fit[[c for c in df_fit.columns if c.startswith(var_plot + '_rep')]]
    c_pre_rep_t = t_func(c_pre_rep, axis=1)

    obs_t = t_func(obs_data[var_plot])

    # plot
    ax.hist(c_pre_rep_t, bins=20, alpha=0.5, label='MCMC samples')
    ax.axvline(obs_t, label='observed data')
    ax.legend()
    ax.set(title=var_plot + ' ' + p_title, xlabel=xlabel, ylabel='Frequency');


################################################################
# plotting replicates
################################################################ 
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


def create_aamut_col_df(d, df_fit, col_pre='diff_r'):
    ''' create df with aamut and column name to index aamut
    '''
    df_mut = pd.DataFrame.from_dict(list(d.items()), orient='columns')
    df_mut = df_mut.rename(columns={0: 'aa_mut', 1: 'num'})

    # get diff_r columns
    diff_cols = [c for c in df_fit.columns if c.startswith(col_pre)]
    # make dictionary of aamut number to diff_r
    num_to_col = dict(zip(map(int, [c.lstrip(col_pre)[1:-1] for c in diff_cols]), diff_cols))
    # convert to dataframe
    df_col = pd.DataFrame.from_dict(list(num_to_col.items()))
    df_col = df_col.rename(columns={0: 'num', 1: 'col'})
    df_merge = df_mut.merge(df_col, left_on='num', right_on='num')
    return df_merge


def compute_HDI_upper(chain, interval=.95):
    return compute_HDI(chain, 'Upper', interval=.95)


def compute_HDI_lower(chain, interval=.95):
    return compute_HDI(chain, 'Lower', interval=.95)


def compute_HDI(chain, bound, interval=.95):
    '''
    computes HDI based on the shorted interval that can be found that spans 95% of samples
    ok for unimodal distributions, but will fail for multi-modal distributions if some mass in the middle needs to be 
    excluded
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
    width = np.array([chain[i + nSampleCred] - chain[i] for i in range(nCI)])
    # index of lower bound of shortest interval (which is the HDI) 
    best = width.argmin()
    # put it in a dictionary
    HDI = {'Lower': chain[best], 'Upper': chain[best + nSampleCred], 'Width': width.min()}
    return HDI[bound]


def create_df_mean_hdi(dic_aakey, df_fit, col_pre='diff_r'):
    '''
    takes a df_fit from stan, and returns a dataframe of the mean and 95% Highest density interval of the posterior
    for each growth rate
    returns df with mean_fit, hdi_lower, hdi_upper
    '''
    df = create_aamut_col_df(dic_aakey, df_fit, col_pre=col_pre)
    samples = df_fit[df.col.values]

    df['mean_fit'] = np.mean(samples, axis=0).values

    df['hdi_lower'] = (samples.apply(compute_HDI_lower, axis=0, raw=True)).values
    df['hdi_upper'] = (samples.apply(compute_HDI_upper, axis=0, raw=True)).values
    return df


def plot_corr_reps_errs(dic_aakey,df_fit1, df_fit2):
    # to plot the errorbar from poisson modeling
    df1 = create_df_mean_hdi(dic_aakey, df_fit1)
    df2 = create_df_mean_hdi(dic_aakey, df_fit2)
    plt.figure(figsize=(10, 10))
    plt.errorbar(df1.mean_fit, df2.mean_fit,

                 xerr=[(df1.mean_fit - df1.hdi_lower).values, (df1.hdi_upper - df1.mean_fit).values],
                 yerr=[(df2.mean_fit - df2.hdi_lower).values, (df2.hdi_upper - df2.mean_fit).values],
                 alpha=0.2,
                 fmt='o',
                 markersize=3,
                 elinewidth=0.5
                 )
    plt.xlabel('rep1')
    plt.ylabel('rep2')
    plt.show()


################################################################
# plot heatmap
################################################################
def plot_fitness_heatmap_aa(df_rm):
    # converting dataframe into data_dic for feeding into plot_mutation_matrix
    data_dic = {}
    for i, r in df_rm.iterrows():
        aa_mut = r['aa_mut']

        pos = int(aa_mut[1:-1])
        mut_aa = aa_mut[-1]
        if pos not in data_dic:
            data_dic[pos] = {}
            data_dic[pos][mut_aa] = float(r['mean_fit'])
        else:
            data_dic[pos][mut_aa] = float(r['mean_fit'])

        # reference
        data_dic[0] = {}
        data_dic[0]["M"] = 0

    plot_mutation_matrix(
        data_dic, template='pare',
        plottingFitness=True,
    )


def plot_mutation_matrix(
        data_points_dict,
        template,
        transformation=lambda x: x,
        dot_marker=True,
        show_grid=False,
        fout=None,
        plot_title=None,
        conservation=None,
        aa_list=AA_LIST_ALPHABETICAL,
        use_new=True,
        show_wt_aa=True,
        label_filter=lambda x: True,
        readThreshold=None,
        plottingFitness=False,
        **kwargs
):
    plotFit = plottingFitness

    longest_residue_len = 1  # len(data_points_dict)
    position_labels = []

    mutation_matrix = np.zeros((len(data_points_dict), len(aa_list)))
    matrix_columns = enumerate(sorted(data_points_dict.keys()))

    wt_sequence_markers = []
    if template == "pare":
        wt_seq = WT_PARE_DNA_AA
    elif template == "pard":
        wt_seq = WT_PARD_DNA_AA

    if use_new:
        aa_list = list(reversed(aa_list))

    for i, column in enumerate(sorted(data_points_dict.keys())):
        for j, res_to in enumerate(aa_list):
            if column in data_points_dict and res_to in data_points_dict[column]:
                mutation_matrix[i, j] = transformation(data_points_dict[column][res_to])
            else:
                mutation_matrix[i, j] = float("nan")
        wt_res = wt_seq[i]
        wt_sequence_markers.append(wt_res)

        if label_filter(column):
            if show_wt_aa:
                position_labels.append(
                    "{aa:>1}{index:>{maxlength}}".format(
                        aa=wt_res, index=column, maxlength=longest_residue_len
                    )
                )
            else:
                position_labels.append(
                    "{index:>{maxlength}}".format(
                        index=column, maxlength=longest_residue_len
                    )
                )
        else:
            position_labels.append("")

    mutation_matrix_masked = np.ma.masked_where(
        np.isnan(mutation_matrix), mutation_matrix
    )

    # set min and max, and colormap
    colormap = deepcopy(plt.cm.GnBu)

    # to choose diverging colormap for plotting fitness
    if plotFit == True:
        colormap = deepcopy(plt.cm.coolwarm)
        # colormap = deepcopy(plt.cm.Blues)

    plot_mutation_matrix_base_v2(
        mutation_matrix,
        matrix_columns,
        aa_list,
        colormap,
        output_file=fout,
        position_labels=position_labels,
        dot_marker=dot_marker,
        wt_sequence_markers=wt_sequence_markers,
        plot_title=plot_title,
        conservation=conservation,
        readThreshold=readThreshold,
        plottingFitness=plotFit,
        **kwargs
    )


def plot_mutation_matrix_base_v2(
        mutation_matrix,
        matrix_columns,
        aa_list,
        colormap,
        position_labels=None,
        wt_sequence_markers=None,
        output_file=None,
        dot_marker=True,
        m_max=None,
        m_min=None,
        plot_title=None,
        secondary_structure=None,
        conservation=None,
        position_label_size=8,
        aa_label_size=8,
        colorbar_label_size=8,
        font_override=None,
        colorbar_label_width=5,
        colorbar_indicate_bounds=True,
        dpi=300,
        readThreshold=None,
        plottingFitness=False,
):
    from matplotlib import rc_context

    LINEWIDTH = 0.0
    LABEL_X_OFFSET = 0.55
    LABEL_Y_OFFSET = 0.45

    def _draw_rect(x_range, y_range, linewidth):
        r = plt.Rectangle(
            (min(x_range), min(y_range)),
            max(x_range) - min(x_range),
            max(y_range) - min(y_range),
            fc="None",
            linewidth=linewidth,
        )
        plt.gca().add_patch(r)

    if font_override is None:
        font = "Helvetica"
        # font="DejaVu Sans Mono" if use_monospace else "Helvetica"
    else:
        font = font_override

    with rc_context(
            {
                "font.family"    : font,
                "ytick.labelsize": colorbar_label_size,
                "pdf.fonttype"   : 42,
            }
    ):
        matrix_width = mutation_matrix.shape[0]
        matrix_height = len(aa_list)

        # mask NaN entries in mutation matrix
        mutation_matrix_masked = np.ma.masked_where(
            np.isnan(mutation_matrix), mutation_matrix
        )

        # figure out maximum and minimum values for color map
        if m_max is None or m_min is None:

            # normal standardization
            m_max = np.abs(mutation_matrix_masked).max()

            if plottingFitness == True:
                m_min = -m_max
                m_max = m_max
            else:
                m_min = 0

        # start plotting
        num_rows = (
                len(aa_list)
                + (conservation is not None)
                + (secondary_structure is not None)
        )
        ratio = matrix_width / float(num_rows)
        fig = plt.figure(figsize=(ratio * 5, 5), dpi=dpi)
        plt.gca().set_aspect("equal", "box")

        # define matrix coordinates
        # always add +1 because coordinates are used by
        # pcolor(mesh) as beginning and start of rectangles
        x_range = range(matrix_width + 1)
        y_range = range(matrix_height + 1)
        y_range_avg = range(-2, 0)
        x_range_avg = range(matrix_width + 1, matrix_width + 3)
        y_range_cons = np.array(y_range_avg) - 1.5

        # coordinates for text labels (fixed axis)
        x_left_aa = min(x_range) - 1
        x_right_aa = max(x_range_avg) + 1

        if conservation is None:
            y_bottom_res = min(y_range_avg) - 0.5
        else:
            y_bottom_res = min(y_range_cons) - 0.5

        # coordinates for additional annotation
        y_ss = max(y_range) + 2
        ss_width = 0.8

        # 1) main mutation matrix
        X, Y = np.meshgrid(x_range, y_range)
        cm = plt.pcolormesh(
            X, Y, mutation_matrix_masked.T, cmap=colormap, vmax=m_max, vmin=m_min
        )
        _draw_rect(x_range, y_range, LINEWIDTH)

        # 2) sum column effect (bottom "subplot")
        sum_pos = np.sum(mutation_matrix_masked, axis=1)[:, np.newaxis]
        X_pos, Y_pos = np.meshgrid(x_range, y_range_avg)
        plt.pcolormesh(
            X_pos,
            Y_pos,
            sum_pos.T,
            cmap=plt.cm.RdPu,
            vmax=max(sum_pos),
            vmin=min(sum_pos),
        )
        _draw_rect(x_range, y_range_avg, LINEWIDTH)

        # 3) amino acid average (right "subplot")
        mean_aa = np.mean(mutation_matrix_masked, axis=0)[:, np.newaxis]
        X_aa, Y_aa = np.meshgrid(x_range_avg, y_range)
        plt.pcolormesh(X_aa, Y_aa, mean_aa, cmap=colormap, vmax=m_max, vmin=m_min)
        _draw_rect(x_range_avg, y_range, LINEWIDTH)

        # mark wildtype residues
        if wt_sequence_markers:
            for i, aa in enumerate(wt_sequence_markers):
                # skip unspecified entries
                if aa is not None and aa != "":
                    if dot_marker:
                        marker = plt.Circle(
                            (x_range[i] + 0.5, y_range[aa_list.index(aa)] + 0.5),
                            0.1,
                            fc="k",
                        )
                    else:
                        marker = plt.Rectangle(
                            (i, x_range[aa_list.index(aa)]),
                            1,
                            1,
                            fc="None",
                            linewidth=LINEWIDTH,
                        )
                    plt.gca().add_patch(marker)

        # put tick labels on
        if position_labels:
            for i, res in zip(x_range, position_labels):
                plt.text(
                    i + LABEL_X_OFFSET,
                    y_bottom_res,
                    res,
                    size=position_label_size,
                    fontname=font,
                    horizontalalignment="center",
                    verticalalignment="top",
                    rotation=90,
                )

        for j, aa in zip(y_range, aa_list):
            plt.text(
                x_left_aa,
                j + LABEL_Y_OFFSET,
                aa,
                size=aa_label_size,
                fontname=font,
                horizontalalignment="center",
                verticalalignment="center",
            )

            plt.text(
                x_right_aa,
                j + LABEL_Y_OFFSET,
                aa,
                size=aa_label_size,
                fontname=font,
                horizontalalignment="center",
                verticalalignment="center",
            )

        plt.xlim(
            [
                min(list(x_range) + list(x_range_avg)),
                max(list(x_range) + list(x_range_avg)),
            ]
        )

        # draw colorbar
        cb = plt.colorbar(
            cm, ticks=[m_min, m_max], shrink=0.3, pad=0.15 / ratio, aspect=8
        )

        if colorbar_indicate_bounds:
            symbol_min, symbol_max = u"\u2264", u"\u2265"
        else:
            symbol_min, symbol_max = "", ""

        cb.ax.set_yticklabels(
            [
                u"{symbol} {value:>+{width}.10f}".format(
                    symbol=s, value=v, width=colorbar_label_width
                )
                for (v, s) in [(m_min, symbol_min), (float(m_max), symbol_max)]
            ]
        )
        cb.ax.xaxis.set_ticks_position("none")
        cb.ax.yaxis.set_ticks_position("none")
        cb.outline.set_linewidth(0)

        # plot conservation
        if conservation is not None:
            cons = np.array(
                [conservation.get(i, float("nan")) for i in matrix_columns]
            )[:, np.newaxis]
            cons_masked = np.ma.masked_where(np.isnan(cons), cons)
            X_cons, Y_cons = np.meshgrid(x_range, y_range_cons)

            oranges = deepcopy(plt.cm.Oranges)
            oranges.set_bad(color="0.75", alpha=None)
            plt.pcolormesh(X_cons, Y_cons, cons_masked.T, cmap=oranges, vmax=1, vmin=0)
            _draw_rect(x_range, y_range_cons, LINEWIDTH)

        # remove chart junk
        for line in ["top", "bottom", "right", "left"]:
            plt.gca().spines[line].set_visible(False)
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), visible=False)

        if plot_title is not None:

            if readThreshold:
                plt.title(plot_title + "readThreshold=" + readThreshold)
            else:
                plt.title(plot_title)

        if output_file is not None:
            print(output_file)

            plt.savefig(output_file + ".pdf", format="pdf", bbox_inches="tight")
            plt.savefig(output_file + ".eps", format="eps", bbox_inches="tight")
        else:
            plt.show()
