import os
import numpy as np
import matplotlib.pyplot as plt
from bcipy.helpers.load import load_csv_data, read_data_csv
from mne.io import read_raw_edf

import logging
log = logging.getLogger(__name__)


def generate_offline_analysis_screen(
        x,
        y,
        model=None,
        folder=None,
        plot_lik_dens=True,
        save_figure=True,
        down_sample_rate=2,
        fs=300,
        plot_x_ticks=8,
        plot_average=False,
        show_figure=False,
        channel_names=None) -> None:
    """ Offline Analysis Screen.

    Generates the information figure following the offlineAnalysis.
    The figure has multiple tabs containing the average ERP plots

    PARAMETERS
    ----------

    x(ndarray[float]): C x N x k data array
    y(ndarray[int]): N x k observation (class) array
        N is number of samples k is dimensionality of features
        C is number of channels
    model(): trained model for data
    folder(str): Folder of the data
    plot_lik_dens: boolean: whether or not to plot likelihood densities
    save_figures: boolean: whether or not to save the plots as PDF
    down_sample_rate: downsampling rate applied to signal (if any)
    fs (sampling_rate): original sampling rate of the signal
    plot_x_ticks: number of ticks desired on the ERP plot
    plot_average: boolean: whether or not to average over all channels
    show_figure: boolean: whether or not to show the figures generated
    channel_names: dict of channel names keyed by their position.
    """

    channel_names = channel_names or {}
    classes = np.unique(y)

    means = [np.squeeze(np.mean(x[:, np.where(y == i), :], 2))
             for i in classes]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    if plot_average:
        # find the mean across rows for non target and target
        non_target_mean = np.mean(means[0], axis=0)
        target_mean = np.mean(means[1], axis=0)

        # plot the means
        ax1.plot(non_target_mean)
        ax2.plot(target_mean)

    else:
        count = 0
        # Loop through all the channels and plot each on the non target/target
        # subplots
        while count < means[0].shape[0]:
            lbl = channel_names.get(count, count)
            ax1.plot(means[0][count, :], label=lbl)
            ax2.plot(means[1][count, :], label=lbl)
            count += 1
        ax1.legend(loc='upper left', prop={'size': 8})
        ax2.legend(loc='upper left', prop={'size': 8})

    # data points
    data_length = len(means[0][1, :])

    # generate appropriate data labels for the figure
    lower = 0

    # find the upper length of data and convert to seconds
    upper = data_length * down_sample_rate / fs * 1000

    # make the labels
    labels = [round(lower + x * (upper - lower) / (plot_x_ticks - 1)) for x in
              range(plot_x_ticks)]

    # make sure it starts at zero
    labels.insert(0, 0)

    # set the labels
    ax1.set_xticklabels(labels)
    ax2.set_xticklabels(labels)

    # Set common labels
    fig.text(0.5, 0.04, 'Time (Seconds)', ha='center', va='center')
    fig.text(0.06, 0.5, r'$\mu V$', ha='center', va='center',
             rotation='vertical')

    ax1.set_title('Non-target ERP')
    ax2.set_title('Target ERP')

    if save_figure:
        fig.savefig(
            os.path.join(
                folder,
                'mean_erp.pdf'),
            bbox_inches='tight',
            format='pdf')

    if plot_lik_dens:
        fig, ax = plt.subplots()
        x_plot = np.linspace(
            np.min(model.line_el[-2]), np.max(model.line_el[-2]), 1000)[:, np.newaxis]
        ax.plot(model.line_el[2][y == 0], -0.005 - 0.01 * np.random.random(
            model.line_el[2][y == 0].shape[0]), 'ro', label='class(-)')
        ax.plot(model.line_el[2][y == 1], -0.005 - 0.01 * np.random.random(
            model.line_el[2][y == 1].shape[0]), 'go', label='class(+)')
        for idx in range(len(model.pipeline[2].list_den_est)):
            log_dens = model.pipeline[2].list_den_est[idx].score_samples(
                x_plot)
            ax.plot(x_plot[:, 0], np.exp(log_dens), 'r-' *
                    (idx == 0) + 'g--' * (idx == 1), linewidth=2.0)

        ax.legend(loc='upper right')
        plt.title('Likelihoods Given the Labels')
        plt.ylabel('p(e|l)')
        plt.xlabel('scores')

        if save_figure:
            fig.savefig(
                os.path.join(
                    folder,
                    'lik_dens.pdf'),
                bbox_inches='tight',
                format='pdf')

    if show_figure:
        plt.show()


def plot_edf(edf_path: str, auto_scale: bool = False):
    """Plot data from the raw edf file. Note: this works from an iPython
    session but seems to throw errors when provided in a script.

    Parameters
    ----------
        edf_path - full path to the generated edf file
        auto_scale - optional; if True will scale the EEG data; this is
            useful for fake (random) data but makes real data hard to read.
    """
    edf = read_raw_edf(edf_path, preload=True)
    if auto_scale:
        edf.plot(scalings='auto')
    else:
        edf.plot()


def visualize_csv_eeg_triggers(trigger_col=None):
    """Visualize CSV EEG Triggers.

    This function is used to load in CSV data and visualize device generated triggers.

    Input:
        trigger_col(int)(optional): Column location of triggers in csv file.
            It defaults to the last column.

    Output:
        Figure of Triggers
    """
    # Load in CSV
    file_name = load_csv_data()
    raw_data, stamp_time, channels, type_amp, fs = read_data_csv(file_name)

    # Pull out the triggers
    if not trigger_col:
        triggers = raw_data[-1]
    else:
        triggers = raw_data[trigger_col]

    # Plot the triggers
    plt.plot(triggers)

    # Add some titles and labels to the figure
    plt.title('Trigger Signal')
    plt.ylabel('Trigger Value')
    plt.xlabel('Samples')

    log.debug('Press Ctrl + C to exit!')
    # Show us the figure! Depending on your OS / IDE this may not close when
    #  The window is closed, see the message above
    plt.show()


def channel_data(raw_data, device_info, channel_name, n_records=None):
    """Get data for a single channel.
    Parameters:
    -----------
        raw_data - complete list of samples
        device_info - metadata
        channel_name - channel for which to get data
        n_records - if present, limits the number of records returned.
    """
    if channel_name not in device_info.channels:
        print(f"{channel_name} column not found; no data will be returned")
        return []
    channel_index = device_info.channels.index(channel_name)
    arr = np.array(raw_data)
    if n_records:
        return arr[:n_records, channel_index]
    return arr[:, channel_index]


def clock_seconds(device_info: DeviceInfo, sample: int) -> float:
    """Convert the given raw_data sample number to acquisition clock
    seconds."""
    assert sample > 0
    return sample / device_info.fs


def plot_triggers(raw_data, device_info, triggers, title=""):
    """Plot raw_data triggers, including the TRG_device_stream data
    (channel streamed from the device; usually populated from a trigger box),
    as well as TRG data populated from the LSL Marker Stream. Also plots data
    from the triggers.txt file if available.

    Parameters:
    -----------
        raw_data - complete list of samples read in from the raw_data.csv file.
        device_info - metadata about the device including the sample rate.
        triggers - list of (trg, trg_type, stamp) values from the triggers.txt
            file. Stamps have been converted to the acquisition clock using
            the offset recorded in the triggers.txt.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('acquisition clock (secs)')
    plt.ylabel('TRG value')
    if title:
        plt.title(title)

    trg_ymax = 1.5

    # Plot TRG_device_stream column; this is a continuous line.
    trg_box_channel = channel_data(raw_data, device_info, 'TRG_device_stream')
    first_trg_box_time = None
    trg_box_y = []
    trg_box_x = []
    for i, val in enumerate(trg_box_channel):
        timestamp = clock_seconds(device_info, i + 1)
        value = int(float(val))
        trg_box_x.append(timestamp)
        trg_box_y.append(value)
        if not first_trg_box_time and value == 1:
            first_trg_box_time = timestamp

    ax.plot(trg_box_x, trg_box_y, label='TRG_device_stream (trigger box)')
    if first_trg_box_time:
        ax.annotate(s=f"{round(first_trg_box_time, 2)}s", xy=(first_trg_box_time, 1.25),
                    fontsize='small', color='#1f77b4', horizontalalignment='right',
                    rotation=270)

    # Plot triggers.txt data if present; vertical line for each value.
    if triggers:
        plt.vlines([stamp for (_name, _trgtype, stamp) in triggers],
                   ymin=-1.0, ymax=trg_ymax, label='triggers.txt (adjusted)',
                   linewidth=0.5, color='cyan')

    # Plot TRG column, vertical line for each one.
    trg_channel = channel_data(raw_data, device_info, 'TRG')
    trg_stamps = [clock_seconds(device_info, i + 1)
                  for i, trg in enumerate(trg_channel) if trg != '0' and trg != '0.0']
    plt.vlines(trg_stamps, ymin=-1.0, ymax=trg_ymax, label='TRG (LSL)',
               linewidth=0.5, color='red')

    # Add labels for TRGs
    first_trg = None
    for i, trg in enumerate(trg_channel):
        if trg != '0' and trg != '0.0':
            secs = clock_seconds(device_info, i + 1)
            secs_lbl = str(round(secs, 2))
            ax.annotate(s=f"{trg} ({secs_lbl}s)", xy=(secs, trg_ymax),
                        fontsize='small', color='red', horizontalalignment='right',
                        rotation=270)
            if not first_trg:
                first_trg = secs

    # Set initial zoom to +-5 seconds around the calibration_trigger
    if first_trg:
        ax.set_xlim(left=first_trg - 5, right=first_trg + 5)

    ax.grid(axis='x', linestyle='--', color="0.5", linewidth=0.4)
    plt.legend(loc='lower left', fontsize='small')
    # display the plot
    plt.show()


if __name__ == '__main__':
    import pickle

    # load some x, y data from test files
    x = pickle.load(
        open(
            'bcipy/helpers/tests/resources/mock_x_generate_erp.pkl',
            'rb'))
    y = pickle.load(
        open(
            'bcipy/helpers/tests/resources/mock_y_generate_erp.pkl',
            'rb'))

    names = {
        0: 'P3',
        1: 'C3',
        2: 'F3',
        3: 'Fz',
        4: 'F4',
        5: 'C4',
        6: 'P4',
        7: 'Cz',
        8: 'A1',
        9: 'Fp1',
        10: 'Fp2',
        11: 'T3',
        12: 'T5',
        13: 'O1',
        14: 'O2',
        15: 'F7',
        16: 'F8',
        17: 'A2',
        18: 'T6',
        19: 'T4'
    }
    # generate the offline analysis screen. show figure at the end
    generate_offline_analysis_screen(
        x,
        y,
        folder='bcipy',
        plot_lik_dens=False,
        save_figure=False,
        show_figure=True,
        channel_names=names)
