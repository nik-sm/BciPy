"""GUI tool to visualize offset analysis when analyzing system latency."""
import os
import pathlib
from bcipy.helpers.load import read_triggers, read_csv_data
from bcipy.helpers.vizualization import plot_triggers


def main(path: str):
    """Run the plot triggers from the display and acquisition file with offset values considered

    Parameters:
    -----------
        data_file - raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much the data is downsampled. A factor of 1
            displays the raw data.
    """
    data_file = os.path.join(path, 'raw_data.csv')
    trg_file = os.path.join(path, 'triggers.txt')
    data, device_info = read_csv_data(data_file)
    triggers = read_triggers(trg_file)

    plot_triggers(data, device_info, triggers, title=pathlib.Path(path).name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graphs trigger data from a bcipy session to visualize system latency"
    )
    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)
    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    main(path)
