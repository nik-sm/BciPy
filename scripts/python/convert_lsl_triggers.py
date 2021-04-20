from bcipy.helpers.triggers import write_lsl_triggers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw_data.csv triggers from LSL to text."
    )
    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)

    parser.add_argument('--calibration', dest='calibration', action='store_true')
    parser.set_defaults(calibration=False)
    args = parser.parse_args()
    path = args.path
    calibration = args.calibration
    if not path:
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    path = write_lsl_triggers(path, calibration=calibration)
    print(f'Converted triggers written to {path}')
