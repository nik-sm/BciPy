"""Data server that streams EEG data over a LabStreamingLayer StreamOutlet
using pylsl."""
import logging
from queue import Empty, Queue
from typing import Generator

from pylsl import StreamInfo, StreamOutlet

from bcipy.acquisition.datastream.generator import random_data_generator
from bcipy.acquisition.datastream.producer import Producer
from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.util import StoppableThread

log = logging.getLogger(__name__)

# pylint: disable=too-many-arguments

MARKER_STREAM_NAME = 'TRG_device_stream'


class LslDataServer(StoppableThread):
    """Data server that streams EEG data over a LabStreamingLayer StreamOutlet
    using pylsl. See https://github.com/sccn/labstreaminglayer/wiki.

    In parameters.json, if the fake_data parameter is set to true and the
    device is set to LSL, this server will be used to mock data. Alternatively,
    fake_data can be set to false and this module can be run standalone in its
    own python instance.

    Parameters
    ----------
        device_spec : DeviceSpec
            parameters used to configure the server. Should have at least
            a list of channels and the sample frequency.
        generator : optional Generator (see generator.py for options)
            used to generate the data to be served. Uses random_data_generator
            by default.
        include_meta: bool, optional
            if True, writes metadata to the outlet stream.
        add_markers: bool, optional
            if True, creates a the marker channel and streams data to this
            channel at a fixed frequency.
        marker_stream_name: str, optional
            name of the sample marker stream
    """

    def __init__(self, device_spec: DeviceSpec, generator: Generator = None, include_meta: bool = True,
                 add_markers: bool = False, marker_stream_name: str = MARKER_STREAM_NAME):
        super(LslDataServer, self).__init__()

        self.device_spec = device_spec
        self.generator = generator or random_data_generator(channel_count=device_spec.channel_count)

        log.debug("Starting LSL server for device: %s", device_spec.name)
        print(f"Serving: {device_spec}")
        info = StreamInfo(device_spec.name,
                          device_spec.content_type, device_spec.channel_count,
                          device_spec.sample_rate,
                          'float32',
                          "uid12345")

        if include_meta:
            meta_channels = info.desc().append_child('channels')
            for channel in device_spec.channels:
                meta_channels.append_child('channel') \
                    .append_child_value('label', channel) \
                    .append_child_value('unit', 'microvolts') \
                    .append_child_value('type', 'EEG')

        self.outlet = StreamOutlet(info)

        self.add_markers = add_markers
        if add_markers:
            # Marker stream (C++ source below):
            # lsl::stream_info marker_info("gUSBamp-"+deviceNumber+"Markers",
            # "Markers",1,0,lsl::cf_string,
            # "gUSBamp_" + boost::lexical_cast<std::string>(deviceNumber) +
            # "_" + boost::lexical_cast<std::string>(serialNumber) +
            # "_markers");
            log.debug("Creating marker stream")
            print(f"Marker channel name: {marker_stream_name}")
            markers_info = StreamInfo(marker_stream_name,
                                      "Markers", 1, 0, 'string',
                                      "uid12345_markers")
            self.markers_outlet = StreamOutlet(markers_info)
        self.started = False

    def stop(self):
        """Stop the thread and cleanup resources."""

        log.debug("[*] Stopping data server")
        super(LslDataServer, self).stop()

        # Allows pylsl to cleanup; The outlet will no longer be discoverable
        # after destruction and all connected inlets will stop delivering data.
        del self.outlet
        self.outlet = None

        if self.add_markers:
            del self.markers_outlet
            self.markers_outlet = None

    def run(self):
        """Main loop of the thread. Continuously streams data to the stream
        outlet at a rate consistent with the sample frequency. May also
        output markers at a different interval."""

        sample_counter = 0
        self.started = True

        data_queue = Queue()
        with Producer(data_queue, generator=self.generator,
                      freq=1 / self.device_spec.sample_rate):
            while self.running():
                sample_counter += 1
                try:
                    sample = data_queue.get(True, 2)
                    self.outlet.push_sample(sample)
                    if self.add_markers and sample_counter % 1000 == 0:
                        self.markers_outlet.push_sample(["1"])
                except (Empty, AttributeError):
                    # outlet.push_sample(sample) may cause an error after
                    # the server has been stopped since the attribute is
                    # deleted in another thread.
                    break

        log.debug("[*] No longer pushing data")


def _settings(filename):
    """Read the daq settings from the given data file"""

    with open(filename, 'r') as datafile:
        daq_type = datafile.readline().strip().split(',')[1]
        sample_hz = int(datafile.readline().strip().split(',')[1])
        channels = datafile.readline().strip().split(',')
        return (daq_type, sample_hz, channels)


def main():
    """Initialize and start the server."""
    import time
    import argparse

    from bcipy.acquisition.datastream.generator import file_data_generator
    from bcipy.acquisition.devices import supported_device

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default=None,
                        help="file containing data to be streamed; "
                        "if missing, random data will be served.")
    parser.add_argument('-m', '--markers', action="store_true", default=False)
    parser.add_argument('-n', '--name', default='LSL', help='Name of the device spec to mock.')
    args = parser.parse_args()

    if args.filename:
        daq_type, sample_rate, channels = _settings(args.filename)
        device_spec = supported_device(daq_type)
        device_spec.sample_rate = sample_rate
        device_spec.channels = channels
        generator = file_data_generator(filename=args.filename)
    else:
        device_spec = supported_device(args.name)
        generator = None

    markers = True if args.markers else False
    try:
        server = LslDataServer(device_spec=device_spec, generator=generator, add_markers=markers)

        log.debug("New server created")
        server.start()
        log.debug("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    # Run this as: python -m bcipy.acquisition.datastream.lsl_server
    main()
