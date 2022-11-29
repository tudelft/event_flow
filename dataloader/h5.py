import os
import h5py
import hdf5plugin
import numpy as np

import torch
import torch.utils.data as data

from .base import BaseDataLoader
from .utils import ProgressBar

from .encodings import binary_search_array


class Frames:
    """
    Utility class for reading the APS frames encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]


class FlowMaps:
    """
    Utility class for reading the optical flow maps encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]


class H5Loader(BaseDataLoader):
    def __init__(self, config, num_bins, round_encoding=False):
        super().__init__(config, num_bins, round_encoding)
        self.last_proc_timestamp = 0

        # "memory" that goes from forward pass to the next
        self.batch_idx = [i for i in range(self.config["loader"]["batch_size"])]  # event sequence
        self.batch_row = [
            0 for i in range(self.config["loader"]["batch_size"])
        ]  # event_idx / time_idx / frame_idx / gt_idx

        # input event sequences
        self.files = []
        for root, dirs, files in os.walk(config["data"]["path"]):
            for file in files:
                if file.endswith(".h5"):
                    self.files.append(os.path.join(root, file))

        # open first files
        self.open_files = []
        self.batch_last_ts = []
        for batch in range(self.config["loader"]["batch_size"]):
            self.open_files.append(h5py.File(self.files[batch], "r"))
            self.batch_last_ts.append(self.open_files[-1]["events/ts"][-1] - self.open_files[-1].attrs["t0"])

        # load frames from open files
        self.open_files_frames = []
        if self.config["data"]["mode"] == "frames":
            for batch in range(self.config["loader"]["batch_size"]):
                frames = Frames()
                self.open_files[batch]["images"].visititems(frames)
                self.open_files_frames.append(frames)

        # load GT optical flow maps from open files
        self.open_files_flowmaps = []
        if config["data"]["mode"] == "gtflow_dt1" or config["data"]["mode"] == "gtflow_dt4":
            for batch in range(self.config["loader"]["batch_size"]):
                flowmaps = FlowMaps()
                if config["data"]["mode"] == "gtflow_dt1":
                    self.open_files[batch]["flow_dt1"].visititems(flowmaps)
                elif config["data"]["mode"] == "gtflow_dt4":
                    self.open_files[batch]["flow_dt4"].visititems(flowmaps)
                self.open_files_flowmaps.append(flowmaps)

        # progress bars
        if self.config["vis"]["bars"]:
            self.open_files_bar = []
            for batch in range(self.config["loader"]["batch_size"]):
                max_iters = self.get_iters(batch)
                self.open_files_bar.append(ProgressBar(self.files[batch].split("/")[-1], max=max_iters))

    def get_iters(self, batch):
        """
        Compute the number of forward passes given a sequence and an input mode and window.
        """

        if self.config["data"]["mode"] == "events":
            max_iters = len(self.open_files[batch]["events/xs"])
        elif self.config["data"]["mode"] == "time":
            max_iters = self.open_files[batch].attrs["duration"]
        elif self.config["data"]["mode"] == "frames":
            max_iters = len(self.open_files_frames[batch].ts) - 1
        elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
            max_iters = len(self.open_files_flowmaps[batch].ts) - 1
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError

        return max_iters // self.config["data"]["window"]

    def get_events(self, file, idx0, idx1):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """

        xs = file["events/xs"][idx0:idx1]
        ys = file["events/ys"][idx0:idx1]
        ts = file["events/ts"][idx0:idx1]
        ps = file["events/ps"][idx0:idx1]
        ts -= file.attrs["t0"]  # sequence starting at t0 = 0
        if ts.shape[0] > 0:
            self.last_proc_timestamp = ts[-1]
        return xs, ys, ts, ps

    def get_event_index(self, batch, window=0):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx: event index
        """

        event_idx0 = None
        event_idx1 = None
        if self.config["data"]["mode"] == "events":
            event_idx0 = self.batch_row[batch]
            event_idx1 = self.batch_row[batch] + window
        elif self.config["data"]["mode"] == "time":
            event_idx0 = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"]
            )
            event_idx1 = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"] + window
            )
        elif self.config["data"]["mode"] == "frames":
            idx0 = int(np.floor(self.batch_row[batch]))
            idx1 = int(np.ceil(self.batch_row[batch] + window))
            if window < 1.0 and idx1 - idx0 > 1:
                idx0 += idx1 - idx0 - 1
            event_idx0 = self.find_ts_index(self.open_files[batch], self.open_files_frames[batch].ts[idx0])
            event_idx1 = self.find_ts_index(self.open_files[batch], self.open_files_frames[batch].ts[idx1])
        elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
            idx0 = int(np.floor(self.batch_row[batch]))
            idx1 = int(np.ceil(self.batch_row[batch] + window))
            if window < 1.0 and idx1 - idx0 > 1:
                idx0 += idx1 - idx0 - 1
            event_idx0 = self.find_ts_index(self.open_files[batch], self.open_files_flowmaps[batch].ts[idx0])
            event_idx1 = self.find_ts_index(self.open_files[batch], self.open_files_flowmaps[batch].ts[idx1])
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError
        return event_idx0, event_idx1

    def find_ts_index(self, file, timestamp):
        """
        Find closest event index for a given timestamp through binary search.
        """

        return binary_search_array(file["events/ts"], timestamp)

    def __getitem__(self, index):
        while True:
            batch = index % self.config["loader"]["batch_size"]

            # trigger sequence change
            len_frames = 0
            restart = False
            if self.config["data"]["mode"] == "frames":
                len_frames = len(self.open_files_frames[batch].ts)
            elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                len_frames = len(self.open_files_flowmaps[batch].ts)
            if (
                self.config["data"]["mode"] == "frames"
                or self.config["data"]["mode"] == "gtflow_dt1"
                or self.config["data"]["mode"] == "gtflow_dt4"
            ) and int(np.ceil(self.batch_row[batch] + self.config["data"]["window"])) >= len_frames:
                restart = True

            # load events
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))
            if not restart:
                idx0, idx1 = self.get_event_index(batch, window=self.config["data"]["window"])

                if (
                    self.config["data"]["mode"] == "frames"
                    or self.config["data"]["mode"] == "gtflow_dt1"
                    or self.config["data"]["mode"] == "gtflow_dt4"
                ) and self.config["data"]["window"] < 1.0:
                    floor_row = int(np.floor(self.batch_row[batch]))
                    ceil_row = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))
                    if ceil_row - floor_row > 1:
                        floor_row += ceil_row - floor_row - 1

                    idx0_change = self.batch_row[batch] - floor_row
                    idx1_change = self.batch_row[batch] + self.config["data"]["window"] - floor_row

                    delta_idx = idx1 - idx0
                    idx1 = int(idx0 + idx1_change * delta_idx)
                    idx0 = int(idx0 + idx0_change * delta_idx)

                xs, ys, ts, ps = self.get_events(self.open_files[batch], idx0, idx1)

            # trigger sequence change
            if (self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]) or (
                self.config["data"]["mode"] == "time"
                and self.batch_row[batch] + self.config["data"]["window"] >= self.batch_last_ts[batch]
            ):
                restart = True

            # handle case with very few events
            if xs.shape[0] <= 10:
                xs = np.empty([0])
                ys = np.empty([0])
                ts = np.empty([0])
                ps = np.empty([0])

            # reset sequence if not enough input events
            if restart:
                self.new_seq = True
                self.reset_sequence(batch)
                self.batch_row[batch] = 0
                self.batch_idx[batch] = max(self.batch_idx) + 1

                self.open_files[batch].close()
                self.open_files[batch] = h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r")
                self.batch_last_ts[batch] = self.open_files[batch]["events/ts"][-1] - self.open_files[batch].attrs["t0"]

                if self.config["data"]["mode"] == "frames":
                    frames = Frames()
                    self.open_files[batch]["images"].visititems(frames)
                    self.open_files_frames[batch] = frames
                elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                    flowmaps = FlowMaps()
                    if self.config["data"]["mode"] == "gtflow_dt1":
                        self.open_files[batch]["flow_dt1"].visititems(flowmaps)
                    elif self.config["data"]["mode"] == "gtflow_dt4":
                        self.open_files[batch]["flow_dt4"].visititems(flowmaps)
                    self.open_files_flowmaps[batch] = flowmaps
                if self.config["vis"]["bars"]:
                    self.open_files_bar[batch].finish()
                    max_iters = self.get_iters(batch)
                    self.open_files_bar[batch] = ProgressBar(
                        self.files[self.batch_idx[batch] % len(self.files)].split("/")[-1], max=max_iters
                    )

                continue

            # event formatting and timestamp normalization
            dt_input = np.asarray(0.0)
            if ts.shape[0] > 0:
                dt_input = np.asarray(ts[-1] - ts[0])
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # data augmentation
            xs, ys, ps = self.augment_events(xs, ys, ps, batch)

            # events to tensors
            event_cnt = self.create_cnt_encoding(xs, ys, ps)
            event_mask = self.create_mask_encoding(xs, ys, ps)
            event_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
            event_list = self.create_list_encoding(xs, ys, ts, ps)
            event_list_pol_mask = self.create_polarity_mask(ps)

            # hot pixel removal
            if self.config["hot_filter"]["enabled"]:
                hot_mask = self.create_hot_mask(event_cnt, batch)
                hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
                hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
                event_voxel = event_voxel * hot_mask_voxel
                event_cnt = event_cnt * hot_mask_cnt
                event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))

            # load frames when required
            if self.config["data"]["mode"] == "frames":
                curr_idx = int(np.floor(self.batch_row[batch]))
                next_idx = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))

                frames = np.zeros((2, self.config["loader"]["resolution"][0], self.config["loader"]["resolution"][1]))
                img0 = self.open_files[batch]["images"][self.open_files_frames[batch].names[curr_idx]][:]
                img1 = self.open_files[batch]["images"][self.open_files_frames[batch].names[next_idx]][:]
                frames[0, :, :] = self.augment_frames(img0, batch)
                frames[1, :, :] = self.augment_frames(img1, batch)
                frames = torch.from_numpy(frames.astype(np.uint8))

            # load GT optical flow when required
            dt_gt = 0.0
            if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                idx = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))
                if self.config["data"]["mode"] == "gtflow_dt1":
                    flowmap = self.open_files[batch]["flow_dt1"][self.open_files_flowmaps[batch].names[idx]][:]
                elif self.config["data"]["mode"] == "gtflow_dt4":
                    flowmap = self.open_files[batch]["flow_dt4"][self.open_files_flowmaps[batch].names[idx]][:]
                flowmap = self.augment_flowmap(flowmap, batch)
                flowmap = torch.from_numpy(flowmap.copy())
                if idx > 0:
                    dt_gt = self.open_files_flowmaps[batch].ts[idx] - self.open_files_flowmaps[batch].ts[idx - 1]
            dt_gt = np.asarray(dt_gt)

            # update window
            self.batch_row[batch] += self.config["data"]["window"]

            # break while loop if everything went well
            break

        # prepare output
        output = {}
        output["event_cnt"] = event_cnt
        output["event_voxel"] = event_voxel
        output["event_mask"] = event_mask
        output["event_list"] = event_list
        output["event_list_pol_mask"] = event_list_pol_mask
        if self.config["data"]["mode"] == "frames":
            output["frames"] = frames
        if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
            output["gtflow"] = flowmap
        output["dt_gt"] = torch.from_numpy(dt_gt)
        output["dt_input"] = torch.from_numpy(dt_input)

        return output
