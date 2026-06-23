import logging
import queue
import threading

import h5py
import numpy as np
from src.settings import cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HDF5Writer:
    def __init__(self, filepath, chunk_size=cfg.hdf5_chunk_size):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.queue = queue.Queue()

        self.obs_buffer = []
        self.speed_buffer = []
        self.gear_buffer = []
        self.rpm_buffer = []
        self.action_buffer = []

        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)

        self.file = h5py.File(self.filepath, "w")

        self.dset_obs = self.file.create_dataset(
            "observations",
            shape=(0, *cfg.image_shape),
            maxshape=(None, *cfg.image_shape),
            chunks=(self.chunk_size, *cfg.image_shape),
            dtype=np.uint8,
            compression=cfg.compression,
        )

        self.dset_speed = self.file.create_dataset(
            "speed",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype=np.float32,
        )
        self.dset_gear = self.file.create_dataset(
            "gear",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype=np.int32,
        )
        self.dset_rpm = self.file.create_dataset(
            "rpm",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype=np.float32,
        )

        self.dset_actions = self.file.create_dataset(
            "actions",
            shape=(0, cfg.action_dim),
            maxshape=(None, cfg.action_dim),
            chunks=(self.chunk_size, cfg.action_dim),
            dtype=np.float32,
            compression=cfg.compression,
        )

        self.current_size = 0
        self.thread.start()
        logging.info(f"Started HDF5Writer for {self.filepath}")

    def append(self, obs, telemetry, action):
        obs_array = np.copy(obs)
        if obs_array.dtype in [np.float32, np.float64]:
            if obs_array.max() <= 1.0:
                obs_array = (obs_array * 255.0).astype(np.uint8)
            else:
                obs_array = obs_array.astype(np.uint8)

        self.queue.put(
            {
                "obs": obs_array,
                "speed": float(telemetry.get("speed", 0.0)),
                "gear": int(telemetry.get("gear", 0)),
                "rpm": float(telemetry.get("rpm", 0.0)),
                "action": np.copy(action),
            }
        )

    def _writer_loop(self):
        while self.running or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.1)
                self.obs_buffer.append(data["obs"])
                self.speed_buffer.append(data["speed"])
                self.gear_buffer.append(data["gear"])
                self.rpm_buffer.append(data["rpm"])
                self.action_buffer.append(data["action"])

                if len(self.obs_buffer) >= self.chunk_size:
                    self._flush()

            except queue.Empty:
                continue

    def _flush(self):
        n = len(self.obs_buffer)
        if n == 0:
            return

        new_size = self.current_size + n

        self.dset_obs.resize(new_size, axis=0)
        self.dset_speed.resize(new_size, axis=0)
        self.dset_gear.resize(new_size, axis=0)
        self.dset_rpm.resize(new_size, axis=0)
        self.dset_actions.resize(new_size, axis=0)

        self.dset_obs[self.current_size : new_size] = np.stack(self.obs_buffer)
        self.dset_speed[self.current_size : new_size] = np.array(
            self.speed_buffer, dtype=np.float32
        )
        self.dset_gear[self.current_size : new_size] = np.array(
            self.gear_buffer, dtype=np.int32
        )
        self.dset_rpm[self.current_size : new_size] = np.array(
            self.rpm_buffer, dtype=np.float32
        )
        self.dset_actions[self.current_size : new_size] = np.stack(self.action_buffer)

        self.current_size = new_size

        self.obs_buffer.clear()
        self.speed_buffer.clear()
        self.gear_buffer.clear()
        self.rpm_buffer.clear()
        self.action_buffer.clear()

    def close(self):
        self.running = False
        self.thread.join()
        self._flush()
        self.file.close()
        logging.info(f"Closed HDF5Writer. Total frames written: {self.current_size}")
