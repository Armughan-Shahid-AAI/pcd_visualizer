class Sensor(object):
    def __init__(self, sensor_id, view):
        self.sensor_id = sensor_id
        self.view = view


class CameraSensor(Sensor):
    def __init__(self, sensor_id, view, tstamp_delay, lens, cam_matrix, cam_matrix_original, distortion, resolution):
        super().__init__(sensor_id, view)
        self.tstamp_delay = tstamp_delay
        self.lens = lens
        self.cam_matrix = cam_matrix
        self.cam_matrix_original = cam_matrix_original
        self.distortion = distortion
        self.resolution = resolution

    def __repr__(self):
        return "CameraSensor: " + self.sensor_id


class LidarSensor(Sensor):
    def __init__(self, sensor_id, view,):
        super().__init__(sensor_id, view)

    def __repr__(self):
        return "LidarSensor: " + self.sensor_id