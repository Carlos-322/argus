"""Associate depth pixels using recorded image headers, CameraInfo and capture clocks."""
import re

import numpy as np


CAMERA_INFO = re.compile(r'(^|/)CameraInfo$')


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _integer(value, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('missing integer geometry')
    value = int(value)
    if value < (1 if positive else 0):
        raise ValueError('invalid integer geometry')
    return value


def _geometry(message):
    """Complete native pixel geometry; uncalibrated CameraInfo cannot prove registration."""
    result = {key: _integer(_field(message, key), True) for key in ('width', 'height')}
    model = _field(message, 'distortion_model')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('missing distortion model')
    result['distortion_model'] = model
    for key, count in (('k', 9), ('p', 12), ('r', 9), ('d', None)):
        value = _field(message, key)
        if value is None:
            value = _field(message, key.upper())
        array = np.asarray(value, dtype=np.float64)
        empty_declared = key == 'd' and model.strip().lower() in ('none', 'no_distortion')
        if (array.ndim != 1 or (not len(array) and not empty_declared)
                or (count is not None and len(array) != count) or not np.isfinite(array).all()):
            raise ValueError('missing or invalid calibration array')
        result[key] = array.tolist()
    coefficient_count = {'plumb_bob': 5, 'rational_polynomial': 8, 'equidistant': 4,
                         'none': 0, 'no_distortion': 0}.get(model.strip().lower())
    if coefficient_count is not None and len(result['d']) != coefficient_count:
        raise ValueError('invalid distortion coefficient count')
    if (result['k'][0] <= 0 or result['k'][4] <= 0 or result['p'][0] <= 0 or result['p'][5] <= 0
            or result['k'][8] != 1 or result['p'][10] != 1):
        raise ValueError('uncalibrated camera')
    rotation = np.asarray(result['r']).reshape(3, 3)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
            np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError('invalid rectification matrix')
    for key in ('binning_x', 'binning_y'):
        result[key] = _integer(_field(message, key))
    roi = _field(message, 'roi')
    result['roi'] = {key: _integer(_field(roi, key)) for key in ('x_offset', 'y_offset', 'width', 'height')}
    rectify = _field(roi, 'do_rectify')
    if not isinstance(rectify, (bool, np.bool_)):
        raise ValueError('missing crop rectification flag')
    result['roi']['do_rectify'] = bool(rectify)
    return result


def _transport_source(topic):
    parts = topic.rstrip('/').split('/')
    if parts[-1] in ('compressed', 'compressedDepth'):
        parts.pop()
    return '/'.join(parts)


def _namespace(topic):
    return _transport_source(topic).rsplit('/', 1)[0]


def _identity_pixel_mapping(geometry):
    """CameraInfo describes both raw and rectified geometry without identifying which image was used."""
    model = geometry['distortion_model'].strip().lower()
    projection = np.asarray(geometry['p']).reshape(3, 4)
    return (model in ('plumb_bob', 'rational_polynomial', 'none', 'no_distortion')
            and not np.any(geometry['d'])
            and np.array_equal(np.asarray(geometry['r']).reshape(3, 3), np.eye(3))
            and np.array_equal(projection[:, :3], np.asarray(geometry['k']).reshape(3, 3))
            and not np.any(projection[:, 3]))


class Registration:
    """Compact consistency summaries from the existing decode loop, never image copies."""

    def __init__(self):
        self.images, self.infos = {}, {}
        self.unreadable = set()

    def invalid_info(self, topic):
        self.infos.setdefault(topic, {})['valid'] = False
        self.unreadable.add(topic)

    def observe(self, topic, message, stamp, *, camera_info=False):
        from prepare.formats import mcap_capture_stamp
        records = self.infos if camera_info else self.images
        record = records.setdefault(topic, {'valid': True})
        header = _field(message, 'header')
        frame = _field(header, 'frame_id')
        header_stamp, _ = mcap_capture_stamp({'header': header})
        if not isinstance(frame, str) or not frame.strip() or header_stamp is None or header_stamp != stamp:
            record['valid'] = False
        if 'frame_id' in record and record['frame_id'] != frame:
            record['valid'] = False
        record['frame_id'] = frame
        if camera_info:
            try:
                geometry = _geometry(message)
                if 'geometry' in record and record['geometry'] != geometry:
                    record['valid'] = False
                record['geometry'] = geometry
            except (TypeError, ValueError, OverflowError):
                record['valid'] = False

    def _info(self, topic, shape, captures):
        image = self.images.get(topic, {})
        if not image.get('valid') or shape is None:
            return None
        records = [(name, info) for name, info in self.infos.items() if _namespace(name) == _namespace(topic)]
        if not records:
            return None
        geometry = None
        names = []
        stamps = set(captures.get(topic, []))
        for name, info in records:
            current = info.get('geometry')
            if (not info.get('valid') or info.get('frame_id') != image.get('frame_id') or current is None
                    or (current['width'], current['height']) != shape
                    or not stamps.intersection(captures.get(name, []))
                    or geometry is not None and geometry != current):
                return None
            geometry = current
            names.append(name)
        return geometry, names

    def resolve(self, depth_topic, cameras, shapes, captures):
        """One pixel geometry may have raw/compressed aliases; other geometries stay separate."""
        failed = {'status': 'unverified', 'depth_topic': depth_topic,
                  'limitation': 'Complete stable matching CameraInfo, image frame IDs, dimensions and capture clocks '
                                'do not establish an RGB pixel registration for this stream.'}
        if self.unreadable:
            failed['unreadable_camera_info_topics'] = sorted(self.unreadable)
        depth = self.images.get(depth_topic, {})
        info = self._info(depth_topic, shapes.get(depth_topic), captures)
        depth_stamps = captures.get(depth_topic, [])
        if (not info or len(depth_stamps) < 2 or any(b <= a for a, b in zip(depth_stamps, depth_stamps[1:]))):
            return None, failed
        geometry, depth_info = info
        if not _identity_pixel_mapping(geometry):
            failed['limitation'] = ('Recorded CameraInfo has a nonidentity or unknown raw-to-rectified pixel mapping; '
                                    'shared calibration alone does not identify the images\' actual processing geometry.')
            return None, failed
        matches = []
        for view, topic in cameras.items():
            if topic == depth_topic:
                continue
            rgb = self.images.get(topic, {})
            rgb_info = self._info(topic, shapes.get(topic), captures)
            stamps = captures.get(topic, [])
            common = set(depth_stamps).intersection(stamps)
            if (rgb_info and rgb.get('frame_id') == depth.get('frame_id') and rgb_info[0] == geometry
                    and len(common) >= 2 and all(b > a for a, b in zip(stamps, stamps[1:]))):
                matches.append((view, topic, rgb_info[1], common))
        if not matches:
            return None, failed
        if len({_transport_source(match[1]) for match in matches}) != 1:
            failed['limitation'] = ('Distinct image-processing topics cannot be treated as raw/compressed '
                                    'transport aliases solely from shared CameraInfo and frame IDs.')
            return None, failed
        # Prefer the native raw transport of the same recorded pixel geometry.
        matches.sort(key=lambda match: (match[1].endswith('/compressed'), len(match[1]), match[1]))
        view = matches[0][0]
        common = sorted(set.intersection(*(match[3] for match in matches)))
        if len(common) < 2:
            return None, failed
        return view, {'status': 'matching_recorded_geometry', 'depth_topic': depth_topic,
                      'rgb_topics': sorted(match[1] for match in matches), 'frame_id': depth['frame_id'],
                      'camera_info_topics': sorted(set(depth_info + [name for match in matches for name in match[2]])),
                      'camera_info': geometry, 'capture_clock': 'header.stamp',
                      'matched_capture_count': len(common), 'first_matched_capture_ns': common[0],
                      'last_matched_capture_ns': common[-1],
                      'limitation': 'The association follows recorded pixel geometry and common capture stamps; '
                                    'sensor accuracy, object tracking and world-space clearance are not verified.'}

    def has_geometry(self, topic):
        image = self.images.get(topic, {})
        return bool(image.get('frame_id') or any(_namespace(name) == _namespace(topic) for name in self.infos))
