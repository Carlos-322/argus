"""Strict canonical CDR recovery for the exact standard ROS2 compressed-image type."""
import struct


class CanonicalCDRError(ValueError):
    """A source payload does not satisfy the standard CDR field layout."""


def compressed_image(schema, encoding, payload):
    """Recover bytes by typed lengths, never by searching for an image signature. Headers remain untrusted."""
    if encoding != 'cdr' or schema is None or schema.encoding != 'ros2msg' or schema.name != 'sensor_msgs/msg/CompressedImage':
        raise CanonicalCDRError('not the exact standard ROS2 compressed-image contract')
    if not isinstance(payload, bytes) or len(payload) < 16 or payload[:4] not in (b'\x00\x01\x00\x00', b'\x00\x00\x00\x00'):
        raise CanonicalCDRError('unsupported or incomplete CDR encapsulation')
    endian, offset = ('<' if payload[1] == 1 else '>'), 4

    def uint32():
        nonlocal offset
        offset = 4 + ((offset - 4 + 3) // 4) * 4
        if offset + 4 > len(payload):
            raise CanonicalCDRError('CDR scalar extends beyond its message')
        value = struct.unpack_from(endian + 'I', payload, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        size = uint32()
        if not 1 <= size <= 262144 or offset + size > len(payload) or payload[offset + size - 1] != 0:
            raise CanonicalCDRError('CDR string length or terminator is invalid')
        data = payload[offset:offset + size - 1]
        if b'\x00' in data:
            raise CanonicalCDRError('CDR string contains an interior terminator')
        offset += size
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError as error:
            raise CanonicalCDRError('CDR string is not UTF-8') from error

    seconds = struct.unpack_from(endian + 'i', payload, offset)[0]
    offset += 4
    nanos = uint32()
    if nanos >= 1_000_000_000:
        raise CanonicalCDRError('CDR nanoseconds are outside their standard range')
    frame_id, format_ = string(), string()
    size = uint32()
    if size == 0 or offset + size != len(payload):
        raise CanonicalCDRError('CDR image byte length does not consume exactly its message')
    # Deliberately omit timestamp/header keys consumed by capture-clock inference.
    return {'format': format_, 'data': payload[offset:], 'frame_id': frame_id,
            'recovered_untrusted_header': {'sec': seconds, 'nanosec': nanos}}
