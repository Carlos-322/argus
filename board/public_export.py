"""Copy JSON for public transport after private source/freshness checks.

This is a typed privacy boundary, not a text scrubber or a publication-rights
decision. Annotation prose, native field names and public citations remain intact.
Keep original inputs and operational build receipts in private storage.
"""
from __future__ import annotations

import ipaddress
import re
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import parse_qsl, unquote, urlsplit


PRIVATE_KEYS = frozenset({
    '_run', '_replaced_label', '_supplements', '_verification', '_withheld_checks', '_carried_verdicts',
    '_uploader_field_path',
})
# These are access mechanisms, not source identities or source-proof digests.
_SECRETS = frozenset({
    'apikey', 'accesstoken', 'refreshtoken', 'idtoken', 'token', 'password', 'passwd',
    'secret', 'secretkey', 'clientsecret', 'secretaccesskey', 'accesskeyid', 'sessiontoken',
    'authorization', 'proxyauthorization', 'cookie', 'setcookie', 'credentials',
    'credential', 'privatekey', 'signedurl', 'presignedurl', 'accessurl',
    'requestheaders', 'authheaders',
})
# Source locations resolve private preparation trees before this projection runs.
_PRIVATE_LOCATIONS = frozenset({
    'sourcelocation', 'rawpath', 'localpath', 'retainedroot', 'retainedevidenceroot',
    'episodedir', 'exampledir', 'jobdir', 'cachedir',
})
_LOCATIONS = frozenset({
    'source', 'sources', 'path', 'paths', 'file', 'files', 'filename', 'directory', 'dir',
    'root', 'packed', 'src', 'location', 'sourceroot', 'sourcepath', 'sourcefile', 'sourcedirectory',
    'archivepath', 'archive', 'url', 'urls', 'uri', 'uris', 'href', 'link', 'links',
    'publicbase', 'publiccitation', 'citation',
})
_URL_QUERY_SECRETS = _SECRETS | {'signature', 'sig', 'keypairid', 'policy', 'googleaccessid'}
_DROP = object()

# These reader-owned note maps use source files as their keys. Keys inside the
# recorded fields remain native identifiers, including absolute HDF paths.
_NOTE_SOURCE_MAPS = frozenset({
    'recordedepisodemetadata', 'recordedannotationtables', 'outsidefiles',
    'unreadableannotationtablesources',
})
_NOTE_SOURCE_VALUES = frozenset({'recordedannotationdatasource'})
_ATTRIBUTE_LOCATIONS = frozenset({'sourcedirectory', 'sourcelocation'})
_METADATA_LIMIT_SUFFIX = (' and its recorded metadata exceed the prompt note limit; complete original rows '
                          'and source claims remain in the episode context and uploader notes.')


def _name(key: str) -> str:
    return re.sub('[^a-z0-9]', '', key.lower())


def _private_url(value: str) -> bool:
    """Recognize access URLs without trying DNS or fetching any source."""
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ('http', 'https'):
            return bool(parsed.scheme)
        host = (parsed.hostname or '').lower().rstrip('.')
        if not host or parsed.username is not None or parsed.password is not None:
            return True
        if '.' not in host or host.endswith(('.localhost', '.local', '.internal', '.lan')):
            return True
        try:
            if not ipaddress.ip_address(host).is_global:
                return True
        except ValueError:
            pass
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True) + parse_qsl(parsed.fragment, keep_blank_values=True):
            name = _name(key)
            if name in _URL_QUERY_SECRETS or name.startswith(('xamz', 'xgoog')):
                return True
    except ValueError:
        return True
    return False


def _private_location(value: str) -> bool:
    value = unquote(value.strip())
    if value.lower().startswith(('file:', 's3:', 'gs:', 'az:', 'abfs:', 'ssh:', 'sftp:', 'ftp:')):
        return True
    if '://' in value or value.startswith('//'):
        return _private_url(value if not value.startswith('//') else 'https:' + value)
    windows = PureWindowsPath(value)
    return (PurePosixPath(value).is_absolute() or bool(windows.drive)
            or value.startswith(('~/', '~\\', '\\'))
            or '..' in value.replace('\\', '/').split('/'))


def _source_name(value: str) -> str:
    """A source file's basename is useful identity; its access directory is private."""
    value = unquote(value.strip())
    if '://' in value or value.lower().startswith(('file:', 's3:', 'gs:', 'az:')):
        value = urlsplit(value).path
    return PureWindowsPath(value).name if '\\' in value else PurePosixPath(value).name


def _reader_attribute(key: str) -> str | None:
    """Decode the exact file/episode attribute names emitted by formats.h5_text."""
    for prefix in ('file attribute ', 'episode attribute '):
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def _location_field(field: str) -> bool:
    return field in _LOCATIONS or field.endswith(('url', 'uri', 'path', 'paths', 'urls', 'uris'))


def _uploader_item(item: dict) -> dict:
    """Project the reader's flattened display schema using its original field path."""
    result = dict(item)
    parts = item.get('_uploader_field_path')
    if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
        # Historical boards used this exact separator but retained no path array.
        parts = str(item.get('name') or '').split(' / ')
    shown = list(parts)
    for i, part in enumerate(parts):
        if any(_name(parent) == 'fields' for parent in parts[:i]):
            break
        if i and _name(parts[i - 1]) in _NOTE_SOURCE_MAPS and _private_location(part):
            shown[i] = _source_name(part) or 'source file'
    result['name'] = ' / '.join(shown) if shown else (item.get('name') or 'note')
    attributes = [_name(attribute) for i, part in enumerate(parts)
                  if not any(_name(parent) == 'fields' for parent in parts[:i])
                  and (attribute := _reader_attribute(part)) is not None]
    if any(_name(part) in _SECRETS for part in parts) or any(a in _SECRETS for a in attributes):
        result['value'] = 'withheld'
        return result
    value = item.get('value')
    terminal = _name(parts[-1]) if parts else ''
    # A reader source record owns its immediate source property. The fields
    # below that record are the publisher's native schema, not access metadata.
    native_fields = any(_name(part) == 'fields' for part in parts[:-1])
    owner_property = (len(parts) >= 3 and _name(parts[-3]) in _NOTE_SOURCE_MAPS
                      and terminal == 'source')
    table_claim = (terminal == 'source' and 'recordedannotationtables' in [_name(p) for p in parts]
                   and 'claims' in [_name(p) for p in parts[:-1]])
    source_value = not native_fields and (owner_property or table_claim or terminal in _NOTE_SOURCE_VALUES
                                         or any(a in _ATTRIBUTE_LOCATIONS for a in attributes)
                                         or any(a not in ('source', 'path') and _location_field(a) for a in attributes)
                                         or (bool(attributes) and _reader_attribute(parts[-1]) is None
                                             and _location_field(terminal)))
    if isinstance(value, str) and source_value and _private_location(value):
        result.pop('value', None)
        result['value'] = _source_name(value) or 'source file'
    return result


def _uploader_group(value: dict) -> dict:
    if not isinstance(value.get('items'), list) or value.get('kind') not in ('notes', 'row'):
        return value
    result = dict(value)
    result['items'] = [_uploader_item(item) if isinstance(item, dict) else item for item in value['items']]
    title = value.get('title')
    if isinstance(title, str) and title.startswith('Row of ') and _private_location(title[7:]):
        result['title'] = 'Row of ' + (_source_name(title[7:]) or 'source file')
    return result


def _reader_issue(value: dict) -> dict:
    """Only machine reader diagnostics receive source-location migration."""
    result = dict(value)
    what = value.get('what')
    source = value.get('source')
    if isinstance(what, str) and isinstance(source, str) and _private_location(source):
        result['what'] = what.replace(source, _source_name(source) or 'source file')
    elif (value.get('kind') == 'metadata_limit' and isinstance(what, str)
          and what.endswith(_METADATA_LIMIT_SUFFIX)):
        source = what[:-len(_METADATA_LIMIT_SUFFIX)]
        if _private_location(source):
            result['what'] = (_source_name(source) or 'Recorded annotation data') + _METADATA_LIMIT_SUFFIX
    return result


def _project(value, field: str = '', attribute_owner: str = '', native_fields: bool = False):
    if isinstance(value, dict):
        native_fields = native_fields or field == 'fields'
        if field == 'uploadernotes':
            value = _uploader_group(value)
        elif field == 'readerissues':
            value = _reader_issue(value)
        result = {}
        for key, item in value.items():
            name = _name(key)
            attribute = _reader_attribute(key) if not native_fields else None
            owner = _name(attribute) if attribute is not None else attribute_owner
            if (key in PRIVATE_KEYS or name in _SECRETS or name in _PRIVATE_LOCATIONS
                    or (attribute is not None and owner in _SECRETS)
                    or name.endswith(('apikey', 'accesstoken', 'secretaccesskey'))
                    or (field == 'compare' and name in ('runid', 'code'))):
                continue
            shown = _project(item, name, owner, native_fields)
            if shown is not _DROP:
                result[key] = shown
        return result
    if isinstance(value, (list, tuple)):
        # Do not move sensor sample positions when a typed string is withheld.
        return [None if (shown := _project(item, field, attribute_owner, native_fields)) is _DROP else shown for item in value]
    if isinstance(value, str):
        if attribute_owner in _ATTRIBUTE_LOCATIONS and _private_location(value):
            return _source_name(value) or 'source file'
        # An entire URL is a typed value even in an extension field. Do not scan
        # or rewrite URL fragments embedded in publisher prose.
        stripped = value.strip()
        if (stripped.lower().startswith(('http://', 'https://'))
                and not any(char.isspace() for char in stripped)
                and _private_url(stripped)):
            return _DROP
        # Direct source/path attributes can name HDF members; nested access fields retain protection.
        direct_attribute = attribute_owner in ('source', 'path') and field in (
            'fileattribute' + attribute_owner, 'episodeattribute' + attribute_owner)
        location = not direct_attribute and _location_field(field)
        # context_path is a JSON pointer; it is not a filesystem path.
        if field == 'contextpath':
            location = False
        if location and _private_location(value):
            return _DROP
    return value


def project(value):
    """Return independent public JSON; omit private fields, use null in arrays.

    Values must already be JSON-compatible. This function does not read, resolve,
    write or mutate inputs. Run evidence freshness and human overrides first.
    Native HDF keys and JSON pointers remain meaningful public identifiers.
    """
    result = _project(value)
    return None if result is _DROP else result


def public_build_manifest(build: dict) -> dict:
    """Publish build identity and availability, without operational directories.

    Unknown build fields fail closed. Detailed skip diagnostics remain in the
    original private receipt because exception messages can contain local paths.
    """
    result = {key: build[key] for key in (
        'build_id', 'built_at', 'code', 'enc', 'public_base', 'episodes', 'datasets',
        'media', 'missing_media',
    ) if key in build}
    for key in ('compare', 'hands', 'keypoints', 'sensors'):
        value = build.get(key)
        if isinstance(value, dict):
            result[key] = {k: value[k] for k in ('files', 'bytes') if k in value}
            if isinstance(value.get('skipped'), dict):
                result[key]['skipped'] = {name: 'unavailable' for name in value['skipped']}
        elif key in build:
            result[key] = None
    return project(result)
