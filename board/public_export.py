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
    'root', 'packed', 'src', 'location', 'sourceroot', 'sourcepath', 'sourcefile',
    'archivepath', 'archive', 'url', 'urls', 'uri', 'uris', 'href', 'link', 'links',
    'publicbase', 'publiccitation', 'citation',
})
_URL_QUERY_SECRETS = _SECRETS | {'signature', 'sig', 'keypairid', 'policy', 'googleaccessid'}
_DROP = object()


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


def _project(value, field: str = ''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = _name(key)
            if (key in PRIVATE_KEYS or name in _SECRETS or name in _PRIVATE_LOCATIONS
                    or name.endswith(('apikey', 'accesstoken', 'secretaccesskey'))
                    or (field == 'compare' and name in ('runid', 'code'))):
                continue
            shown = _project(item, name)
            if shown is not _DROP:
                result[key] = shown
        return result
    if isinstance(value, (list, tuple)):
        # Do not move sensor sample positions when a typed string is withheld.
        return [None if (shown := _project(item, field)) is _DROP else shown for item in value]
    if isinstance(value, str):
        # An entire URL is a typed value even in an extension field. Do not scan
        # or rewrite URL fragments embedded in publisher prose.
        stripped = value.strip()
        if (stripped.lower().startswith(('http://', 'https://'))
                and not any(char.isspace() for char in stripped)
                and _private_url(stripped)):
            return _DROP
        location = (field in _LOCATIONS or field.endswith(('url', 'uri', 'path', 'paths', 'urls', 'uris')))
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
