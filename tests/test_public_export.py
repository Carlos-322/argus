"""Privacy belongs at every JSON boundary, after source/freshness interpretation."""
import copy
import json

import pytest

from board.public_export import project, public_build_manifest


def test_nested_access_is_private_without_changing_source_or_annotation():
    label = {
        'publisher_labels': {'text': 'Open /data/example, then type api_key.',
                             'path': 'left to right'},
        'dataset_source': {'url': 'https://example.org/data?version=2', 'publisher': 'Example'},
        'data_dictionary': {'fields': [{'source_path': '/srv/uploads/episode/signals.npz',
                                       'source': 'signals.npz#force', 'shape': [3, 2]}]},
        'evidence_inspection': {'parts': [{'source_location': 'pieces/episode__p00',
                                          'record': {'source_current': True},
                                          'descriptor': {'url': 'https://example.org/o?X-Amz-Signature=secret',
                                                         'unit': 'mm'}}]},
        'sensor_evidence': {'sensors': [{'source': {'path': '/srv/source.h5',
                                                  'key': '/observations/force',
                                                  'url': 'https://example.org/source'}}]},
        'nested': [{'_run': {'run_id': 'private'}, 'credentials': {'password': 'private'},
                    'access_descriptor_digest': 'a' * 64, 'source_proof': [{'sha256': 'b' * 64}]}],
        '_compare': {'run_id': 'private', 'code': 'private', 'name': 'Other model'},
    }
    before = copy.deepcopy(label)
    public = project(label)
    assert public['publisher_labels'] == label['publisher_labels']
    assert public['dataset_source'] == label['dataset_source']
    assert public['data_dictionary']['fields'] == [{'source': 'signals.npz#force', 'shape': [3, 2]}]
    assert public['evidence_inspection']['parts'] == [
        {'record': {'source_current': True}, 'descriptor': {'unit': 'mm'}}]
    assert public['sensor_evidence']['sensors'][0]['source'] == {
        'key': '/observations/force', 'url': 'https://example.org/source'}
    assert public['nested'] == [{'access_descriptor_digest': 'a' * 64,
                                'source_proof': [{'sha256': 'b' * 64}]}]
    assert public['_compare'] == {'name': 'Other model'}
    assert label == before
    public['publisher_labels']['text'] = 'changed'
    assert label == before
    assert project(project(before)) == project(before)


@pytest.mark.parametrize('location', [
    '/mnt/private/source.h5', r'C:\Users\owner\source.h5', r'\\server\share\source.h5',
    '../uploads/source.h5', '~/uploads/source.h5', 'file:///srv/source.h5',
    's3://private-bucket/source.h5', 'gs://private-bucket/source.h5',
    'file:/srv/private/source.h5',
])
def test_typed_locations_are_removed(location):
    assert project({'provenance': [{'source_path': location}], 'source': location}) == {
        'provenance': [{}]}


@pytest.mark.parametrize('url', [
    'https://example.org/object?X-Amz-Signature=secret',
    'https://example.org/object?x-goog-credential=secret',
    'https://example.org/object?sig=secret&sv=2025',
    'https://example.org/object?access_token=secret',
    'https://example.org/object#token=secret',
    'https://user:secret@example.org/object',
    'http://127.0.0.1/object', 'http://10.0.0.1/object',
    'http://[::1]/object', 'http://storage.internal/object',
])
def test_unsafe_urls_in_typed_fields_and_arrays(url):
    public = project({'source_url': url, 'descriptor': {'urls': [url, 'https://example.org/paper']}})
    assert public == {'descriptor': {'urls': [None, 'https://example.org/paper']}}


def test_preserves_array_positions_and_schema_values():
    data = {'values': [None, 0, False, [1, 2]], 'source': {'context_path': '/signals/0',
            'key': '/hdf/channel', 'file': 'data/chunk-000.parquet'},
            'provenance': 'human', 'access_kind': 'depth', 'source_current': False}
    assert project(data) == data
    assert project({'url': ['https://example.org/a?token=secret', 'https://example.org/b']}) == {
        'url': [None, 'https://example.org/b']}


def test_secret_key_spelling_does_not_bypass_nested_projection():
    assert project({'descriptor': [{'apiKey': 'secret', 'Authorization': 'Bearer secret',
                                   'access-token': 'secret', 'Cookie': 'secret', 'secret_key': 'secret',
                                   'shape': [2], 'dtype': 'float32'}]}) == {
        'descriptor': [{'shape': [2], 'dtype': 'float32'}]}


def test_standalone_access_url_is_typed_even_under_an_extension_field():
    assert project({'extension': ['https://example.org/object?token=secret'],
                    'description': 'Read https://example.org/paper for the task.'}) == {
        'extension': [None], 'description': 'Read https://example.org/paper for the task.'}


def test_public_build_manifest_contains_counts_and_no_operational_locations():
    build = {'build_id': 'demo', 'built_at': '2026-10-11', 'code': 'a' * 40,
             'qa': '/srv/private/qa', 'clips': '/srv/private/clips',
             'enc': 'h264', 'public_base': 'https://example.org/board/', 'episodes': 3,
             'datasets': {'example': 3}, 'compare': {'dir': '/srv/private/compare', 'files': 2},
             'hands': {'dir': '/srv/private/hands', 'files': 1,
                       'skipped': {'ep.json': 'cannot read /srv/private/keypoints.npz'}},
             'sensors': {'dir': '/srv/private/sensors', 'files': 3},
             'keypoints': {'dir': '/srv/private/keypoints', 'files': 3, 'bytes': 123},
             'media': {'videos': 3, 'videos_present': 2}, 'missing_media': ['v/example.mp4'],
             'future_private_field': '/srv/private/future'}
    before = copy.deepcopy(build)
    public = public_build_manifest(build)
    assert public['compare'] == {'files': 2}
    assert public['hands'] == {'files': 1, 'skipped': {'ep.json': 'unavailable'}}
    assert public['keypoints'] == {'files': 3, 'bytes': 123}
    assert public['media'] == build['media']
    assert public['missing_media'] == build['missing_media']
    assert public['code'] == build['code']
    assert '/srv/private' not in json.dumps(public)
    assert 'future_private_field' not in public and 'qa' not in public and 'clips' not in public
    assert build == before


def test_flattened_source_note_owners_are_private_but_native_fields_and_prose_survive():
    from board.build import uploader_groups
    private = '/srv/private-upload/metadata/info.json'
    data = '/srv/private-upload/data/chunk-000/episode.parquet'
    native = '/observations/right/force'
    prose = 'Open /data/example, then type api_key.'
    notes = {'recorded episode metadata': {private: {'source': private,
              'fields': {native: 12, 'instruction': prose, 'path': native,
                         'source': '/publisher/source_identifier',
                         'unknown_extension_url': '/publisher/native_identity',
                         'api_key': 'unusable-test-value',
                         'credentials': {'access_id': 'unusable-other-value'},
                         'token_count': 17, 'unknown_extension_token': 'native token identity'}}},
             'recorded annotation data source': data,
             'recorded annotation tables': {private: {'native_step_index': {'claims': {'0': [
                 {'row': 0, 'source': private, 'fields': {'path': native,
                  'source': '/publisher/source_identifier'}}]}}}}}
    groups = uploader_groups(notes, None, 10)
    before = copy.deepcopy(groups)
    # Existing saved boards have only flattened name/value entries.
    legacy = [{**g, 'items': [{k: v for k, v in item.items() if k != '_uploader_field_path'}
                            for item in g['items']]} for g in groups]
    for input_groups in (groups, legacy):
        shown = project({'uploader_notes': input_groups})['uploader_notes']
        assert private not in json.dumps(shown) and data not in json.dumps(shown)
        assert native in json.dumps(shown) and prose in json.dumps(shown)
        assert '/publisher/source_identifier' in json.dumps(shown)
        assert '/publisher/native_identity' in json.dumps(shown)
        assert 'unusable-test-value' not in json.dumps(shown)
        assert 'unusable-other-value' not in json.dumps(shown)
        assert 'native token identity' in json.dumps(shown)
        assert any('info.json' in row['name'] for group in shown for row in group['items'])
        assert project({'uploader_notes': shown}) == {'uploader_notes': shown}
    assert groups == before and notes['recorded episode metadata'][private]['fields'][native] == 12
    plain = project({'uploader_notes': uploader_groups(prose, None, 10)})
    assert plain['uploader_notes'][0]['items'] == [{'name': 'note', 'value': prose}]


def test_reader_metadata_limit_paths_use_typed_source_identity_without_rewriting_publisher_text(tmp_path, monkeypatch):
    path = '/srv/private-upload/data/chunk-000/episode.parquet'
    message = (path + ' and its recorded metadata exceed the prompt note limit; complete original rows '
               'and source claims remain in the episode context and uploader notes.')
    label = {'dataset_checks': {'reader_issues': [{'kind': 'metadata_limit', 'what': message,
             'family': 'metadata-unreadable'}]}, 'publisher_labels': {'text': message}}
    before = copy.deepcopy(label)
    shown = project(label)
    issue = shown['dataset_checks']['reader_issues'][0]
    assert path not in issue['what'] and 'episode.parquet' in issue['what']
    assert issue['kind'] == 'metadata_limit' and issue['family'] == 'metadata-unreadable'
    assert shown['publisher_labels']['text'] == message and label == before
    from prepare import formats
    context = {}
    for _ in range(2):
        formats.add_issue(context, 'metadata_limit', 'episode.parquet exceeds the note limit.', source=path)
    assert len(context['reader_issues']) == 1
    assert context['reader_issues'][0]['source'] == path
    formats.add_issue(context, 'metadata_limit', 'episode.parquet exceeds the note limit.',
                      source='/srv/other-upload/episode.parquet')
    assert len(context['reader_issues']) == 2
    public = project({'dataset_checks': context})
    assert path not in json.dumps(public)
    # Exercise the actual annotation reader twice, not just its issue constructor.
    import pandas as pd
    (tmp_path / 'meta').mkdir()
    data = tmp_path / 'episode.parquet'
    frame = pd.DataFrame({'timestamp': [0., 1., 2.], 'frame_index': [0, 1, 2],
                          'native_step': ['one', 'one', 'one']})
    frame.to_parquet(data)
    index = {'dir': tmp_path, 'fps': 1, 'metadata_read': set()}
    context = {'episode_index': 0, 'fps': 1, 'n_state_frames': 3}
    monkeypatch.setattr(formats, 'ANNOTATION_MAX_CHARS', 1)
    for _ in range(2):
        formats.lerobot_annotations(context, index, frame, data)
    issues = [issue for issue in context['reader_issues']
              if issue['kind'] == 'metadata_limit' and issue.get('source') == str(data)]
    assert len(issues) == 1 and str(tmp_path) not in issues[0]['what']
