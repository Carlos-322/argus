"""The cards a person watching a recording sees on the video, chosen and written from its bound sensor findings.

The analysis passes find every supported change and describe it in analysis terms. This one text-only call reads
those findings, chooses the few a watcher should see while they happen, and writes every finding in plain English: a
chosen finding gains a video_card, and every other one a list_copy, so the list reads as one voice. Every finding stays
in the list. An entry names its finding by the finding's own content (finding_key), never by position, so a rebind
that changes a finding drops its entry.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from label import sensor_copy, sensor_display
from label import sensor_evidence as se

VERSION = 1
MAX_TOKENS = 16000
# What the prompt asks for, and the most a reply may use before an entry is left out.
HEADLINE_WORDS, HEADLINE_LIMIT = 8, 10
DETAIL_WORDS, DETAIL_LIMIT = 25, 30

PROMPT = '''A person will watch this recording with short cards on the video, one card per sensor finding chosen for it. A card appears when its finding starts and stays on screen for at least two seconds. Every finding below stays available in a list beside the video, so the cards are only for what the watcher should see while it happens.

Choose the cards and write them. Choose so that, in playback order, the cards tell the watcher what the sensors add to the video: the clearest change at each moment, never two cards about the same change, never more than two cards on screen at once, and no more cards than the watcher can read in the time they are on screen. The sensors matter most where the video cannot show the change, such as a change out of view or hidden under the hand, or one part of a region changing differently from the rest. Between findings about the same moment, choose the most specific one the evidence supports. Leave a finding off the video only because another card covers its moment better or the watcher could not read it in time, never because cards should be few: a finding that adds something the video cannot show and competes with no other deserves its card, even when it is the only finding.

For each card write a headline of one short sentence, at most {headline} words, saying what changed and naming the part or parts it changed in, and a detail of one sentence under {detail} words giving the comparison that makes it worth knowing, such as which part changes relative to another part, the start or the visible motion. {plain}Name parts by the names the finding's rows use, written as ordinary words in a sentence (the thumb, the index finger, two spots on the thumb); a numbered label such as Region 1 is not a name, so say what is there, as the finding's claim describes it. Name the measured quantity in everyday words, for example pressure for a pressure sensor, and describe the change as the findings' claims interpret it, for example pressing harder or easing; where a sensor declares no stronger direction (stronger_response is null), say what rose or fell instead. Never use analysis words such as response, loading, signal, reference, component, estimate or configuration. Say only what the chosen finding's observation and claim support. When the finding is uncertain, the single word likely is enough. Use plain hyphens rather than typographic dashes.

Then write the list entry for every finding you did not choose, the same way and in the same plain English, so the list reads as one voice: a headline of at most {headline} words and one sentence under {detail} words.

Return only JSON: {{"cards": [{{"finding": <index>, "headline": "...", "detail": "..."}}], "list": [{{"finding": <index>, "headline": "...", "detail": "..."}}]}}, each in playback order. An empty cards array is valid.
'''.format(headline=HEADLINE_WORDS, detail=DETAIL_WORDS, plain=sensor_copy.PLAIN_ENGLISH)


def _packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def finding_key(finding):
    """A finding's identity: its annotation content, without review decisions or presentation."""
    fields = se.finding_fields(finding)
    fields.pop('review_status', None)
    fields.pop('overlay_detail', None)
    return hashlib.sha256(_packed(fields)).hexdigest()


def _digest(request):
    return hashlib.sha256(_packed({key: request[key] for key in ('version', 'config', 'finding_keys', 'content_sha256')})).hexdigest()


def build(doc, *, task=None, config):
    """The card request for a bound sensor document, or None when it has no timed finding."""
    if not isinstance(doc, dict):
        return None
    shown = sensor_display.attach(copy.deepcopy(doc))
    # Only what the viewer can show: a finding that adds nothing beyond the video never reaches the video.
    findings = sorted((f for f in shown.get('findings') or [] if isinstance(f, dict) and f.get('adds_beyond_video') is True
                       and f.get('review_status') != 'rejected'), key=lambda f: (f['start_s'], f['end_s']))
    if not findings:
        return None
    sensors = [{'name': s.get('name'), 'quantity': s.get('quantity'), 'stronger_response': s.get('response_direction')}
               for s in shown.get('sensors') or [] if isinstance(s, dict) and not str(s.get('id', '')).startswith('mechanics:')
               and 'access_field_id' not in s]
    rows = [{'index': i, 'start_s': round(f['start_s'], 2), 'end_s': round(f['end_s'], 2),
             'headline': f.get('headline'), 'detail': f.get('detail'), 'observation': f.get('observation'),
             'claim': f.get('claim'), 'confidence': f.get('confidence'),
             'rows': [r.get('label') or r.get('series_id') for r in (f.get('display') or {}).get('rows') or []]}
            for i, f in enumerate(findings)]
    text = PROMPT + '\nRECORDING\n' + json.dumps({'task': task, 'sensors': sensors, 'findings': rows}, ensure_ascii=True)
    content = [{'type': 'text', 'text': text}]
    settings = {key: copy.deepcopy(config[key]) for key in ('model', 'reasoning') if key in config}
    request = {'version': VERSION, 'config': {**settings, 'max_tokens': MAX_TOKENS},
               'finding_keys': [finding_key(f) for f in findings],
               'content_sha256': hashlib.sha256(_packed(content)).hexdigest()}
    return {**request, 'content': content, 'digest': _digest(request)}


def _words(text, limit):
    return isinstance(text, str) and text.strip() and len(text.split()) <= limit and not re.search('[\u2013\u2014]', text)


def _entries(rows, count, taken):
    """The well-formed entries of one reply array: each names a listed finding once, in short plain copy."""
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        index = row.get('finding')
        if type(index) is not int or index < 0 or (count is not None and index >= count) or index in taken:
            continue
        if not _words(row.get('headline'), HEADLINE_LIMIT) or not _words(row.get('detail'), DETAIL_LIMIT):
            continue
        taken.add(index)
        out.append({'finding': index, 'headline': row['headline'].strip(), 'detail': row['detail'].strip()})
    return out


def parse_response(text, count=None):
    """The cards and the list entries in a complete reply. An entry that names no listed finding, names one already
    written, or whose copy is not a short plain sentence, is left out; the rest stand. A card wins over a list entry."""
    if not isinstance(text, str):
        raise ValueError('Sensor card reply has no retained response text')
    fenced = re.fullmatch(r'\s*```(?:json)?\s*(.*?)\s*```\s*', text, re.S)
    reply = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(reply, dict) or not isinstance(reply.get('cards'), list):
        raise ValueError('Sensor card reply must return a cards array')
    taken = set()
    cards = _entries(reply['cards'], count, taken)
    return {'cards': cards, 'list': _entries(reply.get('list'), count, taken)}


def apply(doc, request, receipt):
    """Give each chosen finding its video card and every other written finding its list copy. Only the exact request
    may authorize them, and an entry whose finding is no longer in the document is dropped. Returns how many cards
    were placed."""
    if not isinstance(doc, dict):
        return 0
    for finding in doc.get('findings') or []:
        if isinstance(finding, dict):
            finding.pop('video_card', None)
            finding.pop('list_copy', None)
    doc.pop('video_cards', None)
    if request.get('digest') != _digest(request) or receipt.get('digest') != request['digest']:
        raise ValueError('Sensor card request proof changed')
    if receipt.get('state') != 'received' or receipt.get('finish_reason') != 'stop':
        raise ValueError('Sensor card request has no complete received response')
    keys = request['finding_keys']
    by_key = {}
    for finding in doc.get('findings') or []:
        if isinstance(finding, dict):
            by_key.setdefault(finding_key(finding), finding)
    reply = parse_response(receipt.get('response_text'), len(keys))
    placed = 0
    for field, entries in (('video_card', reply['cards']), ('list_copy', reply['list'])):
        for entry in entries:
            finding = by_key.get(keys[entry['finding']])
            if finding is None:
                continue
            finding[field] = {'version': VERSION, 'headline': entry['headline'], 'detail': entry['detail']}
            placed += field == 'video_card'
    doc['video_cards'] = {'version': VERSION, 'request_digest': request['digest'], 'cards': placed,
                          'model': receipt.get('model'), 'generation_id': receipt.get('generation_id')}
    return placed
