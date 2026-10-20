"""Locally requested sensor explanation contract, separate from model fields."""
VERSION = 5
# Versions whose detail is the shared explanation on video and in the findings list. A saved request keeps the exact
# rule text of the version it was sent with, so earlier versions remain reproducible.
SHARED_VERSIONS = (2, 3, 4, 5)
SHARED_RULE_V2 = (
    'Use detail as the shared explanation in the findings list and on video: one or two short sentences, no more than 45 words. '
    'Headline and detail must together preserve every substantive qualitative conclusion in claim, including each comparison or temporal relationship. '
    'Do not leave the added inference only in claim or observation. Include only the uncertainty essential to read it correctly. '
    'Headline and detail appear together, so avoid repeating the headline. Put supporting measurement lists, component IDs and detailed technical limitations in observation and alternative. A compact task-relevant quantity may remain in detail when it explains the finding. Use plain hyphens rather than typographic dashes.')
SHARED_RULE_V3 = (
    'Headline and detail appear together on the video while the finding plays, for someone watching the clip. '
    'The headline says what changed in a few plain words. The detail is one short sentence, under 25 words, giving the comparison '
    'that makes it worth knowing: what changes relative to another part, the start, an earlier phase or the visible motion. '
    'Name each part by the name the source declares for it, such as a declared region, and the measured quantity in everyday words; '
    'on the card these names need no qualification. Components inside a named part are parts of it, such as one part of that region, '
    'not cells, patches or sensors. Describe a change in the direction the recording declares as a stronger response, '
    'for example pressing harder or easing for a pressure response, never as raw readings rising or falling. '
    'Keep component IDs, values, method and qualifications about mapping, convention, calibration or reference levels out of these two '
    'fields; they belong in observation, claim and alternative, which the viewer shows on request. '
    'When the interpretation is uncertain, one plain word such as likely is enough. '
    'Use plain hyphens rather than typographic dashes.')
# Version 4 keeps version 3's card copy and resolves two conflicts with the older sensor rules: a recording that
# declares no stronger direction is described by what rose or fell, and measurement limits and layout dependence
# stay in observation, claim and alternative instead of on the card (V4_LIMITS replaces the older sentences).
SHARED_RULE_V4 = SHARED_RULE_V3.replace(
    'Describe a change in the direction the recording declares as a stronger response, '
    'for example pressing harder or easing for a pressure response, never as raw readings rising or falling. ',
    'When the recording declares which direction is a stronger response, describe a change that way, '
    'for example pressing harder or easing for a pressure response, never as raw readings rising or falling; '
    'when it declares none, say what rose or fell. ')
assert SHARED_RULE_V4 != SHARED_RULE_V3
# Version 5 keeps version 4 and replaces how the headline and detail are worded. "A few plain words" read as fewer
# facts, and the model compressed each finding into a coined phrase ("thumb pressure returns locally") that a reader
# had to decode. PLAIN_ENGLISH asks for what a person would say aloud, with every specific kept; the card prompt
# (label/sensor_cards.py) uses the same sentence, so the annotation and the cards read in one voice.
PLAIN_ENGLISH = (
    'Write the headline and detail in plain English, the way a person watching would say it aloud, and keep every '
    'specific the finding supports: which part, how many places on it, which way each one went, what it is compared '
    'with and when in the action. Plain means ordinary words in whole sentences, never fewer facts. Never compress a '
    'pattern into an abstract word such as concentrates, shifts, diverges, uneven, locally, returns or rebuilds; say '
    'the thing it stands for, for example "two spots on the thumb press harder again while a third stays the same" '
    'rather than "thumb pressure returns locally". A component inside a named part is a spot on that part, counted '
    'when there are several. ')
SHARED_RULE_V5 = SHARED_RULE_V4.replace(
    'The headline says what changed in a few plain words. ',
    PLAIN_ENGLISH + 'The headline is one short sentence of at most eight words; when eight words cannot hold the '
    'specifics, the headline keeps the part and the change, and the detail carries the comparison. ').replace(
    'Components inside a named part are parts of it, such as one part of that region, not cells, patches or sensors. ',
    'Never call them cells, patches or sensors. ')
assert SHARED_RULE_V5.count(PLAIN_ENGLISH) == 1 and 'one part of that region' not in SHARED_RULE_V5
V4_LIMITS = ('Keep measurement limits and any dependence on a candidate layout or response convention explicit in '
             'observation, claim and alternative; the headline and detail follow the copy rule above. ')
LEGACY_LIMITS_LEAD = 'Apply measurement limits to the headline and video overlay, not only to the alternative explanation. '
LEGACY_LIMITS_MAPPING = ('Keep the candidate mapping dependency explicit in the interpretation and retain essential '
                         'uncertainty in the headline or overlay. ')
SENSOR_RULE = (
    'Also provide overlay_detail: one short sentence explaining the added information and any uncertainty needed to read the headline correctly. '
    'Headline and overlay_detail appear together on video, so avoid redundant wording and keep them readable at a glance. '
    'Put measurements and technical limitations in observation and alternative; retain essential uncertainty on video. '
    'Preserve the concrete added information across headline and overlay_detail, including any supported comparison, temporal pattern or contrast with recorded state.')
INSPECTION_RULE = (
    'Supply overlay_detail as one short sentence explaining what this adds and essential uncertainty. '
    'These two fields appear together on video; avoid repeating the headline or filling the card with technical measurements. '
    'Preserve the concrete added information across headline and overlay_detail, including any supported comparison, temporal pattern or contrast with recorded state.')


def shared(version):
    return type(version) is int and version in SHARED_VERSIONS


def rule(kind, version):
    if shared(version):
        return {2: SHARED_RULE_V2, 3: SHARED_RULE_V3, 4: SHARED_RULE_V4, 5: SHARED_RULE_V5}[version]
    return SENSOR_RULE if kind == 'sensor' else INSPECTION_RULE


def limits_lead(version):
    """Where measurement limits go: the card from version 4 on keeps them off, earlier prompts kept their text."""
    return '' if version in (4, 5) else LEGACY_LIMITS_LEAD


def limits_mapping(version):
    return V4_LIMITS if version in (4, 5) else LEGACY_LIMITS_MAPPING


def schema(kind, version):
    if version in (3, 4, 5):
        return ('"detail": "one plain sentence for the video card, under 25 words",'
                if kind == 'sensor' else
                '"detail":"one plain sentence for the video card, under 25 words",')
    if shared(version):
        return ('"detail": "shared concise on-video and list explanation; one or two short sentences, at most 45 words",'
                if kind == 'sensor' else
                '"detail":"shared concise on-video and list explanation; one or two short sentences, at most 45 words",')
    return ('"detail": "what the sensor adds", "overlay_detail": "short explanation for the video overlay",'
            if kind == 'sensor' else
            '"detail":"what this adds","overlay_detail":"short explanation for the video overlay",')
