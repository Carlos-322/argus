"""Locally requested sensor explanation contract, separate from model fields."""
VERSION = 3
# Versions whose detail is the shared explanation on video and in the findings list. A saved request keeps the exact
# rule text of the version it was sent with, so earlier versions remain reproducible.
SHARED_VERSIONS = (2, 3)
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
        return SHARED_RULE_V3 if version == 3 else SHARED_RULE_V2
    return SENSOR_RULE if kind == 'sensor' else INSPECTION_RULE


def schema(kind, version):
    if version == 3:
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
