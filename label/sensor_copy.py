"""Locally requested sensor explanation contract, separate from model fields."""
VERSION = 2
SHARED_RULE = (
    'Use detail as the shared explanation in the findings list and on video: one or two short sentences, no more than 45 words. '
    'Headline and detail must together preserve every substantive qualitative conclusion in claim, including each comparison or temporal relationship. '
    'Do not leave the added inference only in claim or observation. Include only the uncertainty essential to read it correctly. '
    'Headline and detail appear together, so avoid repeating the headline. Put supporting measurement lists, component IDs and detailed technical limitations in observation and alternative. A compact task-relevant quantity may remain in detail when it explains the finding. Use plain hyphens rather than typographic dashes.')
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
    return type(version) is int and version == VERSION


def rule(kind, version):
    return SHARED_RULE if shared(version) else SENSOR_RULE if kind == 'sensor' else INSPECTION_RULE


def schema(kind, version):
    if shared(version):
        return ('"detail": "shared concise on-video and list explanation; one or two short sentences, at most 45 words",'
                if kind == 'sensor' else
                '"detail":"shared concise on-video and list explanation; one or two short sentences, at most 45 words",')
    return ('"detail": "what the sensor adds", "overlay_detail": "short explanation for the video overlay",'
            if kind == 'sensor' else
            '"detail":"what this adds","overlay_detail":"short explanation for the video overlay",')
