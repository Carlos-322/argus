"""A later statement retracts or answers an earlier one only when it is tied to it by identity (board/polarity.py), measured
over a grid of subjects, joins and predicates rather than one wording at a time: every cell must read as the rule says,
so a new wording of an old shape cannot slip through untested."""
from __future__ import annotations

import itertools
import re

import pytest

from board import polarity

FROZEN = re.compile(r'\bfrozen\b', re.I)
GLARE = re.compile(r'\bglare\b', re.I)

IMPRESSIONS = ['The exo camera appears frozen', 'The view looks frozen', 'It seems frozen']
# how a later statement follows the impression
JOINS = [', but ', '. ', '. On closer inspection, ', '; on review, ']
# (subject, predicate, retracts): retracts is whether the statement is about the impression's subject and denies it
SAME_SUBJECT = [('it', 'is not frozen'), ('it', 'is not'), ('it', 'never was'), ('that', 'is not the case'),
                ('that', 'was wrong')]
OTHER_SUBJECT = [('we', 'actually see the robot moving'), ('I', 'just checked the wrist view'),
                 ('he', 'is just moving the robot'), ('you', 'can see the arm move'),
                 ('that wrist camera', 'is just fine'), ('these timestamps', 'actually advance'),
                 ('the wrist camera', 'is not frozen'), ('the operator', 'does not touch it'),
                 ('the arm', 'works normally'), ('this wrist view', 'is not frozen'),
                 # the same subject, but the denial is limited in time, so it limits the impression and denies nothing
                 ('it', 'is not frozen after 9 s'), ('it', 'is not frozen for the first 2 s')]


def _cells():
    for impression, join in itertools.product(IMPRESSIONS, JOINS):
        noun = impression.rsplit(' ', 2)[0]
        for subject, predicate in SAME_SUBJECT + [(noun.lower(), 'is not frozen')] * (noun != 'It'):
            yield impression, join, subject, predicate, True
        for subject, predicate in OTHER_SUBJECT:
            yield impression, join, subject, predicate, False


def _text(impression, join, subject, predicate):
    later = f'{subject} {predicate}'
    if join.endswith('. '):
        later = later[0].upper() + later[1:]
    return f'{impression}{join}{later}.'


def test_every_cell_of_the_reference_grid_reads_as_identity_decides():
    wrong = []
    cells = list(_cells())
    for impression, join, subject, predicate, retracts in cells:
        text = _text(impression, join, subject, predicate)
        # the impression's own "frozen" is the first match; a retraction leaves no asserted "frozen" before the
        # later statement
        first = FROZEN.search(text)
        spans = polarity.denied_spans(text)
        retracted = any(a <= first.start() < b for a, b in spans)
        if retracted != retracts:
            wrong.append((text, retracts))
    assert len(cells) > 150
    assert not wrong, wrong


@pytest.mark.parametrize('question', ['Frozen?', 'Is the view frozen?', 'Did the feed freeze, frozen?'])
@pytest.mark.parametrize('answer,asserted', [
    ('Yes, at 4 s.', True), ('Yes.', True), ('Briefly, yes, for 2 s.', True), ('It is, from 4 s.', True),
    ('It did, twice.', True), ('Definitely, at 0:58.', True),
    ('No.', False), ('It is not.', False), ('No question, the arm moves.', False), ('The arm moves on.', False),
    ('I think so.', False), ('', False)])
def test_a_question_is_asserted_only_by_an_answer_with_nothing_of_its_own(question, answer, asserted):
    text = f'{question} {answer}'.strip()
    assert polarity.asserted(FROZEN, text) is asserted, text


@pytest.mark.parametrize('text,asserted', [
    ('I initially suspected glare. I am withdrawing the glare flag.', False),
    ('I initially suspected glare and I am withdrawing the glare flag.', False),
    ('Glare on the plate at 2 s. Nothing is clipped. Withdrawing the glare finding.', False),
    ('Glare on the plate at 2 s. I am withdrawing the occlusion flag.', True),
    ('Glare on the plate at 2 s. The occlusion flag was withdrawn.', True),
    ('Glare on the plate at 2 s. The glare flag was withdrawn.', False),
    ('Glare on the plate at 2 s. Scratch that.', False),
    ('Glare on the plate at 2 s. The frozen frames are real. Scratch that.', True),
])
def test_a_withdrawal_reaches_only_what_it_names(text, asserted):
    assert polarity.asserted(GLARE, text) is asserted, text
