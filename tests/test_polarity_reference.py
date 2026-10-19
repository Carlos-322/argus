"""What reaches an impression (board/polarity.py), measured over a grid of subjects, joins and predicates rather than one
wording at a time. Within its sentence, a correction retracts an impression only when its subject is the impression's
subject; beyond its sentence, nothing retracts it, and only an answer with nothing of its own reaches a question."""
from __future__ import annotations

import itertools
import re

import pytest

from board import polarity

FROZEN = re.compile(r'\bfrozen\b', re.I)
GLARE = re.compile(r'\bglare\b', re.I)

IMPRESSIONS = ['The exo camera appears frozen', 'The view looks frozen', 'It seems frozen']
SAME_SENTENCE = [', but ', ', but actually ']
LATER_SENTENCE = ['. ', '. On closer inspection, ', '; on review, ', '. Correction: ']
# a correction about the impression's own subject with nothing of its own to say: it retracts within the sentence
BARE_DENIALS = [('it', 'is not'), ('it', 'never was'), ('that', 'is not the case'), ('it', "isn't")]
# statements about another subject, or about a person, or limited in time: they never retract
OTHER_SUBJECTS = [('we', 'see the robot moving'), ('I', 'just checked the wrist view'), ('he', 'is just moving the robot'),
                  ('you', 'can see the arm move'), ('that wrist camera', 'is just fine'),
                  ('these timestamps', 'advance'), ('the wrist camera', 'is not frozen'),
                  ('the wrist camera', 'plays normally'), ('the operator', 'does not touch it'),
                  ('the arm', 'works normally'), ('this wrist view', 'is not frozen'), ('the clock', 'keeps ticking')]


def _text(impression, join, subject, predicate):
    later = f'{subject} {predicate}'
    if join.endswith(('. ', ': ')):
        later = later[0].upper() + later[1:]
    return f'{impression}{join}{later}.'


def _retracted(text):
    first = FROZEN.search(text)
    return any(a <= first.start() < b for a, b in polarity.denied_spans(text))


def test_every_cell_of_the_reference_grid():
    cells, wrong = 0, []
    for impression, join in itertools.product(IMPRESSIONS, SAME_SENTENCE + LATER_SENTENCE):
        within = join in SAME_SENTENCE
        for (subject, predicate), retracts in [*((d, within) for d in BARE_DENIALS),
                                               *((o, False) for o in OTHER_SUBJECTS)]:
            text = _text(impression, join, subject, predicate)
            cells += 1
            if _retracted(text) != retracts:
                wrong.append((text, retracts))
    assert cells == len(IMPRESSIONS) * len(SAME_SENTENCE + LATER_SENTENCE) * len(BARE_DENIALS + OTHER_SUBJECTS)
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


@pytest.mark.parametrize('later', [
    'I am withdrawing the glare flag.', 'I am not withdrawing the glare flag.', 'Should I withdraw the glare flag?',
    'If we withdraw the glare flag, the report will be wrong.', 'The glare flag was withdrawn.', 'Scratch that.',
    'I am withdrawing the wrist camera glare flag.', 'Nothing is clipped. Withdrawing the glare finding.'])
def test_a_withdrawal_in_a_later_sentence_retracts_nothing(later):
    text = f'Glare on the plate at 2 s. {later}'
    assert polarity.asserted(GLARE, text), text


@pytest.mark.parametrize('text', ['Suspected glare on the plate, retracted.', 'The glare I flagged is withdrawn.',
                                  'Retracting the glare flag.', 'I am withdrawing the glare flag.'])
def test_a_withdrawal_in_its_own_clause_retracts_what_it_withdraws(text):
    assert not polarity.asserted(GLARE, text), text
