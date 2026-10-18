"""Which words of a model's free text are asserted, and which are denied, retracted or only asked about.

A problem family is claimed by a word in an issue's text (board/families.py), and a gate case can ask whether a label
names a word (gate/score.py). Both need to know whether the sentence says the thing happened. Earlier rounds answered
that with one regex per wording, and each new regex broke another wording. This module answers it once, from the
grammar English uses to deny something, so a new wording of an old form needs no new rule:

  - a negative determiner denies the noun phrase after it, and every bare noun coordinated with it: "no glare",
    "without occlusion, rotation or freezing", "free of glare", "absence of glare", "rather than glare". When that
    phrase is the subject ("no frame is frozen", "nobody can see"), the predicate is denied too, and an elliptical
    "none" denies what the clause looked for ("occlusion was looked for and none observed");
  - a negated predicate denies its own words ("is not frozen", "never covers the lens"), and also its subject when
    the predicate says only that something exists ("glare is not present", "occlusion never happens", "dropout is
    never an issue"). Any other negated predicate leaves its subject asserted: "glare never clears" and "the
    occlusion is not removed until 9 s" say the problem lasted. A negated remedy asserts what it would have ended
    ("never turned back", "does not fix the upside-down camera");
  - two negations cancel ("not without glare");
  - a question asserts nothing unless the next sentence answers it ("Frozen? Yes, at 4 s"); a label is denied by a
    bare negative ("Occlusion: none"), and a negative label denies the bare list it introduces ("Absent: glare,
    occlusion");
  - an impression that is then corrected is retracted ("looks like glare, actually a white tablecloth", "seems frozen
    but it isn't", "what looks like glare is a white plate", "resembles a black frame, but it is a shadow"); a hedge
    on its own still asserts ("appears to be frozen");
  - a condition ("if glare hits the plate") asserts nothing.

Scope never crosses a clause: a sentence boundary, a subordinating or contrasting conjunction ("because", "while",
"but", "then", "which"), or a comma or "and" that starts a new subject. So an unrelated negation ("the grasp is not
visible because the view is frozen") never reaches the problem.

The word lists below are closed grammatical classes (negators, auxiliaries, determiners, prepositions,
conjunctions) and a few small semantic classes: predicates that only state existence, predicates that end or undo a
state, hedges, and words that introduce a correction. None is a wording of any one problem.
tests/test_family_polarity.py measures the module on a corpus.

    denied_spans(text)            -> [(start, end)] character spans the text does not assert
    asserted(pattern, text)       -> whether some match of the compiled pattern is asserted
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+(?:[-'][^\W\d_]+)*|\n|[^\w\s]")
_SENTENCE_END = {'.', '!', ';', '\n', '?'}

AUX = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', "'s", 'has', 'have', 'had', 'does', 'do', 'did',
       'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', "it's", "that's", "there's"}
# the finite forms of "be", which link a subject to what it is
COPULA = {'is', 'are', 'was', 'were', "'s", "it's", "that's"}
# auxiliaries with a negation fused on, written with or without the apostrophe
_NT = {'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt', 'werent', 'cant', 'couldnt', 'wont', 'wouldnt', 'hasnt',
       'havent', 'hadnt', 'shouldnt', 'aint', 'mustnt', 'neednt'}
PREDICATE_NEGATORS = {'not', 'never', 'cannot'}
# a negative determiner or quantifier: denies the noun phrase that follows it
DETERMINER_NEGATORS = {'no', 'without', 'neither', 'nor', 'zero', 'lacks', 'lacking', 'nothing', 'nobody', 'none'}
# two-word negative determiners
DETERMINER_PAIRS = {('free', 'of'), ('free', 'from'), ('clear', 'of'), ('absence', 'of'), ('lack', 'of'),
                    ('rather', 'than'), ('instead', 'of'), ('none', 'of'), ('no', 'one'), ('ruled', 'out'),
                    ('rule', 'out'), ('rules', 'out'), ('ruling', 'out'), ('mistaken', 'for'),
                    ('confused', 'with')}
# negative subjects: when the denied phrase is followed by its verb, the verb's predicate is denied as well
QUANTIFIERS = {'no', 'neither', 'nor', 'zero', 'nothing', 'nobody', 'none'}
# a negator standing after its subject
POST_NEGATORS = {'absent', 'nowhere'}
# predicates that say only that something exists or was seen; negating one denies its subject
EXISTENCE = {'present', 'visible', 'seen', 'observed', 'observable', 'detected', 'detectable', 'noticed',
             'noticeable', 'found', 'apparent', 'evident', 'shown', 'there', 'recorded', 'appear', 'appears',
             'appeared', 'occur', 'occurs', 'occurred', 'happen', 'happens', 'happened', 'exist', 'exists',
             'existed', 'arise', 'arises', 'arose', 'show', 'shows', 'showed'}
EXISTENCE_NOUNS = {'issue', 'issues', 'problem', 'problems', 'concern', 'concerns', 'factor', 'case'}
# predicates that end or undo a state: negating one says the state lasted
REMEDY = {'stop', 'stops', 'stopped', 'end', 'ends', 'ended', 'cease', 'ceases', 'ceased', 'clear', 'clears',
          'cleared', 'recover', 'recovers', 'recovered', 'resolve', 'resolves', 'resolved', 'disappear', 'disappears',
          'disappeared', 'fade', 'fades', 'faded', 'subside', 'subsides', 'subsided', 'lift', 'lifts', 'fix', 'fixes',
          'fixed', 'correct', 'corrects', 'corrected', 'repair', 'repairs', 'repaired', 'restore', 'restores',
          'restored', 'straighten', 'straightens', 'straightened', 'reposition', 'repositions', 'repositioned',
          'undo', 'undoes', 'undone', 'address', 'addresses', 'addressed'}
# verbs of motion that, with "back" or "away", undo a state ("turned back", "put back", "goes away")
_RETURNING = {'turn', 'turns', 'turned', 'put', 'puts', 'move', 'moves', 'moved', 'bring', 'brings', 'brought',
              'come', 'comes', 'came', 'set', 'sets', 'go', 'goes', 'went', 'gone', 'get', 'gets', 'got', 'rotate',
              'rotates', 'rotated', 'swing', 'swings', 'swung', 'placed', 'returned'}
_GOING = {'go', 'goes', 'went', 'gone'}
# words between a negator and the predicate it negates
SKIP = AUX | {'even', 'ever', 'always', 'really', 'actually', 'once', 'yet', 'fully', 'completely', 'entirely',
              'quite', 'truly', 'clearly', 'obviously', 'visibly', 'noticeably', 'very', 'too', 'particularly',
              'significantly', 'seriously', 'much', 'all', 'at', 'to', 'get', 'gets', 'got', 'become', 'becomes',
              'became', 'any', 'anymore'}
# verbs of seeming that take an infinitive: "does not appear to be frozen" negates "frozen"
_RAISING = {'appear', 'appears', 'seem', 'seems', 'look', 'looks', 'happen', 'happens', 'tend', 'tends'}
# words that start a new clause: subordinating and contrasting conjunctions and relative pronouns
BOUNDARY = {'but', 'however', 'whereas', 'because', 'since', 'while', 'whilst', 'although', 'though', 'when',
            'whenever', 'so', 'then', 'which', 'who', 'where', 'except', 'unless', 'until', 'after', 'before', 'if',
            'whether', 'thus', 'therefore', 'hence', 'otherwise', 'meanwhile', 'actually', 'yet', 'once',
            'nevertheless', 'nonetheless'}
# words that open a clause contrasting with the one before it
CONTRAST = {'but', 'however', 'though', 'although', 'yet', 'whereas', 'actually', 'in', 'instead', 'rather'}
# clauses that state a condition, not a fact
CONDITIONAL = {'if', 'unless', 'whether'}
# words that open a noun phrase with its own subject, so a comma or "and" before them starts a new clause
DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'its', 'it', "it's", 'their', 'they', 'his',
               'her', 'he', 'she', 'we', 'i', 'you', 'our', 'my', 'there', "there's", 'someone', 'something',
               'each', 'every'}
_ARTICLES = {'the', 'a', 'an', 'its', 'their', 'his', 'her', 'this', 'that', 'some'}
# prepositions that end a negated predicate: what follows is an adjunct of it, not part of it
PREDICATE_END = {'in', 'on', 'at', 'from', 'to', 'for', 'with', 'during', 'until', 'after', 'before', 'since',
                 'throughout', 'across', 'near', 'around', 'within', 'beyond', 'behind', 'due', 'because',
                 'despite', 'except', 'between', 'into', 'onto', 'toward', 'towards', 'here', 'anywhere',
                 'whatsoever', 'again', 'either', 'over', 'under', 'through', 'as', 'than'}
# prepositions that end a denied noun phrase: a cause or a time is not part of what is denied
PHRASE_END = {'during', 'due', 'because', 'owing', 'after', 'before', 'until', 'since', 'throughout', 'despite',
              'from', 'by', 'when', 'as', 'caused', 'causing', 'leading', 'resulting', 'making', 'thanks'}
# words that present something as an impression rather than a finding
HEDGES = {'looks', 'look', 'looked', 'looking', 'seems', 'seem', 'seemed', 'appears', 'appear', 'appeared',
          'initially', 'first', 'resembles', 'resemble', 'resembled', 'resembling', 'might', 'may', 'could',
          'apparently', 'possibly', 'probably', 'likely', 'superficially', 'seemingly', 'suspected', 'suspect',
          'apparent', 'possible', 'potential', 'presumed', 'presumably'}
# words that introduce what something really is
CORRECTIONS = {'actually', 'just', 'merely', 'simply', 'really', 'reality', 'fact', 'only', 'instead'}
# words that withdraw an impression outright
VERDICTS = {'retracted', 'dismissed', 'disproved', 'disproven', 'unfounded'}
# nouns that say what caused something, so "X is the cause" explains rather than corrects
_EXPLAINS = EXISTENCE_NOUNS | {'cause', 'causes', 'reason', 'source', 'culprit', 'result'}
# words that restrict an impression in time or extent instead of correcting it ("but only for 2 s")
_RESTRICTS = {'briefly', 'moment', 'moments', 'second', 'seconds', 'minute', 'minutes', 'instant', 'partly',
              'partially'}
# answers that leave a question open
_UNSURE = {'unclear', 'uncertain', 'unknown', 'unsure', 'maybe', 'perhaps', 'possibly', 'undetermined',
           'unconfirmed', 'inconclusive', 'undecided', 'unverified', 'ambiguous'}
_UNSURE_AFTER_NEGATOR = {'tell', 'say', 'sure', 'certain', 'clear', 'determine', 'confirm', 'know', 'verify'}
# words that add nothing to a bare answer ("No, not at all", "None observed", "Not present")
_ANSWER_FILLER = EXISTENCE | EXISTENCE_NOUNS | AUX | {'at', 'all', 'really', 'here', 'either', 'so', 'it', 'this',
                                                      'that', 'the', 'any', 'anywhere', 'whatsoever', 'ever',
                                                      'definitely', 'certainly', 'clearly', 'of', 'course', 'sir',
                                                      'a', 'an', 'in', 'frame', 'frames', 'view', 'footage'}
_BARE_NO = {'no', 'none', 'nope', 'nah', 'negative'}
NEGATORS = PREDICATE_NEGATORS | DETERMINER_NEGATORS | POST_NEGATORS | _NT | {'nope', 'nah', 'negative'}


def _transpositions(word):
    return {word[:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len(word) - 1)}


# a negator typed with two letters swapped ("nto"); none of these spellings is an English word
_TYPOS = set().union(*(_transpositions(w) for w in ('not', 'never', 'none', 'without'))) - {'not', 'never', 'none',
                                                                                           'without'}


class _Tok:
    __slots__ = ('w', 'start', 'end')

    def __init__(self, w, start, end):
        self.w, self.start, self.end = w, start, end


def _tokens(text: str) -> list[_Tok]:
    text = text.replace('’', "'").replace('‘', "'")
    return [_Tok(m.group().lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _pred_negator(w: str) -> bool:
    return w in PREDICATE_NEGATORS or w in _NT or w.endswith("n't") or w in _TYPOS


def _negator(w: str) -> bool:
    return w in NEGATORS or _pred_negator(w)


def _is_word(t: _Tok) -> bool:
    return t.w[:1].isalpha()


def _negative_only(toks: list[_Tok]) -> bool:
    """A bare negative answer: a negator and nothing else that says anything ("No.", "None observed", "Not at all",
    "No, ..." where the comma closes the answer)."""
    words = [t for t in toks if _is_word(t) or t.w == ',']
    if not words:
        return False
    if words[0].w in _BARE_NO and (len(words) == 1 or words[1].w == ','):
        return True
    words = [t.w for t in words if t.w != ',']
    return any(map(_negator, words)) and all(_negator(w) or w in _ANSWER_FILLER for w in words)


def _unsure(toks: list[_Tok]) -> bool:
    """An answer that leaves the question open ("Unclear from these frames", "Hard to tell", "Should it be flagged?")."""
    words = [t.w for t in toks]
    return (any(w in _UNSURE for w in words) or bool(words) and words[-1] == '?'
            or any(_negator(a) and b in _UNSURE_AFTER_NEGATOR for a, b in zip(words, words[1:]))
            or any(a in ('hard', 'difficult', 'impossible') and b == 'to' for a, b in zip(words, words[1:])))


def _bare_list(toks: list[_Tok]) -> bool:
    """Noun phrases separated by commas or "and"/"or", with no verb of their own ("glare, occlusion, dropout")."""
    words = [t.w for t in toks if _is_word(t)]
    return bool(words) and words[0] not in DETERMINERS and not any(w in AUX or _negator(w) for w in words)


def _clauses(toks: list[_Tok]) -> list[list[_Tok]]:
    """Split one sentence into clauses at conjunctions that start a clause and at a comma, "and" or "or" followed by a
    new subject."""
    out, cur = [], []
    n = len(toks)
    for i, t in enumerate(toks):
        prev = toks[i - 1].w if i else ''
        nxt = toks[i + 1].w if i + 1 < n else ''
        split = False
        if t.w == 'actually':
            # "actually" opens a correction only at the head of a clause ("glare, actually a tablecloth"); inside
            # one it qualifies the predicate ("is actually still")
            split = prev in ('', ',', ';')
        elif t.w in BOUNDARY:
            # "not yet", "never once", "at once" are adverbs
            split = not (t.w in ('yet', 'once') and (_pred_negator(prev) or prev in ('at', 'all')))
        elif t.w == 'as':
            # "as" starts a clause only before a subject
            split = nxt in DETERMINERS and prev not in ('such', 'as', 'well')
        elif t.w == 'in' and nxt in ('fact', 'reality'):
            split = True
        elif t.w in (',', 'and', 'or', '(', ')', '-') and nxt in DETERMINERS:
            split = True
        if split and cur:
            out.append(cur)
            cur = []
        if t.w in (',', '(', ')', '-') and split:
            continue
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def _phrase_end(clause, j, stop):
    """Index just past a noun or predicate phrase starting at j: it runs to a word in stop, an auxiliary or the clause
    end, and on over "and", "or", "nor" and commas while the next word does not start a new subject."""
    n = len(clause)
    k = j
    while k < n:
        w = clause[k].w
        if w in stop or w in AUX or w in BOUNDARY:
            break
        if w in (',', 'and', 'or', 'nor', '&', '/'):
            nxt = clause[k + 1].w if k + 1 < n else ''
            if not nxt or nxt in DETERMINERS or nxt in AUX or nxt in BOUNDARY or nxt in stop:
                break
        elif not (_is_word(clause[k]) or clause[k].w[:1].isdigit()):
            break
        k += 1
    return k


def _subject_start(clause, i):
    """Index where the subject of a predicate negated at i starts: back over the auxiliaries just before it, then
    over the noun phrase, stopping at a relative pronoun, another auxiliary, punctuation, or a verb that already has
    its object ("glare on the bowl means the contact is not visible": the subject is "the contact")."""
    k = i - 1
    while k >= 0 and clause[k].w in AUX:
        k -= 1
    start = k + 1
    while k >= 0:
        t = clause[k]
        nxt = clause[k + 1].w if k + 1 < len(clause) else ''
        if (not _is_word(t) and not t.w[:1].isdigit()) or t.w in AUX or t.w in ('that', 'which', 'who'):
            break
        if (t.w.endswith('s') or t.w.endswith('ed')) and nxt in _ARTICLES and k + 1 < i:
            break
        start = k
        k -= 1
    return start


def _span(clause, a, b):
    return (clause[a].start, clause[b - 1].end) if b > a else None


def _clause_denials(clause: list[_Tok]) -> list[tuple[int, int]]:
    spans = []
    n = len(clause)
    words = [t.w for t in clause]
    if words and words[0] in CONDITIONAL:
        return [_span(clause, 0, n)]
    cancelled = set()
    i = 0
    while i < n:
        w = words[i]
        nxt = words[i + 1] if i + 1 < n else ''
        # a hyphenated "-free" denies the word it is attached to ("glare-free")
        if w.endswith('-free'):
            spans.append((clause[i].start, clause[i].end))
        pair = (w, nxt) in DETERMINER_PAIRS
        if i in cancelled:
            i += 2 if pair else 1
            continue
        if pair or w in DETERMINER_NEGATORS:
            width = 2 if pair else 1
            j = i + width
            # "ruled out" after its subject ("glare was ruled out") denies the subject, before an object the object
            if w in ('ruled', 'rule', 'rules', 'ruling') and (j >= n or not _is_word(clause[j])
                                                              or words[j] in PREDICATE_END):
                spans.append(_span(clause, _subject_start(clause, i), i))
                i = j
                continue
            # an elliptical "none" denies what the clause is about: "glare: none", "occlusion was looked for and
            # none observed", "glare and clipping were checked and none appear"
            if w == 'none' and not pair and (j >= n or not _is_word(clause[j]) or words[j] in EXISTENCE
                                             or words[j] in AUX or words[j] in ('at', 'whatsoever')):
                spans.append(_span(clause, 0, i))
                i = j
                continue
            k = _phrase_end(clause, j, PHRASE_END)
            span = _span(clause, j, k)
            if span:
                spans.append(span)
            # a negative subject: its verb's predicate is denied as well ("no frame is frozen")
            if (w in QUANTIFIERS or (w, nxt) == ('none', 'of')) and k < n and words[k] in AUX:
                p = k
                while p < n and words[p] in SKIP:
                    p += 1
                q = _phrase_end(clause, p, PREDICATE_END)
                if _span(clause, p, q):
                    spans.append(_span(clause, p, q))
            # an existence predicate before the negative phrase denies its subject ("glare appears in no frame")
            before = [x for x in words[max(0, i - 3):i] if x not in PREDICATE_END]
            if w in QUANTIFIERS and any(x in EXISTENCE for x in before):
                e = max(x for x in range(max(0, i - 3), i) if words[x] in EXISTENCE)
                if _span(clause, _subject_start(clause, e), e):
                    spans.append(_span(clause, _subject_start(clause, e), e))
            i = j
            continue
        if w in POST_NEGATORS:
            s = _subject_start(clause, i)
            if s < i:
                spans.append(_span(clause, s, i))
            i += 1
            continue
        if _pred_negator(w) or (w == 'no' and nxt == 'longer'):
            j = i + (2 if w == 'no' else 1)
            while j < n and (words[j] in SKIP or (words[j] in _RAISING and j + 1 < n and words[j + 1] == 'to')):
                j += 2 if words[j] in _RAISING else 1
            if j < n and (words[j] in DETERMINER_NEGATORS or words[j] in POST_NEGATORS
                          or (words[j], words[j + 1] if j + 1 < n else '') in DETERMINER_PAIRS):
                # two negations cancel: "not without glare", "not free of glare", "not absent"
                cancelled.add(j)
                i = j
                continue
            k = _phrase_end(clause, j, PREDICATE_END)
            if k == j:
                i += 1
                continue
            head = words[j]
            ends = head in REMEDY or any((words[x] == 'back' and words[x - 1] in _RETURNING)
                                         or (words[x] == 'away' and words[x - 1] in _GOING)
                                         for x in range(j + 1, k))
            if not ends:
                spans.append(_span(clause, j, k))
            if head in EXISTENCE or any(x in EXISTENCE_NOUNS for x in words[j:k]):
                s = _subject_start(clause, i)
                if s < i:
                    spans.append(_span(clause, s, i))
            i = k
            continue
        i += 1
    return [s for s in spans if s]


def _hedge_at(words):
    """Index of the first hedge in a clause, or None."""
    for x, w in enumerate(words):
        if w in HEDGES or (w == 'first' and x and words[x - 1] == 'at'):
            return x
    return None


def _identity(words, start):
    """Index of a linking verb at or after start whose complement says what something is ("is a shadow", "are just
    motion"), or None. A complement that names a cause ("is the problem", "is the cause") explains, not corrects."""
    for x in range(start, len(words) - 1):
        if words[x] in COPULA and (words[x + 1] in _ARTICLES or words[x + 1] in CORRECTIONS):
            if not any(w in _EXPLAINS for w in words[x + 1:x + 5]):
                return x
    return None


def _restricts(words):
    return (any(w in _RESTRICTS for w in words) or any(w[:1].isdigit() for w in words)
            or any(a in ('only', 'just') and b in PREDICATE_END for a, b in zip(words, words[1:])))


def _correction(clause, cross_sentence=False):
    """Whether this clause says what an impression before it really was: a bare denial ("but it isn't"), a word that
    introduces what it is ("actually", "just"), a verdict ("retracted") or a link to what it is ("but it is a
    shadow"). A clause that only limits the impression in time or extent ("but only for 2 s") does not."""
    words = [t.w for t in clause if _is_word(t) or t.w[:1].isdigit()]
    lead = words[0] if words and words[0] in CONTRAST else ''
    body = words[1:] if lead else words
    if not body or _restricts(body):
        return False
    if any(w in VERDICTS for w in body):
        return True
    if any(a in ('turns', 'turned') and b == 'out' for a, b in zip(body, body[1:])):
        return True
    if cross_sentence:
        return any(w in CORRECTIONS for w in body)
    if not lead:
        return False
    content = [w for w in body if w not in ('it', 'this', 'that', 'is', 'was', "it's", "that's", 'really', 'the',
                                            'case', 'so', 'not', 'isnt', 'wasnt', "isn't", "wasn't")]
    if not content and any(map(_pred_negator, body)):
        return True
    return lead in ('actually', 'in', 'instead', 'rather') or any(w in CORRECTIONS for w in body) \
        or _identity(body, 0) is not None


def _corrected_within(clause):
    """The part of a clause that an impression and its correction share, which the correction retracts: the hedged
    subject of "what looks like glare is a white plate", "the washed-out look is the white tablecloth", and a whole
    clause given a verdict ("suspected obstruction, retracted"). None when the clause corrects nothing."""
    words = [t.w for t in clause]
    h = _hedge_at(words)
    if h is None:
        return None
    if any(w in VERDICTS for w in words[h + 1:]):
        return (clause[0].start, clause[-1].end)
    x = _identity(words, h + 1)
    if x is not None and not _restricts(words[x:]):
        return _span(clause, 0, x)
    return None


def _sentences(toks):
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if t.w in _SENTENCE_END:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def denied_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of text that the text does not assert: denied, retracted, asked about or supposed."""
    if not text:
        return []
    toks = _tokens(str(text))
    spans = []
    sentences = _sentences(toks)
    flat = []   # (sentence index, clause)
    for si, sentence in enumerate(sentences):
        body = [t for t in sentence if t.w not in _SENTENCE_END]
        # a question asserts only what an answer in the next sentence affirms
        if sentence and sentence[-1].w == '?' and body:
            answer = [t for t in sentences[si + 1] if t.w != '\n'] if si + 1 < len(sentences) else []
            if not answer or _negative_only(answer) or _unsure(answer):
                spans.append((body[0].start, body[-1].end))
        # a label and its answer: either side that is a bare negative denies the other ("Glare: none"), and a
        # negative label denies the bare list it introduces ("None of these were seen: glare, rotation")
        parts, cur = [], []
        for t in body:
            if t.w == ':':
                parts.append(cur)
                cur = []
            else:
                cur.append(t)
        parts.append(cur)
        for a, b in zip(parts, parts[1:]):
            if not (a and b):
                continue
            if _negative_only(b) and not _negative_only(a):
                spans.append((a[0].start, a[-1].end))
            elif (_negative_only(a) or any(_negator(t.w) for t in a) and _bare_list(b)) and not _negative_only(b):
                spans.append((b[0].start, b[-1].end))
        for part in parts:
            for clause in _clauses(part):
                flat.append((si, clause))
                spans.extend(_clause_denials(clause))
    # an impression that is then corrected is retracted, back to the hedge that made it
    for si, clause in flat:
        span = _corrected_within(clause)
        if span:
            spans.append(span)
    for k, (sk, clause) in enumerate(flat):
        if not k:
            continue
        same = flat[k - 1][0] == sk
        if not _correction(clause, cross_sentence=not same):
            continue
        back = k - 1
        while back >= 0 and flat[back][0] == flat[k - 1][0]:
            words = [t.w for t in flat[back][1]]
            if _hedge_at(words) is not None:
                spans.append((flat[back][1][0].start, flat[k - 1][1][-1].end))
                break
            if words and words[0] in CONTRAST or not same:
                break
            back -= 1
    return spans


def asserted(pattern: re.Pattern, text: str, spans: list[tuple[int, int]] | None = None) -> bool:
    """Whether some match of pattern in text is asserted: it starts outside every span the text does not assert."""
    if not text:
        return False
    matches = list(pattern.finditer(text))
    if not matches:
        return False
    spans = denied_spans(text) if spans is None else spans
    return any(not any(a <= m.start() < b for a, b in spans) for m in matches)
