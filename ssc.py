#!/usr/bin/env python3
r"""Construct the Structural Syntactic Congruence (SSC) of an LTL formula.

Python 3.10+, standard library only.

Formula syntax:
    {0}, {1}, ...       alphabet symbols, one symbol per word position
    \top                 true
    \bot                 false (abbreviation for \neg\top)
    \neg phi             negation
    \nXt phi             next
    \Fnl phi             finally (abbreviation for \top\Utl phi)
    \Glb phi             globally (abbreviation for \neg(\top\Utl\neg phi))
    phi \Utl psi         until
    phi \land psi        conjunction (abbreviation for \neg(\neg phi\lor\neg psi))
    phi \lor psi         disjunction
    (phi)                grouping

Precedence, strongest first: unary operators, \Utl, \land, \lor.
Until is right associative; conjunction and disjunction are left associative.
Derived operators are expanded during parsing, without simplification;
the displayed formula shows the expanded core syntax. Implication must
still be expanded explicitly.
The construction follows the formula syntax; it does not minimize a
property automaton or identify all semantically equivalent formulas.

Words:
    {0}{1}{0}{0}
    \eps                 empty word (also: eps or an empty line)

Interactive:
    python3 ssc.py -n 1
    python3 ssc.py -n 1 --formula '\top\Utl{0}'

Command line:
    python3 ssc.py -n 1 --formula '\top\Utl{0}' \
        --word1 '{0}{1}{0}{0}' --word2 '{0}{1}'
    python3 ssc.py --alphabet '{0}{7}{12}' --formula '\nXt{7}'
    python3 ssc.py -n 1 --formula '\Glb({0}\lor\Fnl{1})'
    python3 ssc.py --self-test

With -n N, the alphabet is exactly {0}, ..., {N}, inclusive.
After the classes, print the formula's accepting finite and infinite
idempairs, i.e. alpha_cip([[phi]]). Finite words use strong next; the
empty word is excluded for every formula. Infinite words use ordinary LTL.
Without --word1/--word2, repeatedly prompt "Word:" after these listings.
Each input word is classified immediately; quit or EOF ends the session.
The --word1/--word2 options retain the optional batch comparison mode.
The printed transition table is an exact description of each class:
Ck contains precisely the words whose run from C0 ends in Ck.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import re
import sys
from typing import Callable, Hashable, Iterable, Sequence


class InputError(ValueError):
    """Invalid formula, word, or alphabet."""


class ConstructionLimit(RuntimeError):
    """The exact construction exceeded a user-configurable class limit."""


@dataclass(frozen=True)
class Formula:
    op: str
    args: tuple[Formula, ...] = ()
    symbol: int | None = None


def format_formula(f: Formula) -> str:
    if f.op == "top":
        return r"\top"
    if f.op == "atom":
        return "{" + str(f.symbol) + "}"
    if f.op in ("not", "next"):
        command = r"\neg" if f.op == "not" else r"\nXt"
        return command + "(" + format_formula(f.args[0]) + ")"
    command = r"\lor" if f.op == "or" else r"\Utl"
    return "(" + format_formula(f.args[0]) + " " + command + " " \
        + format_formula(f.args[1]) + ")"


_SYMBOL = re.compile(r"\{\s*([0-9]+)\s*\}")
_COMMAND = re.compile(r"\\[A-Za-z]+")
_COMMANDS = {
    r"\top": "top", r"\neg": "not", r"\lor": "or",
    r"\nXt": "next", r"\Utl": "until",
    r"\bot": "bottom", r"\land": "and",
    r"\Fnl": "finally", r"\Glb": "globally",
}
_EPSILON = {"", r"\eps", "eps"}


def _error(message: str, text: str, position: int) -> InputError:
    return InputError(f"{message} at position {position + 1}:\n"
                      f"  {text}\n  {' ' * position}^")


@dataclass(frozen=True)
class Token:
    kind: str
    value: int | None
    position: int


def _tokens(text: str) -> list[Token]:
    result: list[Token] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        match = _SYMBOL.match(text, i)
        if match:
            result.append(Token("atom", int(match.group(1)), i))
            i = match.end()
            continue
        match = _COMMAND.match(text, i)
        if match:
            command = match.group()
            if command not in _COMMANDS:
                raise _error(f"Unknown operator {command!r}", text, i)
            result.append(Token(_COMMANDS[command], None, i))
            i = match.end()
            continue
        if text[i] in "()":
            result.append(Token(text[i], None, i))
            i += 1
            continue
        raise _error("Expected {natural}, an operator, or parentheses", text, i)
    result.append(Token("end", None, len(text)))
    return result


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokens(text)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def take(self, kind: str) -> Token:
        token = self.current
        if token.kind != kind:
            raise _error(f"Expected {kind!r}, found {token.kind!r}",
                         self.text, token.position)
        self.index += 1
        return token

    def disjunction(self) -> Formula:
        f = self.conjunction()
        while self.current.kind == "or":
            self.take("or")
            f = Formula("or", (f, self.conjunction()))
        return f

    def conjunction(self) -> Formula:
        f = self.until()
        while self.current.kind == "and":
            self.take("and")
            right = self.until()
            f = Formula("not", (Formula("or", (
                Formula("not", (f,)), Formula("not", (right,)),
            )),))
        return f

    def until(self) -> Formula:
        f = self.unary()
        if self.current.kind == "until":
            self.take("until")
            return Formula("until", (f, self.until()))
        return f

    def unary(self) -> Formula:
        token = self.current
        if token.kind in ("not", "next", "finally", "globally"):
            self.take(token.kind)
            child = self.unary()
            if token.kind == "finally":
                return Formula("until", (Formula("top"), child))
            if token.kind == "globally":
                return Formula("not", (Formula("until", (
                    Formula("top"), Formula("not", (child,)),
                )),))
            return Formula(token.kind, (child,))
        if token.kind == "top":
            self.take("top")
            return Formula("top")
        if token.kind == "bottom":
            self.take("bottom")
            return Formula("not", (Formula("top"),))
        if token.kind == "atom":
            self.take("atom")
            return Formula("atom", symbol=token.value)
        if token.kind == "(":
            self.take("(")
            f = self.disjunction()
            self.take(")")
            return f
        raise _error("Expected a formula", self.text, token.position)


def parse_formula(text: str) -> Formula:
    """Parse and expand abbreviations into top/atom/not/next/until/or."""
    parser = _Parser(text)
    f = parser.disjunction()
    parser.take("end")
    return f


def parse_word(text: str, alphabet: Iterable[int] | None = None) -> tuple[int, ...]:
    text = text.strip()
    if text in _EPSILON:
        return ()
    allowed = None if alphabet is None else frozenset(alphabet)
    symbols = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        match = _SYMBOL.match(text, i)
        if not match:
            raise _error(r"Expected {natural}; use \eps for the empty word",
                         text, i)
        symbol = int(match.group(1))
        if allowed is not None and symbol not in allowed:
            raise _error(f"Symbol {{{symbol}}} is outside the alphabet", text, i)
        symbols.append(symbol)
        i = match.end()
    return tuple(symbols)


def format_word(word: Iterable[int]) -> str:
    result = "".join("{" + str(symbol) + "}" for symbol in word)
    return result or r"\eps"


def _red(items: Iterable[int]) -> tuple[int, ...]:
    """Delete all occurrences after the first occurrence, preserving order."""
    return tuple(dict.fromkeys(items))


class Quotient:
    """A completely enumerated SSC quotient with BFS-numbered classes.

    Class 0 is the identity. A class's parent pointer records a shortest
    representative. Internal signatures use IDs from auxiliary quotients.
    """

    def __init__(
        self, alphabet: tuple[int, ...], kind: str,
        identity_signature: Hashable,
        letter_signatures: dict[int, Hashable],
        combine: Callable[[Hashable, Hashable], Hashable],
        max_classes: int,
    ):
        self.alphabet = alphabet
        self.kind = kind
        self._combine = combine
        self.signatures: list[Hashable] = [identity_signature]
        self._ids: dict[Hashable, int] = {identity_signature: 0}
        self._parents: list[tuple[int, int] | None] = [None]
        self.transitions: list[tuple[int, ...]] = []
        self._alphabet_index = {symbol: i for i, symbol in enumerate(alphabet)}
        self._products: dict[tuple[int, int], int] = {}

        # Enumerate precisely the signatures reachable by finite words.
        cursor = 0
        while cursor < len(self.signatures):
            row = []
            for symbol in alphabet:
                signature = combine(self.signatures[cursor],
                                    letter_signatures[symbol])
                target = self._ids.get(signature)
                if target is None:
                    if max_classes and len(self.signatures) >= max_classes:
                        raise ConstructionLimit(
                            f"Exceeded --max-classes={max_classes} while "
                            f"constructing {kind}. No complete quotient was "
                            "produced. Raise the limit, or use 0 for no limit."
                        )
                    target = len(self.signatures)
                    self._ids[signature] = target
                    self.signatures.append(signature)
                    self._parents.append((cursor, symbol))
                row.append(target)
            self.transitions.append(tuple(row))
            cursor += 1
        self.letters = dict(zip(alphabet, self.transitions[0]))

    def __len__(self) -> int:
        return len(self.signatures)

    def multiply(self, left: int, right: int) -> int:
        if not (0 <= left < len(self) and 0 <= right < len(self)):
            raise IndexError("Class ID is outside this quotient.")
        if left == 0:
            return right
        if right == 0:
            return left
        key = (left, right)
        if key not in self._products:
            signature = self._combine(self.signatures[left], self.signatures[right])
            try:
                self._products[key] = self._ids[signature]
            except KeyError as exc:
                raise RuntimeError("Internal error: quotient is not closed.") from exc
        return self._products[key]

    def classify(self, word: Iterable[int]) -> int:
        state = 0
        for symbol in word:
            try:
                column = self._alphabet_index[symbol]
            except KeyError as exc:
                raise InputError(f"Symbol {{{symbol}}} is outside the alphabet.") from exc
            state = self.transitions[state][column]
        return state

    def equivalent(self, left: Iterable[int], right: Iterable[int]) -> bool:
        return self.classify(left) == self.classify(right)

    def representative(self, class_id: int) -> tuple[int, ...]:
        if not 0 <= class_id < len(self):
            raise IndexError("Class ID is outside this quotient.")
        reversed_word = []
        while class_id:
            parent = self._parents[class_id]
            assert parent is not None
            class_id, symbol = parent
            reversed_word.append(symbol)
        return tuple(reversed(reversed_word))


class SSCBuilder:
    def __init__(self, alphabet: Iterable[int], max_classes: int = 10000):
        unique_symbols = set(alphabet)
        if any(not isinstance(x, int) or x < 0 for x in unique_symbols):
            raise InputError("Alphabet symbols must be natural numbers.")
        symbols = tuple(sorted(unique_symbols))
        if max_classes < 0:
            raise InputError("--max-classes must be nonnegative.")
        self.alphabet = symbols
        self.max_classes = max_classes
        self._formulas: dict[Formula, Quotient] = {}
        self._intersections: dict[tuple[Quotient, Quotient], Quotient] = {}

    def _new(self, kind, identity, letters, combine) -> Quotient:
        return Quotient(self.alphabet, kind, identity, letters,
                        combine, self.max_classes)

    def _intersection(self, left: Quotient, right: Quotient) -> Quotient:
        # Every SSC separates epsilon. Its intersection with the top
        # congruence therefore leaves its partition unchanged.
        if left is right or right.kind == "top":
            return left
        if left.kind == "top":
            return right
        key = (left, right)
        if key not in self._intersections:
            def combine(s, t):
                return (left.multiply(s[0], t[0]),
                        right.multiply(s[1], t[1]))
            self._intersections[key] = self._new(
                "intersection", (0, 0),
                {x: (left.letters[x], right.letters[x]) for x in self.alphabet},
                combine,
            )
        return self._intersections[key]

    def build(self, f: Formula) -> Quotient:
        if f in self._formulas:
            return self._formulas[f]
        alphabet = self.alphabet
        if f.op == "top":
            q = self._new("top", 0, {x: 1 for x in alphabet},
                          lambda s, t: int(bool(s or t)))
        elif f.op == "atom":
            if f.symbol not in alphabet:
                raise InputError(f"Formula symbol {{{f.symbol}}} is outside the alphabet.")
            q = self._new(
                "atom", 0,
                {x: 1 if x == f.symbol else 2 for x in alphabet},
                lambda s, t: s if s != 0 else t,
            )
        elif f.op == "not":
            q = self.build(f.args[0])
        elif f.op == "or":
            q = self._intersection(self.build(f.args[0]), self.build(f.args[1]))
        elif f.op == "next":
            child = self.build(f.args[0])
            def combine_next(s, t):
                if s is None:
                    return t
                if t is None:
                    return s
                # signature(w) = (class(w), class(tail_1(w)))
                # tail_1(uv) = tail_1(u)v for nonempty u.
                return (child.multiply(s[0], t[0]),
                        child.multiply(s[1], t[0]))
            q = self._new(
                "next", None,
                {x: (child.letters[x], 0) for x in alphabet},
                combine_next,
            )
        elif f.op == "until":
            rho = self._intersection(self.build(f.args[0]), self.build(f.args[1]))
            def combine_until(s, t):
                # The first entry of a nonempty suffix list is the rho
                # class of its entire word; the empty word has class 0.
                right_word_class = t[0] if t else 0
                return _red(
                    [rho.multiply(a, right_word_class) for a in s] + list(t)
                )
            q = self._new(
                "until", (),
                {x: (rho.letters[x],) for x in alphabet},
                combine_until,
            )
        else:
            raise InputError(f"Unknown formula operation: {f.op!r}")
        self._formulas[f] = q
        return q


def build_ssc(formula: str | Formula, alphabet: Iterable[int],
              max_classes: int = 10000) -> Quotient:
    """Build the exact reachable quotient; a limit failure returns no quotient."""
    f = parse_formula(formula) if isinstance(formula, str) else formula
    return SSCBuilder(alphabet, max_classes).build(f)


def _satisfies_on_path(
    formula: Formula,
    letters: tuple[int, ...],
    successors: tuple[int | None, ...],
) -> bool:
    """Evaluate at position 0 of a finite path or a finite lasso graph."""
    size = len(letters)
    if size == 0:
        # The manuscript excludes eps even for negated formulas.
        return False
    predecessors: list[list[int]] = [[] for _ in letters]
    for source, target in enumerate(successors):
        if target is not None:
            predecessors[target].append(source)

    @lru_cache(maxsize=None)
    def values(f: Formula) -> tuple[bool, ...]:
        if f.op == "top":
            return (True,) * size
        if f.op == "atom":
            return tuple(symbol == f.symbol for symbol in letters)
        if f.op == "not":
            return tuple(not value for value in values(f.args[0]))
        if f.op == "or":
            left, right = values(f.args[0]), values(f.args[1])
            return tuple(a or b for a, b in zip(left, right))
        if f.op == "next":
            child = values(f.args[0])
            return tuple(child[target] if target is not None else False
                         for target in successors)
        if f.op == "until":
            left, right = values(f.args[0]), values(f.args[1])
            # Strong until is the LEAST solution of
            # U = right OR (left AND next(U)).
            # Propagate from actual right-operand witnesses. A cycle
            # with left true everywhere but no right witness stays false.
            truth = list(right)
            pending = deque(i for i, value in enumerate(right) if value)
            while pending:
                target = pending.popleft()
                for source in predecessors[target]:
                    if left[source] and not truth[source]:
                        truth[source] = True
                        pending.append(source)
            return tuple(truth)
        raise InputError(f"Unknown formula operation: {f.op!r}")

    return values(formula)[0]


def satisfies_finite(formula: str | Formula, word: Sequence[int]) -> bool:
    """Mixed-word finite semantics: strong next, with eps always excluded."""
    f = parse_formula(formula) if isinstance(formula, str) else formula
    letters = tuple(word)
    size = len(letters)
    successors = tuple(i + 1 if i + 1 < size else None for i in range(size))
    return _satisfies_on_path(f, letters, successors)


def satisfies_lasso(
    formula: str | Formula, prefix: Sequence[int], period: Sequence[int],
) -> bool:
    """Exact ordinary LTL satisfaction on prefix * period^omega."""
    f = parse_formula(formula) if isinstance(formula, str) else formula
    prefix_word, period_word = tuple(prefix), tuple(period)
    if not period_word:
        raise InputError("An infinite lasso requires a nonempty period.")
    letters = prefix_word + period_word
    successors = tuple(range(1, len(letters))) + (len(prefix_word),)
    return _satisfies_on_path(f, letters, successors)


def idempair_specification(
    formula: str | Formula, quotient: Quotient,
) -> dict[tuple[int, int], bool]:
    """Map every linked pair (P,S) to acceptance by the formula.

    Precondition: quotient is this formula's SSC quotient (or a refinement).
    SSC homogeneity makes testing one representative exact for the whole
    interpretation P*S^omega. For S=C0 this interpretation is finite.
    The accepted set is alpha_cip([[formula]]) and is already overlap-closed:
    interpretations meeting the property are wholly contained in it.
    """
    f = parse_formula(formula) if isinstance(formula, str) else formula
    representatives = [quotient.representative(i) for i in range(len(quotient))]
    idempotents = [s for s in range(len(quotient))
                   if quotient.multiply(s, s) == s]
    specification: dict[tuple[int, int], bool] = {}
    for s in idempotents:
        for p in range(len(quotient)):
            if quotient.multiply(p, s) != p:
                continue
            if s == 0:
                accepted = satisfies_finite(f, representatives[p])
            else:
                accepted = satisfies_lasso(f, representatives[p], representatives[s])
            specification[p, s] = accepted
    return specification


def print_idempair_specification(
    specification: dict[tuple[int, int], bool],
) -> None:
    finite = sorted(pair for pair, accepted in specification.items()
                    if accepted and pair[1] == 0)
    infinite = sorted(pair for pair, accepted in specification.items()
                      if accepted and pair[1] != 0)
    total_finite = sum(s == 0 for _, s in specification)
    total_infinite = len(specification) - total_finite
    print("\nFormula idempairs:")
    print(r"Valid pairs satisfy P*S=P and S*S=S; C0=[\eps].")
    print(f"Linked pairs: {len(specification)} "
          f"(finite: {total_finite}, infinite: {total_infinite})")
    print(f"Accepting: {len(finite) + len(infinite)}; "
          f"rejecting: {len(specification) - len(finite) - len(infinite)}")
    print("Specification alpha_cip([[phi]]) is the union of the two sets below.")
    print(r"Finite (P,C0) represents P; \eps satisfies no formula.")
    print("Infinite (P,S) represents P*S^omega, with S != C0.")
    for label, pairs in (("Accepting finite idempairs", finite),
                         ("Accepting infinite idempairs", infinite)):
        print(f"\n{label} ({len(pairs)}):", end=' ')
        if not pairs:
            print("{}")
            continue
        print("{ ", end='')
        for p, s in pairs:
            print(f"(C{p}, C{s}) ", end='')
        print("}")


def print_classes(q: Quotient, show_signatures: bool = False) -> None:
    print(f"\nSSC classes: {len(q)}")
    print(r"C0 is the empty-word class [\eps].")
    print("Each class consists exactly of the words whose run from C0 ends there.")
    print("The delta columns below give that exact membership automaton.")
    header = ["Class", "Representative"] + [
        "delta({" + str(x) + "})" for x in q.alphabet
    ]
    if show_signatures:
        header.append("Structural signature")
        print("Signature integers refer to class IDs of auxiliary quotients.")
    rows = []
    for i in range(len(q)):
        row = [f"C{i}", format_word(q.representative(i))]
        row += [f"C{target}" for target in q.transitions[i]]
        if show_signatures:
            row.append(repr(q.signatures[i]))
        rows.append(row)
    widths = [max(len(row[j]) for row in [header] + rows)
              for j in range(len(header))]
    def print_row(row):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    print()
    print_row(header)
    print_row(["-" * width for width in widths])
    for row in rows:
        print_row(row)


def print_classification(q: Quotient, word: tuple[int, ...]) -> None:
    class_id = q.classify(word)
    print(f"Class: C{class_id} = "
          f"[{format_word(q.representative(class_id))}]_ssc")


def print_comparison(q: Quotient, left: tuple[int, ...],
                     right: tuple[int, ...]) -> None:
    a, b = q.classify(left), q.classify(right)
    print(f"\nWord 1: {format_word(left)}")
    print(f"Class: C{a} = [{format_word(q.representative(a))}]_ssc")
    print(f"Word 2: {format_word(right)}")
    print(f"Class: C{b} = [{format_word(q.representative(b))}]_ssc")
    print("SSC-equivalent: " + ("YES" if a == b else "NO"))


def _read_word(prompt: str, alphabet: tuple[int, ...]) -> tuple[int, ...] | None:
    while True:
        text = input(prompt)
        if text.strip().lower() in {"quit", "exit"}:
            return None
        try:
            return parse_word(text, alphabet)
        except InputError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List SSC classes and formula idempairs, then classify words.",
        epilog=(r"Unary > \Utl > \land > \lor. "
                r"\bot, \Fnl, \Glb, and \land expand to core syntax. "
                "Use parentheses to make grouping explicit."),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--max-symbol", type=int,
                       help="alphabet {0}, ..., {N}, inclusive")
    group.add_argument("--alphabet", help="explicit alphabet, e.g. '{0}{7}{12}'")
    parser.add_argument("--formula", help="LTL formula; quote literal backslashes in the shell")
    parser.add_argument("--word1", help="first finite word")
    parser.add_argument("--word2", help="second finite word")
    parser.add_argument("--list-only", action="store_true",
                        help="list classes and formula idempairs, then exit")
    parser.add_argument("--signatures", action="store_true",
                        help="also print internal structural signatures")
    parser.add_argument("--max-classes", type=int, default=10000,
                        help="maximum classes per auxiliary quotient; 0 means unlimited (default: 10000)")
    parser.add_argument("--self-test", action="store_true", help="run independent regression tests")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_tests()
    if (args.word1 is None) != (args.word2 is None):
        parser.error("--word1 and --word2 must be supplied together")
    if args.max_symbol is not None and args.max_symbol < 0:
        parser.error("--max-symbol must be nonnegative")
    if args.max_classes < 0:
        parser.error("--max-classes must be nonnegative")
    try:
        if args.alphabet is not None:
            alphabet = tuple(sorted(set(parse_word(args.alphabet))))
        else:
            n = args.max_symbol
            if n is None:
                while True:
                    text = input("Largest alphabet index n (alphabet {0}, ..., {n}): ")
                    try:
                        n = int(text)
                        if n < 0:
                            raise ValueError
                        break
                    except ValueError:
                        print("Enter a nonnegative integer.", file=sys.stderr)
            alphabet = tuple(range(n + 1))
        source = args.formula
        if source is None:
            source = input("Formula: ")
        f = parse_formula(source)
        print("Alphabet: " + (", ".join("{" + str(x) + "}" for x in alphabet) or "(empty)"))
        print("Formula: " + format_formula(f))
        q = build_ssc(f, alphabet, args.max_classes)
        print_classes(q, args.signatures)
        print_idempair_specification(idempair_specification(f, q))
        if args.list_only:
            return 0
        if args.word1 is not None:
            print_comparison(q, parse_word(args.word1, alphabet),
                             parse_word(args.word2, alphabet))
            return 0
        print("\nEnter a word; use "
              + r"\eps for the empty word, or quit to exit.")
        while True:
            word = _read_word("Word: ", alphabet)
            if word is None:
                break
            print_classification(q, word)
        return 0
    except EOFError:
        return 0
    except (InputError, ConstructionLimit) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RecursionError:
        print("Error: formula nesting exceeded Python's recursion limit.", file=sys.stderr)
        return 2


def run_self_tests() -> int:
    """Checks use an independent direct suffix-signature definition."""
    import unittest

    @lru_cache(maxsize=None)
    def reference(f: Formula, w: tuple[int, ...]) -> Hashable:
        if f.op == "top":
            return bool(w)
        if f.op == "atom":
            return 0 if not w else 1 if w[0] == f.symbol else 2
        if f.op == "not":
            return reference(f.args[0], w)
        if f.op == "or":
            return reference(f.args[0], w), reference(f.args[1], w)
        if f.op == "next":
            return ("empty",) if not w else (
                "nonempty", reference(f.args[0], w), reference(f.args[0], w[1:]))
        # Calculate suffixes directly, without the multiplication algorithm.
        result = []
        for i in range(len(w)):
            value = (reference(f.args[0], w[i:]), reference(f.args[1], w[i:]))
            if value not in result:
                result.append(value)
        return tuple(result)

    phi = r"\neg(\top\Utl\neg(\top\Utl{1}))\lor(\top\Utl\neg(\top\Utl{2}))"

    class Tests(unittest.TestCase):
        def test_known_class_counts(self):
            cases = [
                (r"\top", 2), (r"{1}", 3),
                (r"\top\Utl{1}", 5), (r"\nXt\top", 3),
                (r"\neg(\top\Utl\neg(\top\Utl{1}))", 9),
                (phi, 35), (r"(\top\Utl\neg\nXt\top)\lor(" + phi + ")", 38),
            ]
            for source, count in cases:
                with self.subTest(source=source):
                    self.assertEqual(len(build_ssc(source, range(3))), count)
            self.assertEqual(len(build_ssc(r"{0}", [0])), 2)
            self.assertEqual(len(build_ssc(r"\top", [])), 1)

        def test_parser_and_words(self):
            f = parse_formula(r"\neg{0}\lor{1}\Utl\nXt{2}")
            self.assertEqual(f.op, "or")
            self.assertEqual(f.args[1].op, "until")
            self.assertEqual(f.args[1].args[1].op, "next")
            right = parse_formula(r"{0}\Utl{1}\Utl{2}")
            self.assertEqual(right.args[1].op, "until")
            self.assertEqual(parse_word(" {0} {1}{0}{0} "), (0, 1, 0, 0))
            self.assertEqual(parse_word(r"\eps"), ())
            self.assertEqual(format_word(()), r"\eps")
            self.assertEqual(parse_word(""), ())
            self.assertEqual(parse_word("{00012}"), (12,))
            for source in [r"\top\land", "{0}{1}", "()", r"\nXt", "{-1}",
                           r"\Fnl", r"\Glb", r"\bot{0}", r"\unknown{0}"]:
                with self.subTest(source=source):
                    with self.assertRaises(InputError):
                        parse_formula(source)
            with self.assertRaises(InputError):
                parse_word("{2}", [0, 1])
            with self.assertRaises(InputError):
                build_ssc("{2}", [0, 1])

        def test_derived_operator_expansion(self):
            cases = [
                (r"\bot", r"\neg\top"),
                (r"\Fnl{0}", r"\top\Utl{0}"),
                (r"\Glb{0}", r"\neg(\top\Utl\neg{0})"),
                (r"{0}\land{1}", r"\neg(\neg{0}\lor\neg{1})"),
                (r"\Glb\Fnl{0}", r"\neg(\top\Utl\neg(\top\Utl{0}))"),
                (r"\Fnl\Glb{0}", r"\top\Utl\neg(\top\Utl\neg{0})"),
                (r"\Glb(\Fnl{0}\land\neg\bot)",
                 r"\neg(\top\Utl\neg(\neg(\neg(\top\Utl{0})"
                 r"\lor\neg(\neg\neg\top))))"),
            ]
            for source, expanded in cases:
                with self.subTest(source=source):
                    f = parse_formula(source)
                    self.assertEqual(f, parse_formula(expanded))
                    self.assertEqual(parse_formula(format_formula(f)), f)
                    q = build_ssc(source, range(2))
                    core_q = build_ssc(expanded, range(2))
                    self.assertEqual(q.transitions, core_q.transitions)
                    self.assertEqual(idempair_specification(source, q),
                                     idempair_specification(expanded, core_q))

        def test_derived_operator_precedence(self):
            cases = [
                (r"{0}\lor{1}\land{2}", r"{0}\lor({1}\land{2})"),
                (r"{0}\land{1}\lor{2}", r"({0}\land{1})\lor{2}"),
                (r"{0}\land{1}\Utl{2}", r"{0}\land({1}\Utl{2})"),
                (r"{0}\Utl{1}\land{2}", r"({0}\Utl{1})\land{2}"),
                (r"\Fnl{0}\Utl\Glb{1}", r"(\Fnl{0})\Utl(\Glb{1})"),
                (r"\Glb{0}\land\Fnl{1}", r"(\Glb{0})\land(\Fnl{1})"),
                (r"{0}\land{1}\land{2}", r"({0}\land{1})\land{2}"),
            ]
            for source, grouped in cases:
                with self.subTest(source=source):
                    self.assertEqual(parse_formula(source), parse_formula(grouped))

        def test_derived_operator_semantics(self):
            words = [w for n in range(5) for w in product(range(2), repeat=n)]
            for word in words:
                with self.subTest(word=word):
                    self.assertEqual(satisfies_finite(r"\Fnl{0}", word), 0 in word)
                    self.assertEqual(satisfies_finite(r"\Glb{0}", word),
                                     bool(word) and all(x == 0 for x in word))
                    self.assertFalse(satisfies_finite(r"\bot", word))
                    self.assertEqual(satisfies_finite(r"\Fnl{0}\land\Fnl{1}", word),
                                     0 in word and 1 in word)
            prefixes = [w for n in range(3) for w in product(range(2), repeat=n)]
            periods = [w for n in range(1, 3) for w in product(range(2), repeat=n)]
            for prefix in prefixes:
                for period in periods:
                    with self.subTest(prefix=prefix, period=period):
                        word = prefix + period
                        self.assertEqual(satisfies_lasso(r"\Fnl{0}", prefix, period),
                                         0 in word)
                        self.assertEqual(satisfies_lasso(r"\Glb{0}", prefix, period),
                                         all(x == 0 for x in word))
                        self.assertFalse(satisfies_lasso(r"\bot", prefix, period))
                        self.assertEqual(satisfies_lasso(
                            r"\Fnl{0}\land\Fnl{1}", prefix, period),
                            0 in word and 1 in word)

        def test_suffixes_not_letters(self):
            q = build_ssc(r"\top\Utl\nXt{0}", range(2))
            self.assertFalse(q.equivalent((0, 1), (0, 1, 0)))

        def test_requested_comparison(self):
            q = build_ssc(r"\top\Utl{0}", range(2))
            self.assertTrue(q.equivalent((0, 1, 0, 0), (0, 1)))
            self.assertFalse(q.equivalent((), (0,)))
            self.assertFalse(q.equivalent((0,), (1,)))

        def test_interactive_single_word_lookup(self):
            from contextlib import redirect_stdout
            from io import StringIO
            from unittest.mock import patch
            for supplied_formula in (False, True):
                with self.subTest(supplied_formula=supplied_formula):
                    answers = [r"{0}{1}{0}{0}", "{0}{1}", r"\eps", "quit"]
                    arguments = ["-n", "1"]
                    expected_prompts = ["Word: "] * len(answers)
                    if supplied_formula:
                        arguments += ["--formula", r"\top\Utl{0}"]
                    else:
                        answers.insert(0, r"\top\Utl{0}")
                        expected_prompts.insert(0, "Formula: ")
                    answer_iter = iter(answers)
                    prompts = []
                    def read(prompt):
                        prompts.append(prompt)
                        return next(answer_iter)
                    output = StringIO()
                    with patch("builtins.input", read), redirect_stdout(output):
                        self.assertEqual(main(arguments), 0)
                    self.assertEqual(prompts, expected_prompts)
                    text = output.getvalue()
                    self.assertEqual(text.count("SSC classes: 5"), 1)
                    self.assertEqual(text.count("Formula idempairs:"), 1)
                    self.assertLess(text.index("SSC classes: 5"),
                                    text.index("Formula idempairs:"))
                    self.assertLess(text.index("Formula idempairs:"),
                                    text.index("Class: C3"))
                    self.assertEqual(text.count("Class: C3 = [{0}{1}]_ssc"), 2)
                    self.assertIn(r"Class: C0 = [\eps]_ssc", text)
                    self.assertLess(text.index("SSC classes: 5"), text.index("Class: C3"))
                    self.assertNotIn("Word 1:", text)
                    self.assertNotIn("Word 2:", text)

        def test_finite_and_lasso_semantics(self):
            self.assertFalse(satisfies_finite(r"\top", ()))
            self.assertFalse(satisfies_finite(r"\neg\top", ()))
            self.assertTrue(satisfies_finite(r"\top", (0,)))
            self.assertFalse(satisfies_finite(r"\nXt\top", (0,)))
            self.assertTrue(satisfies_lasso(r"\nXt\top", (), (0,)))
            self.assertTrue(satisfies_finite(r"\nXt{1}", (0, 1)))
            self.assertFalse(satisfies_finite(r"\nXt{1}", (1, 0)))
            gf0 = r"\neg(\top\Utl\neg(\top\Utl{0}))"
            fg0 = r"\top\Utl\neg(\top\Utl\neg{0})"
            self.assertTrue(satisfies_lasso(gf0, (), (0, 1)))
            self.assertFalse(satisfies_lasso(fg0, (), (0, 1)))
            self.assertFalse(satisfies_lasso(gf0, (0,), (1,)))
            self.assertTrue(satisfies_lasso(fg0, (1,), (0,)))
            self.assertFalse(satisfies_lasso(r"\top\Utl{0}", (), (1,)))
            self.assertFalse(satisfies_lasso(r"{0}\Utl{1}", (2,), (1,)))
            self.assertTrue(satisfies_lasso(r"{0}\Utl{1}", (), (0, 0, 1)))
            with self.assertRaises(InputError):
                satisfies_lasso(r"\top", (0,), ())

        def test_lasso_unrolling(self):
            sources = [
                r"{0}", r"\neg{0}", r"\nXt{0}", r"\top\Utl{0}",
                r"{0}\Utl{1}", r"\neg(\top\Utl\neg(\top\Utl{0}))",
                r"\top\Utl\neg(\top\Utl\neg{0})",
                r"\nXt({0}\Utl\nXt{1})",
            ]
            prefixes = [w for n in range(3) for w in product(range(2), repeat=n)]
            periods = [w for n in range(1, 3) for w in product(range(2), repeat=n)]
            for source in sources:
                f = parse_formula(source)
                for prefix in prefixes:
                    for period in periods:
                        expected = satisfies_lasso(f, prefix, period)
                        self.assertEqual(expected, satisfies_lasso(
                            f, prefix + period, period))
                        self.assertEqual(expected, satisfies_lasso(
                            f, prefix, period + period))

        def test_idempair_specifications(self):
            f = parse_formula(r"\top\Utl{0}")
            q = build_ssc(f, range(2))
            specification = idempair_specification(f, q)
            self.assertEqual(len(specification), 15)
            self.assertEqual(sum(specification.values()), 12)
            self.assertEqual(sum(accepted and s == 0
                                 for (_, s), accepted in specification.items()), 3)
            self.assertFalse(specification[0, 0])
            complement = idempair_specification(Formula("not", (f,)), q)
            for pair, accepted in specification.items():
                if pair != (0, 0):
                    self.assertEqual(complement[pair], not accepted)
            self.assertFalse(complement[0, 0])
            # Every reported pair must satisfy both linking equations.
            for p, s in specification:
                self.assertEqual(q.multiply(p, s), p)
                self.assertEqual(q.multiply(s, s), s)
            # Check representative homogeneity on other concrete block choices.
            words = [w for n in range(4) for w in product(range(2), repeat=n)]
            for prefix in words:
                p = q.classify(prefix)
                self.assertEqual(specification[p, 0], satisfies_finite(f, prefix))
                for period in words:
                    if not period:
                        continue
                    s = q.classify(period)
                    if (p, s) in specification:
                        self.assertEqual(specification[p, s],
                                         satisfies_lasso(f, prefix, period))

        def test_recurrence_property_idempairs(self):
            f = parse_formula(phi)
            q = build_ssc(f, range(3))
            specification = idempair_specification(f, q)
            self.assertEqual(len(specification), 144)
            for (p, s), accepted in specification.items():
                if s == 0:
                    word = q.representative(p)
                    expected = bool(word) and word[-1] != 2
                else:
                    period = q.representative(s)
                    expected = 1 in period or 2 not in period
                self.assertEqual(accepted, expected)
            mixed = parse_formula(r"(\top\Utl\neg\nXt\top)\lor(" + phi + ")")
            mixed_q = build_ssc(mixed, range(3))
            mixed_specification = idempair_specification(mixed, mixed_q)
            for p in range(len(mixed_q)):
                self.assertEqual(mixed_specification[p, 0], p != 0)

        def test_exact_class_limit(self):
            with self.assertRaises(ConstructionLimit):
                build_ssc(phi, range(3), max_classes=34)
            self.assertEqual(len(build_ssc(phi, range(3), max_classes=35)), 35)

        def test_independent_reference_partitions(self):
            by_size = {1: {Formula("top"), Formula("atom", symbol=0),
                           Formula("atom", symbol=1)}}
            for size in range(2, 5):
                level = set()
                for child in by_size[size - 1]:
                    level.add(Formula("not", (child,)))
                    level.add(Formula("next", (child,)))
                for left_size in range(1, size - 1):
                    right_size = size - 1 - left_size
                    for left in by_size[left_size]:
                        for right in by_size[right_size]:
                            level.add(Formula("or", (left, right)))
                            level.add(Formula("until", (left, right)))
                by_size[size] = level
            words = [w for length in range(6)
                     for w in product(range(2), repeat=length)]
            for level in by_size.values():
                for f in level:
                    q = build_ssc(f, range(2))
                    classes_to_reference = {}
                    reference_to_classes = {}
                    for w in words:
                        i, signature = q.classify(w), reference(f, w)
                        self.assertEqual(classes_to_reference.setdefault(i, signature), signature)
                        self.assertEqual(reference_to_classes.setdefault(signature, i), i)
                        if w:
                            self.assertNotEqual(i, 0)
            for source in [phi, r"(\top\Utl{0})\Utl{1}"]:
                f = parse_formula(source)
                q = build_ssc(f, range(3))
                classes_to_reference = {}
                reference_to_classes = {}
                for length in range(6):
                    for w in product(range(3), repeat=length):
                        i, signature = q.classify(w), reference(f, w)
                        self.assertEqual(classes_to_reference.setdefault(i, signature), signature)
                        self.assertEqual(reference_to_classes.setdefault(signature, i), i)

        def test_multiplication_and_associativity(self):
            f = parse_formula(phi)
            q = build_ssc(f, range(3))
            representatives = [q.representative(i) for i in range(len(q))]
            for i, u in enumerate(representatives):
                for j, v in enumerate(representatives):
                    k = q.multiply(i, j)
                    self.assertEqual(k, q.classify(u + v))
                    self.assertEqual(reference(f, representatives[k]), reference(f, u + v))
            for i, j, k in product(range(len(q)), repeat=3):
                self.assertEqual(q.multiply(q.multiply(i, j), k),
                                 q.multiply(i, q.multiply(j, k)))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
