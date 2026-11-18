# Structural Syntactic Congruences (SSC) for Linear Temporal Logic (LTL) (ssc4ltl)

The code is LLM-generated and human-reviewed.

Python 3.10+, standard library only.

Formula syntax:
```
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
```

Precedence (strongest first): unary operators, `\Utl`, `\land`, `\lor`.
Until is right-associative; conjunction and disjunction are left-associative.
Derived operators are expanded during parsing, without simplification;
The displayed formula shows the expanded core syntax. Implication must
still be expanded explicitly.
The construction follows the formula syntax; it does not minimize a
property automaton or identify all semantically equivalent formulas.

```
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
```

With `-n N`, the alphabet is exactly `{0}, ..., {N}`, inclusive.
After the classes, print the formula's accepting finite and infinite
idempairs, i.e. $\alpha_\mathrm{cip}(E_\phi)$. Finite words use strong next; the
empty word is excluded for every formula. Infinite words use ordinary LTL.
Without --word1/--word2, repeatedly prompt "Word:" after these listings.
Each input word is classified immediately; quit or EOF ends the session.
The --word1/--word2 options retain the optional batch comparison mode.
The printed transition table is an exact description of each class:
Ck contains precisely the words whose run from C0 ends in Ck.
