"""
Tests for the Randomness Detection module.

Tests cover:
- Detection of random module functions
- Detection of numpy.random functions
- Detection of torch random functions
- Seed function tracking
- Session state persistence
- Allow-random annotation suppression
"""

import warnings
from types import SimpleNamespace

from cash.notebook.randomness import (
    RandomnessDetector,
    RandomnessVisitor,
    CashRandomnessWarning,
    check_and_warn_randomness,
)


class TestRandomnessVisitor:
    """Tests for the AST visitor that detects randomness calls."""
    
    def test_detect_random_random(self):
        """Test detection of random.random()."""
        code = """
import random
x = random.random()
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'random'
        
    def test_detect_random_choice(self):
        """Test detection of random.choice()."""
        code = """
import random
x = random.choice([1, 2, 3])
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'choice'
    
    def test_detect_numpy_random(self):
        """Test detection of np.random.rand()."""
        code = """
import numpy as np
x = np.random.rand(10)
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].module == 'numpy.random'
        assert visitor.random_calls[0].function == 'rand'
    
    def test_detect_numpy_randn(self):
        """Test detection of np.random.randn()."""
        code = """
import numpy as np
y = np.random.randn(5, 5)
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'randn'
    
    def test_detect_seed_call(self):
        """Test detection of seed function calls."""
        code = """
import random
random.seed(42)
x = random.random()
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.seed_calls) == 1
        assert len(visitor.random_calls) == 1
    
    def test_detect_numpy_seed(self):
        """Test detection of np.random.seed()."""
        code = """
import numpy as np
np.random.seed(42)
x = np.random.rand(10)
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.seed_calls) == 1
        assert len(visitor.random_calls) == 1
    
    def test_no_random_calls(self):
        """Test code with no random calls."""
        code = """
x = 1 + 2
y = [1, 2, 3]
"""
        import ast
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        
        assert len(visitor.random_calls) == 0
        assert len(visitor.seed_calls) == 0


class TestRandomnessDetector:
    """Tests for the RandomnessDetector class."""
    
    def test_unseeded_random_detected(self):
        """Test that unseeded random calls are detected."""
        detector = RandomnessDetector()
        code = """
import random
x = random.random()
"""
        calls, warnings_list, has_seed = detector.analyze_code(code)
        
        assert len(calls) == 1
        assert len(warnings_list) == 1
        assert 'Unseeded randomness' in warnings_list[0]
    
    def test_seeded_random_not_detected(self):
        """Test that seeded random calls don't trigger warnings."""
        detector = RandomnessDetector()
        code = """
import random
random.seed(42)
x = random.random()
"""
        calls, warnings_list, has_seed = detector.analyze_code(code)
        
        assert len(calls) == 0
        assert len(warnings_list) == 0
        assert has_seed == True  # Seed was set
    
    def test_session_state_persistence(self):
        """Test that seeding in one call persists to next call."""
        detector = RandomnessDetector()
        
        # First code sets seed
        code1 = """
import random
random.seed(42)
"""
        detector.analyze_code(code1)
        
        # Second code uses random
        code2 = """
import random
x = random.random()
"""
        calls, warnings_list, has_seed = detector.analyze_code(code2)
        
        # Should not warn because seed was set previously
        assert len(calls) == 0
        assert len(warnings_list) == 0
        assert has_seed == False  # No new seed in this code
    
    def test_reset_clears_state(self):
        """Test that reset() clears seeding state."""
        detector = RandomnessDetector()
        
        # Seed
        detector.analyze_code("import random; random.seed(42)")
        
        # Reset
        detector.reset()
        
        # Use random - should warn now
        calls, warnings_list, has_seed = detector.analyze_code("import random; x = random.random()")
        
        assert len(calls) == 1
        assert len(warnings_list) == 1
    
    def test_numpy_unseeded(self):
        """Test detection of unseeded numpy random."""
        detector = RandomnessDetector()
        code = """
import numpy as np
x = np.random.rand(100)
"""
        calls, warnings_list, has_seed = detector.analyze_code(code)
        
        assert len(calls) == 1
        assert 'numpy.random' in warnings_list[0]
    
    def test_numpy_seeded(self):
        """Test that seeded numpy random doesn't warn."""
        detector = RandomnessDetector()
        code = """
import numpy as np
np.random.seed(42)
x = np.random.rand(100)
"""
        calls, warnings_list, has_seed = detector.analyze_code(code)
        
        assert len(calls) == 0
        assert len(warnings_list) == 0
    
    def test_multiple_random_calls(self):
        """Test detection of multiple random calls."""
        detector = RandomnessDetector()
        code = """
import random
import numpy as np
x = random.random()
y = np.random.rand(10)
"""
        calls, warnings_list, has_seed = detector.analyze_code(code)
        
        assert len(calls) == 2
        assert len(warnings_list) == 2


class TestCheckAndWarnRandomness:
    """Tests for the check_and_warn_randomness function."""
    
    def test_warning_issued(self):
        """Test that warnings are issued for unseeded random."""
        detector = RandomnessDetector()
        code = """
import random
x = random.random()
"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness(code, detector, suppress_warning=False)
            
            assert len(w) == 1
            assert issubclass(w[0].category, CashRandomnessWarning)
    
    def test_warning_suppressed(self):
        """Test that warnings can be suppressed."""
        detector = RandomnessDetector()
        code = """
import random
x = random.random()
"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness(code, detector, suppress_warning=True)
            
            # Warning should be suppressed
            assert len(w) == 0
    
    def test_state_updated_when_suppressed(self):
        """Test that session state is still updated when warnings are suppressed."""
        detector = RandomnessDetector()
        
        # First call sets seed but suppresses warning
        code1 = """
import random
random.seed(42)
"""
        check_and_warn_randomness(code1, detector, suppress_warning=True)
        
        # Second call should not warn because seed was tracked
        code2 = """
import random
x = random.random()
"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness(code2, detector, suppress_warning=False)
            
            assert len(w) == 0


class TestSyntaxErrorHandling:
    """Tests for handling of invalid code."""
    
    def test_syntax_error_returns_empty(self):
        """Test that syntax errors don't crash the detector."""
        detector = RandomnessDetector()
        code = "this is not valid python code !@#$"
        
        calls, warnings_list, has_seed = detector.analyze_code(code)
        
        assert len(calls) == 0
        assert len(warnings_list) == 0
        assert has_seed == False


class TestRandomnessVisitorAdvanced:
    """Tests for advanced AST visitor patterns."""

    def test_detect_from_import_random(self):
        """Test detection of 'from random import random'."""
        import ast
        code = """
from random import random
x = random()
"""
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'random'

    def test_detect_from_import_choice(self):
        """Test detection of 'from random import choice'."""
        import ast
        code = """
from random import choice
x = choice([1, 2, 3])
"""
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'choice'

    def test_detect_torch_rand(self):
        """Test detection of torch.rand()."""
        import ast
        code = """
import torch
x = torch.rand(3, 3)
"""
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].module == 'torch'
        assert visitor.random_calls[0].function == 'rand'

    def test_detect_torch_manual_seed(self):
        """Test detection of torch.manual_seed()."""
        import ast
        code = """
import torch
torch.manual_seed(42)
"""
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        assert len(visitor.seed_calls) == 1

    def test_detect_tf_random(self):
        """Test detection of tf.random.uniform()."""
        import ast
        code = """
import tensorflow as tf
x = tf.random.uniform([3, 3])
"""
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'uniform'

    def test_detect_aliased_numpy_random(self):
        """Test detection with custom alias for numpy.random."""
        import ast
        code = """
from numpy import random as nr
x = nr.rand(10)
"""
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        assert len(visitor.random_calls) == 1
        assert visitor.random_calls[0].function == 'rand'

    def test_call_chain_extraction(self):
        """Test _get_call_chain helper."""
        import ast
        code = "np.random.seed(42)"
        tree = ast.parse(code)
        call = tree.body[0].value  # The Call node
        visitor = RandomnessVisitor()
        chain = visitor._get_call_chain(call.func)
        assert chain == ['np', 'random', 'seed']

    def test_visit_call_empty_chain(self):
        """Test visit_Call with a non-attribute/name call (e.g., lambda)."""
        import ast
        code = "(lambda: None)()"
        tree = ast.parse(code)
        visitor = RandomnessVisitor()
        visitor.visit(tree)
        # Should not crash, no random calls detected
        assert len(visitor.random_calls) == 0


class TestRNGStateCapture:
    """Tests for RNG state capture and restore."""

    def test_capture_random_state(self):
        """Test capturing stdlib random state."""
        import random
        from cash.notebook.randomness import capture_rng_state, restore_rng_state
        random.seed(42)
        state = capture_rng_state()
        assert 'random' in state
        # Generate a number, restore, generate again - should match
        val1 = random.random()
        restore_rng_state(state)
        val2 = random.random()
        assert val1 == val2

    def test_capture_numpy_state(self):
        """Test capturing numpy random state."""
        import numpy as np
        from cash.notebook.randomness import capture_rng_state, restore_rng_state
        np.random.seed(42)
        state = capture_rng_state()
        assert 'numpy.random' in state
        val1 = np.random.rand()
        restore_rng_state(state)
        val2 = np.random.rand()
        assert val1 == val2

    def test_restore_empty_state(self):
        """Test restore with empty state does nothing."""
        from cash.notebook.randomness import restore_rng_state
        restore_rng_state({})  # Should not crash

    def test_get_used_rng_modules(self):
        """Test identifying which RNG modules are used in code."""
        from cash.notebook.randomness import get_used_rng_modules
        code = """
import random
import numpy as np
x = random.random()
y = np.random.rand(10)
"""
        modules = get_used_rng_modules(code)
        assert 'random' in modules
        assert 'numpy.random' in modules

    def test_get_used_rng_modules_syntax_error(self):
        """Test get_used_rng_modules with invalid code."""
        from cash.notebook.randomness import get_used_rng_modules
        modules = get_used_rng_modules("not valid python !@#$")
        assert modules == set()

    def test_get_used_rng_modules_no_random(self):
        """Test get_used_rng_modules with code that has no random calls."""
        from cash.notebook.randomness import get_used_rng_modules
        modules = get_used_rng_modules("x = 1 + 2")
        assert modules == set()

    def test_get_used_rng_modules_with_seed(self):
        """Test that seed calls also report the module."""
        from cash.notebook.randomness import get_used_rng_modules
        code = """
import random
random.seed(42)
"""
        modules = get_used_rng_modules(code)
        assert 'random' in modules


class TestRandomnessDetectorAdvanced:
    """Additional tests for RandomnessDetector."""

    def test_mark_seeded_with_parent(self):
        """Test that marking a submodule seeded also marks the parent."""
        detector = RandomnessDetector()
        detector.mark_seeded('numpy.random')
        assert detector.is_seeded('numpy.random')
        assert detector.is_seeded('numpy')

    def test_is_seeded_via_parent(self):
        """Test that seeding parent module covers child."""
        detector = RandomnessDetector()
        detector.mark_seeded('numpy')
        assert detector.is_seeded('numpy.random')

    def test_is_seeded_no_parent(self):
        """Test is_seeded for top-level module (no parent)."""
        detector = RandomnessDetector()
        assert not detector.is_seeded('random')
        detector.mark_seeded('random')
        assert detector.is_seeded('random')

    def test_has_seed_calls_flag(self):
        """Test has_seed_calls return value."""
        detector = RandomnessDetector()
        _, _, has_seed = detector.analyze_code("import random; random.seed(42)")
        assert has_seed is True
        _, _, has_seed = detector.analyze_code("import random; random.random()")
        assert has_seed is False


class TestRngCarrierDetection:
    """Draws off an RNG *object* rather than a module global (CAS-135 hole 1).

    CAS-114's detector was rooted at RNG module names, so numpy's modern
    ``default_rng()`` API — the one numpy's own docs have pushed since 1.17, and
    the one CAS-90 already replays state for — was entirely invisible. An
    unseeded Monte Carlo written against it cached and froze in silence.

    The two halves that matter here: draws off an *unseeded* carrier must warn,
    and draws off a *seeded* one must not. A detector that only satisfied the
    first could just warn on every unresolved method call.
    """

    def _carriers(self, *statements):
        """Feed statements to one detector, as a session would; return draws."""
        detector = RandomnessDetector()
        found = []
        for stmt in statements:
            calls, _msgs, _seed = detector.analyze_code(stmt)
            found.extend(calls)
        return found

    def test_unseeded_default_rng_draw_detected(self):
        calls = self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng()",
            "x = rng.standard_normal(10)",
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'rng'
        assert calls[0].function == 'standard_normal'
        assert calls[0].module == 'numpy.random'

    def test_seeded_default_rng_draw_not_detected(self):
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng(42)",
            "x = rng.standard_normal(10)",
        ) == []

    def test_seed_keyword_counts_as_seeded(self):
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng(seed=42)",
            "x = rng.normal()",
        ) == []

    def test_explicit_none_seed_is_unseeded(self):
        """``default_rng(None)`` is numpy's spelling of 'draw from OS entropy'."""
        assert len(self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng(None)",
            "x = rng.normal()",
        )) == 1

    def test_carrier_local_to_a_function_body_detected(self):
        """The CAS-135 report's actual shape: the generator is a function local.

        ``g`` never reaches ``user_ns``, so this can only be caught on the AST of
        the ``def`` itself — no live-value classifier can ever see it.
        """
        calls = self._carriers(
            "import numpy as np\n"
            "def draw():\n"
            "    g = np.random.default_rng()\n"
            "    return float(g.standard_normal(10).mean())"
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'g'

    def test_rng_param_not_resolved_against_stale_global(self):
        """CAS-154 Symptom B: a parameter draw must not resolve against a
        same-named unseeded GLOBAL left in the session ledger by an earlier cell.

        ``def f(rng): rng.standard_normal()`` is pure w.r.t. the module-level
        ``rng`` — the parameter shadows it. Without lexical scoping the flat
        ledger fired a spurious warning; with it the def is silent whether or not
        the outer name was ``del``-eted.
        """
        # Variant WITHOUT del: the stale global is still bound in the session.
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng()",
            "def f(rng):\n    rng.standard_normal(10)",
        ) == []
        # Variant WITH del: the ``del rng`` that used to be the only silencer.
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng()",
            "del rng",
            "def f(rng):\n    rng.standard_normal(10)",
        ) == []
        # A parameter locally rebound to a concrete unseeded carrier still warns
        # (today's flat resolution is correct there — the binding is real).
        assert len(self._carriers(
            "import numpy as np",
            "def f(rng):\n    rng = np.random.default_rng()\n    rng.standard_normal(10)",
        )) == 1

    def test_unseeded_rng_arg_flows_into_param_draw(self):
        """CAS-154 Symptom A: an unseeded generator passed as a keyword argument
        into a function that draws off that parameter must warn at the call site.

        This is the idiomatic Monte-Carlo shape (``price(..., rng=default_rng())``)
        that cached and froze in total silence, while the equivalent inline draw
        was correctly warned.
        """
        # The def alone is silent (the argument decides seededness).
        # The call with an UNSEEDED argument warns exactly once, at the call site.
        calls = self._carriers(
            "import numpy as np",
            "def price(rng):\n    return rng.standard_normal(5)",
            "price(rng=np.random.default_rng())",
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'rng'
        assert calls[0].function == 'standard_normal'
        assert calls[0].module == 'numpy.random'
        # The warning is attributed to the CALL statement (line 1 of that source).
        assert calls[0].lineno == 1

        # Seeded argument -> reproducible -> no warning.
        assert self._carriers(
            "import numpy as np",
            "def price(rng):\n    return rng.standard_normal(5)",
            "price(rng=np.random.default_rng(42))",
        ) == []

    def test_call_arg_flow_silent_when_callee_never_scanned(self):
        """Conservative bias: an out-of-order / imported callee stays quiet.

        Nothing in the session says ``price`` draws off ``rng``, so passing an
        unseeded generator to it must not warn — matching the module's standing
        false-positive-avoidance philosophy.
        """
        assert self._carriers(
            "import numpy as np",
            "price(rng=np.random.default_rng())",
        ) == []


class TestPositionalRngCarrierArguments:
    """The POSITIONAL half of the call-site argument flow (CAS-154 round 4).

    CAS-154 taught the detector that ``price(rng=default_rng())`` freezes a
    Monte Carlo, but scoped the lesson to KEYWORD arguments: a parameter name is
    written in the source there, while a positional argument knows only its
    index. So the form quants actually write —
    ``price_asian(np.random.default_rng(), S0, K, ...)`` — cached and froze in
    total silence, and a "converged" price was a stale replay of one old draw.

    The index is mapped to a parameter name through the callee's runtime
    signature. That makes the two halves that matter here: an unseeded generator
    in a drawn-from slot must warn, and every slot the signature cannot name
    unambiguously must stay SILENT. A mapping that satisfied only the first could
    just warn on every positional argument.
    """

    @staticmethod
    def _ns(*sources):
        """A live namespace holding the real function objects *sources* define.

        Positional mapping reads the callee's signature off the live object, so
        the test has to supply one — this is the unit-level stand-in for the
        kernel's ``user_ns``.
        """
        ns: dict = {}
        for source in sources:
            exec(source, ns)
        return ns

    def _carriers(self, ns, *statements):
        """Feed statements to one detector wired to *ns*, as a session would."""
        detector = RandomnessDetector(shell=SimpleNamespace(user_ns=ns))
        found = []
        for stmt in statements:
            calls, _msgs, _seed = detector.analyze_code(stmt)
            found.extend(calls)
        return found

    DEF_FIRST = "def price(rng, n):\n    return rng.standard_normal(n)"
    DEF_SECOND = "def price(n, rng):\n    return rng.standard_normal(n)"

    def test_unseeded_rng_positional_arg_flows_into_param_draw(self):
        """The headline gap: an unseeded generator passed POSITIONALLY warns.

        Both slots are exercised, because a mapping that always reported the
        first parameter would pass a first-slot-only test while being wrong.
        """
        calls = self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            self.DEF_FIRST,
            "price(np.random.default_rng(), 5)",
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'rng'
        assert calls[0].function == 'standard_normal'
        assert calls[0].module == 'numpy.random'
        # Attributed to the CALL statement, where the unseeded argument is written.
        assert calls[0].lineno == 1

        # The generator in a LATER slot resolves through the signature too.
        calls = self._carriers(
            self._ns(self.DEF_SECOND),
            "import numpy as np",
            self.DEF_SECOND,
            "price(5, np.random.default_rng())",
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'rng'
        assert calls[0].function == 'standard_normal'

    def test_seeded_rng_positional_arg_does_not_warn(self):
        """Control: a seeded generator is reproducible, so the call site is quiet.

        This is what makes the test above a detector rather than a rubber stamp
        on every positional argument.
        """
        assert self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            self.DEF_FIRST,
            "price(np.random.default_rng(42), 5)",
        ) == []

    def test_keyword_arg_still_warns_with_a_shell_present(self):
        """Regression: CAS-154's keyword path is untouched by the positional one."""
        calls = self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            self.DEF_FIRST,
            "price(rng=np.random.default_rng(), n=5)",
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'rng'

    def test_session_carrier_passed_positionally_warns(self):
        """The other spelling of the same hazard: a session generator handed over
        positionally, rather than constructed inline at the call site."""
        calls = self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            "rng = np.random.default_rng()",
            self.DEF_FIRST,
            "price(rng, 5)",
        )
        assert len(calls) == 1
        assert calls[0].carrier == 'rng'
        # Seeded twin stays quiet.
        assert self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            "rng = np.random.default_rng(1)",
            self.DEF_FIRST,
            "price(rng, 5)",
        ) == []

    def test_positional_flow_silent_when_callee_never_scanned(self):
        """Conservative: nothing in the session says ``price`` draws off its
        first parameter, so a positional generator must not warn — even though
        the callee is right there in the namespace."""
        assert self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            "price(np.random.default_rng(), 5)",
        ) == []

    def test_positional_flow_silent_when_callee_absent_from_namespace(self):
        """Deliberate silence: without the live callee there is no signature, so
        the index cannot be named. A missing warning beats a guessed one."""
        assert self._carriers(
            {},  # empty user_ns: the def was scanned, but nothing is bound
            "import numpy as np",
            self.DEF_FIRST,
            "price(np.random.default_rng(), 5)",
        ) == []

    def test_positional_arg_landing_in_star_args_stays_silent(self):
        """``*args`` swallows the slot, so nothing can be said about it.

        ``price(100, default_rng())`` binds ``rng=100`` and drops the generator
        into ``rest``, where it is never drawn from — the silence is correct as
        well as conservative.
        """
        source = "def price(rng, *rest):\n    return rng.standard_normal(5)"
        assert self._carriers(
            self._ns(source),
            "import numpy as np",
            source,
            "price(100, np.random.default_rng())",
        ) == []

    def test_positional_args_after_an_unpacking_stay_silent(self):
        """``price(*cfg, rng)``: every index after ``*cfg`` depends on its runtime
        length, so no later slot can be named."""
        assert self._carriers(
            self._ns(self.DEF_FIRST),
            "import numpy as np",
            self.DEF_FIRST,
            "price(*cfg, np.random.default_rng())",
        ) == []

    def test_non_rng_positional_arg_stays_silent(self):
        """A positional argument that is not a generator must never warn."""
        ns = self._ns(self.DEF_FIRST)
        # A literal in the risky slot.
        assert self._carriers(ns, "import numpy as np", self.DEF_FIRST, "price(5)") == []
        # A bare name that is not a known carrier.
        assert self._carriers(ns, "import numpy as np", self.DEF_FIRST, "price(config)") == []
        # A generator in a slot the callee does not draw from.
        assert self._carriers(
            self._ns(self.DEF_SECOND),
            "import numpy as np",
            self.DEF_SECOND,
            "price(np.random.default_rng(), 5)",  # slot 0 is ``n``, not ``rng``
        ) == []

    def test_non_function_callee_stays_silent(self):
        """Only a plain function's signature is mapped: a class or a partial
        binds arguments in ways this mapping would have to guess at."""
        ns = self._ns(
            "class price:\n    def __init__(self, rng, n):\n        pass",
        )
        assert self._carriers(
            ns,
            "import numpy as np",
            self.DEF_FIRST,  # teaches the session that ``price`` draws off ``rng``
            "price(np.random.default_rng(), 5)",
        ) == []

    def test_positional_warning_names_the_parameter_and_the_escape_hatch(self):
        """The user-facing half: the warning has to reach a filter and a fix."""
        ns = self._ns(self.DEF_FIRST)
        detector = RandomnessDetector(shell=SimpleNamespace(user_ns=ns))
        detector.analyze_code("import numpy as np")
        detector.analyze_code(self.DEF_FIRST)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness("price(np.random.default_rng(), 5)", detector)
        assert len(w) == 1
        assert issubclass(w[0].category, CashRandomnessWarning)
        assert "rng.standard_normal()" in str(w[0].message)
        assert "@cash:allow-random" in str(w[0].message)

    def test_allow_random_suppresses_the_positional_warning(self):
        """The documented opt-out has to cover the newly-detected form."""
        ns = self._ns(self.DEF_FIRST)
        detector = RandomnessDetector(shell=SimpleNamespace(user_ns=ns))
        detector.analyze_code("import numpy as np")
        detector.analyze_code(self.DEF_FIRST)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness(
                "price(np.random.default_rng(), 5)", detector, suppress_warning=True,
            )
        assert len(w) == 0

    def test_randomness_never_changes_cacheability(self):
        """The advisory-only contract: detection must not make a statement
        uncacheable. ``has_seed_calls`` is the only cacheability signal this
        module emits, and a positional RNG argument is not a seed call."""
        detector = RandomnessDetector(shell=SimpleNamespace(user_ns=self._ns(self.DEF_FIRST)))
        detector.analyze_code("import numpy as np")
        detector.analyze_code(self.DEF_FIRST)
        _calls, _msgs, has_seed = detector.analyze_code("price(np.random.default_rng(), 5)")
        assert has_seed is False

    def test_np_random_seed_does_not_quiet_a_generator(self):
        """The module seed ledger must not reach across to a Generator.

        ``np.random.seed()`` seeds the legacy global singleton; a ``default_rng()``
        Generator is independent of it. Letting the ledger suppress this would
        reintroduce the exact silence CAS-135 is about.
        """
        assert len(self._carriers(
            "import numpy as np\nnp.random.seed(42)",
            "rng = np.random.default_rng()",
            "x = rng.normal()",
        )) == 1

    def test_bit_generator_seed_is_read_through(self):
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.Generator(np.random.PCG64(42))",
            "x = rng.normal()",
        ) == []
        assert len(self._carriers(
            "import numpy as np",
            "rng = np.random.Generator(np.random.PCG64())",
            "x = rng.normal()",
        )) == 1

    def test_randomstate_and_stdlib_random_carriers(self):
        assert len(self._carriers(
            "import numpy as np", "rs = np.random.RandomState()", "x = rs.rand(5)",
        )) == 1
        assert self._carriers(
            "import numpy as np", "rs = np.random.RandomState(0)", "x = rs.rand(5)",
        ) == []
        assert len(self._carriers(
            "import random", "r = random.Random()", "x = r.random()",
        )) == 1
        assert self._carriers(
            "import random", "r = random.Random(7)", "x = r.random()",
        ) == []

    def test_from_import_default_rng(self):
        assert len(self._carriers(
            "from numpy.random import default_rng",
            "rng = default_rng()",
            "x = rng.normal()",
        )) == 1

    def test_alias_carries_the_binding(self):
        assert len(self._carriers(
            "import numpy as np", "rng = np.random.default_rng()", "g = rng", "x = g.normal()",
        )) == 1

    def test_rebinding_to_a_non_carrier_forgets_it(self):
        """A name reused for an ordinary value must stop being an RNG."""
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng()",
            "rng = load_config()",
            "x = rng.normal()",
        ) == []

    def test_reseeding_quiets_the_carrier(self):
        assert self._carriers(
            "import numpy as np",
            "rng = np.random.default_rng()",
            "rng = np.random.default_rng(1)",
            "x = rng.normal()",
        ) == []

    def test_unknown_receiver_stays_silent(self):
        """``df.sample()`` is a documented gap, and must not become a false positive.

        'sample' is a draw name on both RNG modules, so it reaches the carrier
        check — the receiver simply isn't a known carrier, which is exactly why
        nothing is reported.
        """
        assert self._carriers("import pandas as pd", "y = df.sample(5)") == []
        assert self._carriers("y = thing.normal()") == []

    def test_reset_clears_carrier_bindings(self):
        detector = RandomnessDetector()
        detector.analyze_code("import numpy as np\nrng = np.random.default_rng()")
        assert 'rng' in detector.rng_carriers
        detector.reset()
        assert detector.rng_carriers == {}

    def test_carrier_warning_names_the_variable_and_the_escape_hatch(self):
        detector = RandomnessDetector()
        detector.analyze_code("import numpy as np\nrng = np.random.default_rng()")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness("x = rng.standard_normal(10)", detector)
        assert len(w) == 1
        assert issubclass(w[0].category, CashRandomnessWarning)
        message = str(w[0].message)
        assert "rng.standard_normal()" in message
        assert "@cash:allow-random" in message

    def test_allow_random_suppresses_carrier_warning(self):
        detector = RandomnessDetector()
        detector.analyze_code("import numpy as np\nrng = np.random.default_rng()")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_and_warn_randomness(
                "x = rng.standard_normal(10)", detector, suppress_warning=True,
            )
        assert len(w) == 0
