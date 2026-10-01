"""
Customer Churn and Behaviour Prediction Package
"""

import sys
import types

# Ensure SHAP can run on environments where Numba compiled DLLs are restricted by Windows AppControl
if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401
    except Exception:
        class _MockMod(types.ModuleType):
            def __getattr__(self, name):
                m = _MockMod(f"{self.__name__}.{name}")
                m.__path__ = []
                m.njit = lambda *a, **kw: (lambda f: f)
                m.jit = lambda *a, **kw: (lambda f: f)
                return m

        _m = _MockMod("numba")
        _m.__path__ = []
        _m.njit = lambda *a, **kw: (lambda f: f)
        _m.jit = lambda *a, **kw: (lambda f: f)
        sys.modules["numba"] = _m
        sys.modules["numba.typed"] = _MockMod("numba.typed")
        sys.modules["numba.core"] = _MockMod("numba.core")
        sys.modules["numba.core.types"] = _MockMod("numba.core.types")
        sys.modules["numba.experimental"] = _MockMod("numba.experimental")
