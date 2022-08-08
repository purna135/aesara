import itertools

import numpy as np
import pytest
import scipy

import aesara
from aesara import function, grad
from aesara import tensor as at
from aesara.configdefaults import config
from aesara.tensor.slinalg import (
    Cholesky,
    CholeskySolve,
    Solve,
    SolveBase,
    SolveTriangular,
    cholesky,
    eigvalsh,
    expm,
    kron,
    solve,
    solve_triangular,
)
from aesara.tensor.type import dmatrix, matrix, tensor, tensor3
from tests import unittest_tools as utt


class TestCholesky(utt.InferShapeTester):
    def check_lower_triangular(self, pd, ch_f):
        ch = ch_f(pd)
        tril_inds1, tril_inds2 = np.tril_indices(ch.shape[-1], k=0)
        triu_inds1, triu_inds2 = np.triu_indices(ch.shape[-1], k=1)
        assert np.all(ch[..., triu_inds1, triu_inds2] == 0)
        assert np.any(ch[..., tril_inds1, tril_inds2] != 0)
        assert np.allclose(ch @ np.moveaxis(ch, -1, -2), pd)
        assert not np.allclose(np.moveaxis(ch, -1, -2) @ ch, pd)

    def check_upper_triangular(self, pd, ch_f):
        ch = ch_f(pd)
        tril_inds1, tril_inds2 = np.tril_indices(ch.shape[-1], k=-1)
        triu_inds1, triu_inds2 = np.triu_indices(ch.shape[-1], k=0)
        assert np.all(ch[..., tril_inds1, tril_inds2] == 0)
        assert np.any(ch[..., triu_inds1, triu_inds2] != 0)
        assert np.allclose(np.moveaxis(ch, -1, -2) @ ch, pd)
        assert not np.allclose(ch @ np.moveaxis(ch, -1, -2), pd)

    @pytest.mark.parametrize(
        "a_shape",
        [(5, 5), (2, 5, 5), (10, 3, 3)],
    )
    def test_cholesky(self, a_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        r = rng.standard_normal((a_shape)).astype(config.floatX)
        pd = r @ np.swapaxes(r, -1, -2)
        x = matrix() if len(a_shape) == 2 else tensor3()
        chol = cholesky(x)
        # Check the default.
        ch_f = function([x], chol)
        self.check_lower_triangular(pd, ch_f)
        # Explicit lower-triangular.
        chol = Cholesky(lower=True)(x)
        ch_f = function([x], chol)
        self.check_lower_triangular(pd, ch_f)
        # Explicit upper-triangular.
        chol = Cholesky(lower=False)(x)
        ch_f = function([x], chol)
        self.check_upper_triangular(pd, ch_f)
        chol = Cholesky(lower=False, on_error="nan")(x)
        ch_f = function([x], chol)
        self.check_upper_triangular(pd, ch_f)

    def test_cholesky_indef(self):
        x = matrix()
        mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)
        cholesky = Cholesky(lower=True, on_error="raise")
        chol_f = function([x], cholesky(x))
        with pytest.raises(scipy.linalg.LinAlgError):
            chol_f(mat)
        cholesky = Cholesky(lower=True, on_error="nan")
        chol_f = function([x], cholesky(x))
        assert np.all(np.isnan(chol_f(mat)))

    @pytest.mark.parametrize(
        "a_shape",
        [(5, 5), (2, 5, 5), (1, 2, 3, 3)],
    )
    def test_cholesky_grad(self, a_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        r = rng.standard_normal(a_shape).astype(config.floatX)
        pd = r @ np.swapaxes(r, -1, -2)
        # The dots are inside the graph since Cholesky needs separable matrices

        # Check the default.
        utt.verify_grad(cholesky, [pd], 3, rng)
        # Explicit lower-triangular.
        utt.verify_grad(
            Cholesky(lower=True),
            [pd],
            3,
            rng,
            abs_tol=0.05,
            rel_tol=0.05,
        )

        # Explicit upper-triangular.
        utt.verify_grad(
            Cholesky(lower=False),
            [pd],
            3,
            rng,
            abs_tol=0.05,
            rel_tol=0.05,
        )

    def test_cholesky_grad_indef(self):
        x = matrix()
        mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)
        cholesky = Cholesky(lower=True, on_error="raise")
        chol_f = function([x], grad(cholesky(x).sum(), [x]))
        with pytest.raises(scipy.linalg.LinAlgError):
            chol_f(mat)
        cholesky = Cholesky(lower=True, on_error="nan")
        chol_f = function([x], grad(cholesky(x).sum(), [x]))
        assert np.all(np.isnan(chol_f(mat)))

    @pytest.mark.slow
    def test_cholesky_and_cholesky_grad_shape(self):
        rng = np.random.default_rng(utt.fetch_seed())
        x = matrix()
        for l in (cholesky(x), Cholesky(lower=True)(x), Cholesky(lower=False)(x)):
            f_chol = aesara.function([x], l.shape)
            g = aesara.gradient.grad(l.sum(), x)
            f_cholgrad = aesara.function([x], g.shape)
            topo_chol = f_chol.maker.fgraph.toposort()
            topo_cholgrad = f_cholgrad.maker.fgraph.toposort()
            if config.mode != "FAST_COMPILE":
                assert sum(node.op.__class__ == Cholesky for node in topo_chol) == 0
            for shp in [2, 3, 5]:
                m = np.cov(rng.standard_normal((shp, shp + 10))).astype(config.floatX)
                np.testing.assert_equal(f_chol(m), (shp, shp))
                np.testing.assert_equal(f_cholgrad(m), (shp, shp))

    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.int32, np.complex64],
    )
    def test_cholesky_dtype(self, dtype):
        a = np.eye(2, dtype=dtype)
        assert scipy.linalg.cholesky(a).dtype == cholesky(a).eval().dtype


def test_eigvalsh():
    A = dmatrix("a")
    B = dmatrix("b")
    f = function([A, B], eigvalsh(A, B))

    rng = np.random.default_rng(utt.fetch_seed())
    a = rng.standard_normal((5, 5))
    a = a + a.T
    for b in [10 * np.eye(5, 5) + rng.standard_normal((5, 5))]:
        w = f(a, b)
        refw = scipy.linalg.eigvalsh(a, b)
        np.testing.assert_array_almost_equal(w, refw)

    # We need to test None separately, as otherwise DebugMode will
    # complain, as this isn't a valid ndarray.
    b = None
    B = at.NoneConst
    f = function([A], eigvalsh(A, B))
    w = f(a)
    refw = scipy.linalg.eigvalsh(a, b)
    np.testing.assert_array_almost_equal(w, refw)


def test_eigvalsh_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    a = rng.standard_normal((5, 5))
    a = a + a.T
    b = 10 * np.eye(5, 5) + rng.standard_normal((5, 5))
    utt.verify_grad(
        lambda a, b: eigvalsh(a, b).dot([1, 2, 3, 4, 5]), [a, b], rng=np.random
    )


class TestSolveBase(utt.InferShapeTester):
    def test__repr__(self):
        np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        y = SolveBase()(A, b)
        assert y.__repr__() == "SolveBase{lower=False, check_finite=True}.0"


class TestSolve(utt.InferShapeTester):
    def test__init__(self):
        with pytest.raises(ValueError) as excinfo:
            Solve(assume_a="test")
        assert "is not a recognized matrix structure" in str(excinfo.value)

    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((5, 5), (5,)),
            ((5, 5), (5, 1)),
            ((2, 5, 5), (1, 5)),
            ((3, 5, 5), (1, 5, 3)),
        ],
    )
    def test_infer_shape(self, a_shape, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix() if len(a_shape) == 2 else tensor3()
        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = at.as_tensor_variable(b_val).type()
        self._compile_and_check(
            [A, b],
            [solve(A, b)],
            [
                np.asarray(rng.random(a_shape), dtype=config.floatX),
                b_val,
            ],
            Solve,
            warn=False,
        )

    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((5, 5), (5,)),
            ((5, 5), (5, 1)),
            ((2, 5, 5), (1, 5)),
            ((3, 5, 5), (1, 5, 3)),
        ],
    )
    def test_correctness(self, a_shape, b_shape):
        signature = "(m,m),(m,k)->(m,k)"
        if len(b_shape) == len(a_shape) - 1:
            signature = "(m,m),(m)->(m)"

        solve_vfunc = np.vectorize(
            scipy.linalg.solve,
            excluded={"lower", "check_finite", "assume_a"},
            signature=signature,
        )

        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix() if len(a_shape) == 2 else tensor3()
        A_val = np.asarray(rng.random(a_shape), dtype=config.floatX)
        A_val = A_val @ np.swapaxes(A_val, -1, -2)
        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = at.as_tensor_variable(b_val).type()

        y = solve(A, b)
        gen_solve_func = aesara.function([A, b], y)

        assert np.allclose(solve_vfunc(A_val, b_val), gen_solve_func(A_val, b_val))

    @pytest.mark.parametrize(
        "a_shape, b_shape, assume_a, lower",
        [
            ((2, 2), (2,), "gen", False),
            ((2, 2), (2,), "gen", True),
            ((3, 3), (3, 1), "gen", True),
            ((2, 3, 3), (2, 3), "gen", False),
            ((2, 3, 3), (1, 3), "gen", False),
            ((3, 5, 5), (1, 5, 3), "gen", True),
        ],
    )
    def test_solve_grad(self, a_shape, b_shape, assume_a, lower):
        rng = np.random.default_rng(utt.fetch_seed())

        # Ensure diagonal elements of `A` are relatively large to avoid
        # numerical precision issues
        A_val = (rng.normal(size=a_shape) * 0.5 + np.eye(a_shape[-1])).astype(
            config.floatX
        )
        b_val = rng.normal(size=b_shape).astype(config.floatX)

        eps = None
        if config.floatX == "float64":
            eps = 2e-8

        solve_op = Solve(assume_a=assume_a, lower=lower)
        utt.verify_grad(solve_op, [A_val, b_val], 3, rng, eps=eps)


class TestSolveTriangular(utt.InferShapeTester):
    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((5, 5), (5,)),
            ((5, 5), (5, 1)),
            ((2, 5, 5), (1, 5)),
            ((3, 5, 5), (1, 5, 3)),
        ],
    )
    def test_infer_shape(self, a_shape, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix() if len(a_shape) == 2 else tensor3()
        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = at.as_tensor_variable(b_val).type()
        self._compile_and_check(
            [A, b],
            [solve_triangular(A, b)],
            [
                np.asarray(rng.random((a_shape)), dtype=config.floatX),
                b_val,
            ],
            SolveTriangular,
            warn=False,
        )

    @pytest.mark.parametrize(
        "a_shape, b_shape, lower",
        [
            ((5, 5), (5,), True),
            ((5, 5), (5, 1), False),
            ((2, 5, 5), (1, 5), True),
            ((3, 5, 5), (1, 5, 3), False),
        ],
    )
    def test_correctness(self, a_shape, b_shape, lower):
        cholesky_vfunc = np.vectorize(
            scipy.linalg.cholesky,
            excluded={"lower"},
            signature="(m,m)->(m,m)",
        )

        signature = "(m,m),(m,k)->(m,k)"
        if len(b_shape) == len(a_shape) - 1:
            signature = "(m,m),(m)->(m)"

        solve_triangular_vfunc = np.vectorize(
            scipy.linalg.solve_triangular,
            excluded={"lower", "trans", "unit_diagonal", "check_finite"},
            signature=signature,
        )

        rng = np.random.default_rng(utt.fetch_seed())

        A = matrix() if len(a_shape) == 2 else tensor3()
        A_val = np.asarray(rng.random(a_shape), dtype=config.floatX)
        A_val = A_val @ np.swapaxes(A_val, -1, -2)
        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = at.as_tensor_variable(b_val).type()

        C_val = cholesky_vfunc(A_val, lower=lower)

        cholesky = Cholesky(lower=lower)
        C = cholesky(A)
        y_lower = solve_triangular(C, b, lower=lower)
        lower_solve_func = aesara.function([C, b], y_lower)

        assert np.allclose(
            solve_triangular_vfunc(C_val, b_val, lower=lower),
            lower_solve_func(C_val, b_val),
        )

    @pytest.mark.parametrize(
        "a_shape, b_shape, lower",
        [
            ((2, 2), (2,), False),
            ((2, 2), (2,), True),
            ((3, 3), (3, 1), True),
            ((2, 3, 3), (2, 3), False),
            ((2, 3, 3), (1, 3), False),
            ((3, 5, 5), (1, 5, 3), True),
        ],
    )
    def test_solve_grad(self, a_shape, b_shape, lower):
        rng = np.random.default_rng(utt.fetch_seed())

        # Ensure diagonal elements of `A` are relatively large to avoid
        # numerical precision issues
        A_val = (rng.normal(size=a_shape) * 0.5 + np.eye(a_shape[-1])).astype(
            config.floatX
        )
        b_val = rng.normal(size=b_shape).astype(config.floatX)

        eps = None
        if config.floatX == "float64":
            eps = 2e-8

        solve_op = SolveTriangular(lower=lower)
        utt.verify_grad(solve_op, [A_val, b_val], 3, rng, eps=eps)


class TestCholeskySolve(utt.InferShapeTester):
    def setup_method(self):
        self.op_class = CholeskySolve
        self.op = CholeskySolve()
        self.op_upper = CholeskySolve(lower=False)
        super().setup_method()

    def test_repr(self):
        assert repr(CholeskySolve()) == "CholeskySolve{(True, True)}"

    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((5, 5), (5,)),
            ((5, 5), (5, 1)),
            ((2, 5, 5), (1, 5)),
            ((3, 5, 5), (1, 5, 3)),
        ],
    )
    def test_infer_shape(self, a_shape, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix() if len(a_shape) == 2 else tensor3()
        A_val = np.asarray(rng.random(a_shape), dtype=config.floatX)

        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = at.as_tensor_variable(b_val).type()

        self._compile_and_check(
            [A, b],  # aesara.function inputs
            [self.op(A, b)],  # aesara.function outputs
            # A must be square
            [A_val, b_val],
            self.op_class,
            warn=False,
        )

    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((5, 5), (5,)),
            ((5, 5), (5, 1)),
            ((2, 5, 5), (1, 5)),
            ((3, 5, 5), (1, 5, 3)),
        ],
    )
    def test_solve_correctness(self, a_shape, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix() if len(a_shape) == 2 else tensor3()
        A_val = np.tril(np.asarray(rng.random(a_shape), dtype=config.floatX))

        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = at.as_tensor_variable(b_val).type()

        y = self.op(A, b)
        cho_solve_lower_func = aesara.function([A, b], y)

        y = self.op_upper(A, b)
        cho_solve_upper_func = aesara.function([A, b], y)

        # TODO: vectorize cholesky solve
        assert np.allclose(
            scipy.linalg.cho_solve((A_val, True), b_val),
            cho_solve_lower_func(A_val, b_val),
        )

        A_val = np.triu(np.asarray(rng.random(a_shape), dtype=config.floatX))
        assert np.allclose(
            scipy.linalg.cho_solve((A_val, False), b_val),
            cho_solve_upper_func(A_val, b_val),
        )

    def test_solve_dtype(self):
        dtypes = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

        A_val = np.eye(2)
        b_val = np.ones((2, 1))

        # try all dtype combinations
        for A_dtype, b_dtype in itertools.product(dtypes, dtypes):
            A = matrix(dtype=A_dtype)
            b = matrix(dtype=b_dtype)
            x = self.op(A, b)
            fn = function([A, b], x)
            x_result = fn(A_val.astype(A_dtype), b_val.astype(b_dtype))

            assert x.dtype == x_result.dtype


# TODO: Remove this test, this test is already cover in TestCholeskySolve
# def test_cho_solve():
#     rng = np.random.default_rng(utt.fetch_seed())
#     A = matrix()
#     b = matrix()
#     y = cho_solve((A, True), b)
#     cho_solve_lower_func = aesara.function([A, b], y)

#     b_val = np.asarray(rng.random((5, 1)), dtype=config.floatX)

#     A_val = np.tril(np.asarray(rng.random((5, 5)), dtype=config.floatX))

#     assert np.allclose(
#         scipy.linalg.cho_solve((A_val, True), b_val),
#         cho_solve_lower_func(A_val, b_val),
#     )


def test_expm():
    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.standard_normal((5, 5)).astype(config.floatX)

    ref = scipy.linalg.expm(A)

    x = matrix()
    m = expm(x)
    expm_f = function([x], m)

    val = expm_f(A)
    np.testing.assert_array_almost_equal(val, ref)


def test_expm_grad_1():
    # with symmetric matrix (real eigenvectors)
    rng = np.random.default_rng(utt.fetch_seed())
    # Always test in float64 for better numerical stability.
    A = rng.standard_normal((5, 5))
    A = A + A.T

    utt.verify_grad(expm, [A], rng=rng)


def test_expm_grad_2():
    # with non-symmetric matrix with real eigenspecta
    rng = np.random.default_rng(utt.fetch_seed())
    # Always test in float64 for better numerical stability.
    A = rng.standard_normal((5, 5))
    w = rng.standard_normal((5)) ** 2
    A = (np.diag(w**0.5)).dot(A + A.T).dot(np.diag(w ** (-0.5)))
    assert not np.allclose(A, A.T)

    utt.verify_grad(expm, [A], rng=rng)


def test_expm_grad_3():
    # with non-symmetric matrix (complex eigenvectors)
    rng = np.random.default_rng(utt.fetch_seed())
    # Always test in float64 for better numerical stability.
    A = rng.standard_normal((5, 5))

    utt.verify_grad(expm, [A], rng=rng)


class TestKron(utt.InferShapeTester):

    rng = np.random.default_rng(43)

    def setup_method(self):
        self.op = kron
        super().setup_method()

    def test_perform(self):
        for shp0 in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            x = tensor(dtype="floatX", shape=(False,) * len(shp0))
            a = np.asarray(self.rng.random(shp0)).astype(config.floatX)
            for shp1 in [(6,), (6, 7), (6, 7, 8), (6, 7, 8, 9)]:
                if len(shp0) + len(shp1) == 2:
                    continue
                y = tensor(dtype="floatX", shape=(False,) * len(shp1))
                f = function([x, y], kron(x, y))
                b = self.rng.random(shp1).astype(config.floatX)
                out = f(a, b)
                # Newer versions of scipy want 4 dimensions at least,
                # so we have to add a dimension to a and flatten the result.
                if len(shp0) + len(shp1) == 3:
                    scipy_val = scipy.linalg.kron(a[np.newaxis, :], b).flatten()
                else:
                    scipy_val = scipy.linalg.kron(a, b)
                utt.assert_allclose(out, scipy_val)

    def test_numpy_2d(self):
        for shp0 in [(2, 3)]:
            x = tensor(dtype="floatX", shape=(False,) * len(shp0))
            a = np.asarray(self.rng.random(shp0)).astype(config.floatX)
            for shp1 in [(6, 7)]:
                if len(shp0) + len(shp1) == 2:
                    continue
                y = tensor(dtype="floatX", shape=(False,) * len(shp1))
                f = function([x, y], kron(x, y))
                b = self.rng.random(shp1).astype(config.floatX)
                out = f(a, b)
                assert np.allclose(out, np.kron(a, b))
