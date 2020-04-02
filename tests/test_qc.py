import pytest
import numpy as np

from . conftest import ground_states


@pytest.mark.parametrize('vqe_strategy', ['UCCSD', 'HF'])
@pytest.mark.parametrize('vqe_qc_backend', ['Nq-qvm'])
@pytest.mark.parametrize('vqe_parametricflag', [True, False])
def test_qc_parametric_flag(vqe_qc, vqe_strategy, vqe_qc_backend, vqe_parametricflag):
    ec = vqe_qc.objective_function()
    gs = vqe_qc.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_qc.strategy][0], atol=1e-5)
    assert np.isclose(ec, ground_states[vqe_qc.strategy][1], atol=2e-1)


@pytest.mark.parametrize('vqe_qc_backend', ['Nq-qvm'])
@pytest.mark.parametrize('vqe_parametricflag', [True])
def test_qc_custom_program(vqe_qc, vqe_qc_backend, vqe_parametricflag):
    ec = vqe_qc.objective_function()
    gs = vqe_qc.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_qc.strategy][0], atol=1e-5)
    assert np.isclose(ec, ground_states[vqe_qc.strategy][1], atol=2e-1)


@pytest.mark.parametrize('vqe_strategy', ['UCCSD', 'HF'])
@pytest.mark.parametrize('vqe_parametricflag', [False])
def test_qc_custom_program(vqe_qc, vqe_strategy, vqe_parametricflag):
    ec = vqe_qc.objective_function()
    gs = vqe_qc.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_qc.strategy][0], atol=1e-5)
    assert np.isclose(ec, ground_states[vqe_qc.strategy][1], atol=2e-1)
