"""API Integration and Parity Test Suite for Argus Platform."""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.config import FEATURE_COLS, MODELS_DIR


@pytest.fixture(scope='module')
def client():
    """Fixture providing a TestClient with lifespan artifacts loaded."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_transaction_payload():
    """Fixture providing a realistic transaction payload."""
    features = {f'V{i}': float(i * 0.1 - 1.4) for i in range(1, 29)}
    features['Time'] = 43200.0
    features['Amount'] = 149.50
    return features


def test_score_returns_probability_in_range(client, sample_transaction_payload):
    """Test 1: Verifies /score returns a valid probability, decision flag, and threshold."""
    response = client.post('/score', json=sample_transaction_payload)
    assert response.status_code == 200, f'Expected 200, got {response.status_code}: {response.text}'

    data = response.json()
    assert 'fraud_probability' in data
    assert 0.0 <= data['fraud_probability'] <= 1.0
    assert data['decision'] in {'flag', 'allow'}
    assert 'threshold' in data
    assert data['threshold'] == pytest.approx(0.12587, abs=1e-3)
    assert 'model_version' in data
    assert 'smote_lgbm' in data['model_version']


def test_score_rejects_malformed_payload(client, sample_transaction_payload):
    """Test 2: Verifies /score rejects missing features with HTTP 422 (not 500)."""
    # Remove critical feature V14
    malformed = dict(sample_transaction_payload)
    del malformed['V14']

    response = client.post('/score', json=malformed)
    assert response.status_code == 422, f'Expected 422 Unprocessable Entity, got {response.status_code}'
    
    data = response.json()
    assert 'detail' in data
    assert any('V14' in str(err) for err in data.get('errors', []))


def test_score_matches_direct_pipeline_call(client, sample_transaction_payload):
    """Test 3: PARITY GUARD -- Verifies API probability matches direct pipeline execution to 6 decimal places."""
    # 1. Score through API
    response = client.post('/score', json=sample_transaction_payload)
    assert response.status_code == 200
    api_probability = response.json()['fraud_probability']

    # 2. Score directly through loaded training artifact
    pipeline_path = MODELS_DIR / 'fraud_pipeline.joblib'
    direct_pipeline = joblib.load(pipeline_path)

    df_row = pd.DataFrame([sample_transaction_payload])[FEATURE_COLS]
    direct_probability = float(direct_pipeline.predict_proba(df_row)[0, 1])

    # 3. Assert exact mathematical parity (no serving/training skew)
    assert api_probability == pytest.approx(direct_probability, abs=1e-6), (
        f'Serving/Training skew detected: API={api_probability:.8f} vs Direct={direct_probability:.8f}'
    )


def test_ask_returns_structured_citations(client):
    """Test 4: Verifies /ask returns synthesized answer with structured source citations."""
    payload = {
        'question': "What were Apple's total net sales in fiscal 2025?",
        'config': 'section_dense',
        'top_k': 3,
    }
    response = client.post('/ask', json=payload)
    assert response.status_code == 200, f'Expected 200, got {response.status_code}: {response.text}'

    data = response.json()
    assert 'answer' in data
    assert len(data['answer']) > 10
    assert data['abstained'] is False
    assert 'citations' in data
    assert len(data['citations']) > 0

    first_citation = data['citations'][0]
    for key in ['chunk_id', 'ticker', 'fiscal_year', 'section', 'text']:
        assert key in first_citation, f'Missing citation key: {key}'
        assert first_citation[key] is not None


def test_ask_sets_abstention_flag_on_unanswerable(client):
    """Test 5: Verifies /ask explicitly sets abstained=True on unanswerable out-of-corpus queries."""
    payload = {
        'question': "What was Netflix's subscriber count in Antarctica during FY2024?",
        'config': 'section_dense',
    }
    response = client.post('/ask', json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data['abstained'] is True
    assert 'cannot be verified' in data['answer'] or 'insufficient' in data['answer'].lower()


def test_health_reports_loaded_artifacts(client):
    """Test 6: Verifies /health reports operational status, uptime, and loaded artifact metadata."""
    response = client.get('/health')
    assert response.status_code == 200

    data = response.json()
    assert data['status'] == 'ok'
    assert data['uptime_seconds'] >= 0.0

    # Fraud model provenance
    assert data['fraud_model']['loaded'] is True
    assert 'smote_lgbm' in data['fraud_model']['version']
    assert len(data['fraud_model']['artifact_hash']) > 0

    # RAG vector store provenance
    assert data['rag_index']['loaded'] is True
    assert data['rag_index']['chunks'] == 13467
    assert data['rag_index']['embedding_model'] == 'all-MiniLM-L6-v2'
