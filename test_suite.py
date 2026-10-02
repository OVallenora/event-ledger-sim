import pytest
import concurrent.futures
from model import SpatialRiskDetector, MonteCarloRiskEngine, TransactionEvent
from ledger import ThreadSafeLedger


def test_spatial_risk_detector():
    detector = SpatialRiskDetector(k=1, max_radius_deg=0.2)
    # Historical transactions around Central London
    detector.fit([(51.5074, -0.1278), (51.5080, -0.1285)])

    # Nearby point (passes)
    assert detector.is_anomalous(51.5075, -0.1279) is False
    # Far outlier (fails)
    assert detector.is_anomalous(40.7128, -74.0060) is True


def test_monte_carlo_risk_engine():
    engine = MonteCarloRiskEngine(simulation_loops=2000)
    flagged = [
        TransactionEvent("A", "B", 100.0, 0.0, 0.0),
        TransactionEvent("C", "D", 200.0, 0.0, 0.0)
    ]
    # Expected: (100 + 200) * 0.50 = 150.0 approx
    estimated_loss = engine.compute_gross_weighted_loss(flagged, default_fraud_prob=0.50)
    assert 130.0 <= estimated_loss <= 170.0


def test_insufficient_funds_rejection():
    ledger = ThreadSafeLedger()
    detector = SpatialRiskDetector()
    ledger.create_account("user_1", 50.0)
    ledger.create_account("user_2", 10.0)

    event = TransactionEvent("user_1", "user_2", 100.0, 0.0, 0.0)
    success = ledger.process_transaction(event, detector)

    assert success is False
    assert ledger.get_balance("user_1") == 50.0
    assert ledger.get_balance("user_2") == 10.0


def test_concurrent_transfers_no_race_conditions():
    """
    Simulates 20 concurrent threads trying to drain an account simultaneously.
    Total balance must remain strictly non-negative and consistent.
    """
    ledger = ThreadSafeLedger()
    detector = SpatialRiskDetector()
    ledger.create_account("vault", 100.0)
    ledger.create_account("receiver", 0.0)

    event = TransactionEvent("vault", "receiver", 10.0, 0.0, 0.0)

    # 20 concurrent transfer attempts of £10 from a £100 balance
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(ledger.process_transaction, event, detector) for _ in range(20)]
        results = [f.result() for f in futures]

    # Exactly 10 should succeed, and 10 should fail due to insufficient funds
    assert results.count(True) == 10
    assert results.count(False) == 10
    assert ledger.get_balance("vault") == 0.0
    assert ledger.get_balance("receiver") == 100.0
