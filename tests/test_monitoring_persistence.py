import pytest
import os
import tempfile
from aegis.db.session import get_db
from aegis.db.models.base import Base
from aegis.governance.models import ChampionHealth
from aegis.events.contracts import AlertSeverity
from aegis.db.models.monitoring import (
    ReferenceProfileRecord, MonitoringPolicyRecord, 
    HealthAssessmentRecord, MonitoringAlertRecord
)

@pytest.fixture(scope="module", autouse=True)
def test_db():
    # Setup test DB if needed, but since we just want to test if it runs without errors, we can use the default test DB logic or sqlite in memory
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    
    yield Session
    
    Base.metadata.drop_all(bind=engine)

def test_persistence_reference_profile(test_db):
    session = test_db()
    
    ref = ReferenceProfileRecord(
        id="ref1",
        champion_identity="champ1",
        champion_version=1,
        experiment_identity="exp1",
        reference_window_identity="rw1",
        profile_data='{"test": 1}'
    )
    session.add(ref)
    session.commit()
    
    loaded = session.query(ReferenceProfileRecord).filter_by(id="ref1").first()
    assert loaded is not None
    assert loaded.champion_identity == "champ1"

def test_persistence_health_assessment(test_db):
    session = test_db()
    
    assessment = HealthAssessmentRecord(
        id="ha1",
        champion_identity="champ1",
        champion_version=1,
        observation_identity="obs1",
        reference_identity="ref1",
        policy_identity="pol1",
        state=ChampionHealth.HEALTHY,
        reasons='["ok"]',
        drift_report='{}'
    )
    
    alert = MonitoringAlertRecord(
        id="alert1",
        assessment_identity="ha1",
        severity=AlertSeverity.WARNING,
        category="DATA_DRIFT",
        metric="f1_mean",
        direction="ABOVE",
        champion_identity="champ1",
        policy_identity="pol1",
        reason="too high"
    )
    
    session.add(assessment)
    session.add(alert)
    session.commit()
    
    loaded = session.query(HealthAssessmentRecord).filter_by(id="ha1").first()
    assert loaded.state == ChampionHealth.HEALTHY
    
    loaded_alert = session.query(MonitoringAlertRecord).filter_by(id="alert1").first()
    assert loaded_alert.severity == AlertSeverity.WARNING
