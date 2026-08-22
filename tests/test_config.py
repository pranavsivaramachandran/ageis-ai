import pytest
from pydantic import ValidationError
import aegis.core.config

def test_config_default_mode():
    config = aegis.core.config.AegisConfig()
    assert config.SYSTEM_MODE == "PREDICTION_ONLY"

def test_config_invalid_mode_raises_error():
    with pytest.raises(ValidationError):
        aegis.core.config.AegisConfig(SYSTEM_MODE="LIVE_TRADING")
