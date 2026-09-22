"""
Model factory for FCF forecasting - ARIMA Models Only
"""

import logging

logger = logging.getLogger(__name__)


def build_statsforecast_model(model_name: str, params: dict):
    """Build a StatsForecast ARIMA model."""
    from statsforecast.models import ARIMA
    
    logger.info(f"Building StatsForecast model: {model_name}")
    
    if model_name.startswith('arima_'):
        return ARIMA(**params)
    else:
        raise ValueError(f"Unknown StatsForecast model: {model_name}")


def build_model(model_name: str):
    """
    Factory function to build a model based on name.
    Returns the model object ready for training.
    """
    from src.config import settings

    if model_name not in settings.MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(settings.MODELS.keys())}")

    config = settings.MODELS[model_name]
    model_type = config['type']
    params = config.get('params', {})

    logger.info(f"Building model: {model_name} (type: {model_type})")

    if model_type == 'statsforecast':
        return build_statsforecast_model(model_name, params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")