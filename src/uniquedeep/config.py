# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/config.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: Configuration management for UniqueDeep.
'''

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (override=True ensures .env file overrides system environment variables)
load_dotenv(override=True)

def load_models_config() -> dict:
    """Load model configuration from models.json"""
    try:
        # Find models.json
        current_dir = Path.cwd()
        # Assuming this file is in src/uniquedeep/, root is ../../
        root_dir = Path(__file__).parent.parent.parent
        
        paths = [
            current_dir / "models.json",
            root_dir / "models.json",
        ]
        
        config_path = None
        for p in paths:
            if p.exists():
                config_path = p
                break
        
        if not config_path:
            return {}
            
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[yellow]! Failed to load models.json: {e}[/yellow]")
        return {}

def save_models_config(config: dict):
    """Save model configuration to models.json"""
    try:
        current_dir = Path.cwd()
        root_dir = Path(__file__).parent.parent.parent
        
        paths = [
            current_dir / "models.json",
            root_dir / "models.json",
        ]
        
        config_path = None
        for p in paths:
            if p.exists():
                config_path = p
                break
                
        if not config_path:
            config_path = current_dir / "models.json"
            
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[red]! Failed to save models.json: {e}[/red]")

def get_flattened_models(config: dict) -> list:
    """Flatten nested providers configuration into a list of models"""
    models = []
    providers = config.get("providers", {})
    for provider_key, provider_data in providers.items():
        for model in provider_data.get("models", []):
            model_info = model.copy()
            model_info["provider"] = provider_key
            model_info["provider_display"] = provider_key.title()
            # ZhipuAI special handling
            if provider_key == "zhipuai":
                model_info["provider_display"] = "ZhipuAI"
            models.append(model_info)
    return models

def check_api_credentials() -> bool:
    """Check if API credentials are set"""
    # Simple check for now, can be expanded to check specific provider keys based on active model
    # This logic is partly duplicated in agent.py's get_model_config, 
    # but for CLI startup check, checking any key is often enough or we can rely on agent's check.
    
    # We can check for any common API key
    keys = [
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ZHIPUAI_API_KEY", "GLM_API_KEY",
        "MOONSHOT_API_KEY",
        "DOUBAO_API_KEY"
    ]
    
    # Also check if models.json has keys (though we don't parse it deep here to check validity)
    # If models.json exists and has active_model, we assume it might be configured.
    config = load_models_config()
    if config.get("providers"):
        return True

    return any(os.getenv(key) for key in keys)
