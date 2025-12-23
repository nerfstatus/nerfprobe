"""Storage module for NerfProbe results and baselines."""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from platformdirs import user_data_dir
from nerfprobe_core import ProbeResult

APP_NAME = "nerfprobe"
APP_AUTHOR = "nerfstatus"


def get_data_dir() -> Path:
    """Get the user data directory for nerfprobe."""
    data_dir = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class ResultStore:
    """Stores probe results in a JSONL file."""

    def __init__(self, path: Optional[Path] = None):
        self.data_dir = path or get_data_dir()
        self.results_file = self.data_dir / "results.jsonl"

    def append(self, result: ProbeResult):
        """Append a result to the store."""
        with open(self.results_file, "a", encoding="utf-8") as f:
            # Add timestamp only if not present (ProbeResult usually has it, 
            # but we ensure we are storing a complete record)
            data = result.model_dump(mode="json")
            if "timestamp" not in data: # Should be in metadata if anywhere, or we add wrap
                data["stored_at"] = datetime.now().isoformat()
            
            f.write(json.dumps(data) + "\n")

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent results."""
        if not self.results_file.exists():
            return []
        
        # Simple implementation: read all, return last N. 
        # For huge files, we'd want reverse reading.
        lines = []
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
            
        return [json.loads(line) for line in lines[-limit:]][::-1]

    def get_trends(self, model: str, limit: int = 100) -> dict[str, list[tuple[str, float]]]:
        """Get score trends for a model by probe."""
        if not self.results_file.exists():
            return {}
            
        trends: dict[str, list[tuple[str, float]]] = {}
        
        with open(self.results_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("target", {}).get("model_name") != model:
                        continue
                        
                    probe = data.get("probe_name")
                    score = data.get("score")
                    timestamp = data.get("stored_at") or datetime.now().isoformat() # Fallback
                    
                    if probe and score is not None:
                        if probe not in trends:
                            trends[probe] = []
                        trends[probe].append((timestamp, float(score)))
                except json.JSONDecodeError:
                    continue
                    
        return trends


class BaselineStore:
    """Stores baseline scores for models."""

    def __init__(self, path: Optional[Path] = None):
        self.data_dir = path or get_data_dir()
        self.baseline_file = self.data_dir / "baselines.json"

    def _load(self) -> dict[str, Any]:
        if not self.baseline_file.exists():
            return {}
        try:
            with open(self.baseline_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict[str, Any]):
        with open(self.baseline_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_baseline(self, model: str, results: list[ProbeResult]):
        """Calculate and save baseline from a list of results."""
        baselines = self._load()
        if model not in baselines:
            baselines[model] = {}
        
        # Group by probe
        grouped: dict[str, list[float]] = {}
        for r in results:
            if r.probe_name not in grouped:
                grouped[r.probe_name] = []
            grouped[r.probe_name].append(r.score)
            
        # Average
        for probe, scores in grouped.items():
            baselines[model][probe] = {
                "score": statistics.mean(scores),
                "samples": len(scores),
                "last_updated": datetime.now().isoformat()
            }
            
        self._save(baselines)

    def get_baseline_score(self, model: str, probe_name: str) -> Optional[float]:
        """Get baseline score for a specific probe."""
        baselines = self._load()
        return baselines.get(model, {}).get(probe_name, {}).get("score")

    def get_model_baselines(self, model: str) -> dict[str, Any]:
        """Get all baselines for a model."""
        baselines = self._load()
        return baselines.get(model, {})
