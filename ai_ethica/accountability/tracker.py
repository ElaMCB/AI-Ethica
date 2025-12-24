"""
Accountability Tracker Module

This module provides tools for tracking model decisions, maintaining audit trails,
and ensuring accountability in AI systems.
"""

import json
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd


class AccountabilityTracker:
    """
    A class for tracking model decisions and maintaining accountability.
    
    Features:
    - Decision logging
    - Audit trail maintenance
    - Model version tracking
    - Performance monitoring
    - Incident reporting
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the AccountabilityTracker.
        
        Parameters:
        -----------
        log_dir : str, optional
            Directory to store audit logs. If None, uses current directory.
        """
        self.log_dir = Path(log_dir) if log_dir else Path("audit_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.decisions = []
        self.incidents = []
    
    def log_decision(
        self,
        model_id: str,
        input_data: Any,
        prediction: Any,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a model decision for accountability.
        
        Parameters:
        -----------
        model_id : str
            Identifier for the model making the decision
        input_data : Any
            Input data that led to the decision
        prediction : Any
            The model's prediction/decision
        confidence : float, optional
            Confidence score if available
        metadata : Dict, optional
            Additional metadata about the decision
        
        Returns:
        --------
        str: Decision ID for reference
        """
        decision_id = f"decision_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        decision = {
            "decision_id": decision_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": model_id,
            "input_data": str(input_data) if not isinstance(input_data, (dict, list)) else input_data,
            "prediction": str(prediction) if not isinstance(prediction, (dict, list)) else prediction,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        self.decisions.append(decision)
        
        # Save to file
        self._save_decision(decision)
        
        return decision_id
    
    def log_incident(
        self,
        incident_type: str,
        description: str,
        severity: str = "medium",
        model_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log an incident for accountability and review.
        
        Parameters:
        -----------
        incident_type : str
            Type of incident (e.g., 'bias_detected', 'error', 'performance_degradation')
        description : str
            Description of the incident
        severity : str
            Severity level: 'low', 'medium', 'high', 'critical'
        model_id : str, optional
            Model ID associated with the incident
        decision_id : str, optional
            Decision ID associated with the incident
        metadata : Dict, optional
            Additional metadata
        
        Returns:
        --------
        str: Incident ID for reference
        """
        incident_id = f"incident_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        incident = {
            "incident_id": incident_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "incident_type": incident_type,
            "description": description,
            "severity": severity,
            "model_id": model_id,
            "decision_id": decision_id,
            "metadata": metadata or {},
            "status": "open"
        }
        
        self.incidents.append(incident)
        
        # Save to file
        self._save_incident(incident)
        
        return incident_id
    
    def get_audit_trail(
        self,
        model_id: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> pd.DataFrame:
        """
        Get audit trail of decisions.
        
        Parameters:
        -----------
        model_id : str, optional
            Filter by model ID
        start_date : datetime, optional
            Start date for filtering
        end_date : datetime, optional
            End date for filtering
        
        Returns:
        --------
        pd.DataFrame: Audit trail data
        """
        decisions = self.decisions.copy()
        
        if model_id:
            decisions = [d for d in decisions if d["model_id"] == model_id]
        
        if start_date:
            decisions = [d for d in decisions if datetime.datetime.fromisoformat(d["timestamp"]) >= start_date]
        
        if end_date:
            decisions = [d for d in decisions if datetime.datetime.fromisoformat(d["timestamp"]) <= end_date]
        
        return pd.DataFrame(decisions)
    
    def get_incidents(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get logged incidents.
        
        Parameters:
        -----------
        severity : str, optional
            Filter by severity level
        status : str, optional
            Filter by status ('open', 'resolved', 'closed')
        model_id : str, optional
            Filter by model ID
        
        Returns:
        --------
        pd.DataFrame: Incidents data
        """
        incidents = self.incidents.copy()
        
        if severity:
            incidents = [i for i in incidents if i["severity"] == severity]
        
        if status:
            incidents = [i for i in incidents if i["status"] == status]
        
        if model_id:
            incidents = [i for i in incidents if i.get("model_id") == model_id]
        
        return pd.DataFrame(incidents)
    
    def generate_report(
        self,
        model_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict:
        """
        Generate an accountability report.
        
        Parameters:
        -----------
        model_id : str, optional
            Model ID to generate report for
        period_days : int
            Number of days to include in report
        
        Returns:
        --------
        Dict: Accountability report
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=period_days)
        
        audit_trail = self.get_audit_trail(model_id=model_id, start_date=start_date, end_date=end_date)
        incidents = self.get_incidents(model_id=model_id)
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": period_days
            },
            "model_id": model_id or "all_models",
            "summary": {
                "total_decisions": len(audit_trail),
                "total_incidents": len(incidents),
                "open_incidents": len(incidents[incidents["status"] == "open"]) if len(incidents) > 0 else 0,
                "critical_incidents": len(incidents[incidents["severity"] == "critical"]) if len(incidents) > 0 else 0
            },
            "incidents_by_severity": {},
            "recommendations": []
        }
        
        if len(incidents) > 0:
            severity_counts = incidents["severity"].value_counts().to_dict()
            report["incidents_by_severity"] = severity_counts
        
        # Generate recommendations
        if report["summary"]["critical_incidents"] > 0:
            report["recommendations"].append(
                f"Address {report['summary']['critical_incidents']} critical incident(s) immediately"
            )
        
        if report["summary"]["open_incidents"] > 5:
            report["recommendations"].append(
                f"Review and resolve {report['summary']['open_incidents']} open incidents"
            )
        
        if not report["recommendations"]:
            report["recommendations"].append("No immediate action required")
        
        return report
    
    def _save_decision(self, decision: Dict):
        """Save decision to log file."""
        log_file = self.log_dir / f"decisions_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(decision) + '\n')
    
    def _save_incident(self, incident: Dict):
        """Save incident to log file."""
        log_file = self.log_dir / f"incidents_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(incident) + '\n')

