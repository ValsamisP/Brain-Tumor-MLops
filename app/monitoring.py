"""
Monitoring Module
Track predictions, performance metrics, and system health
"""
import json
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List
import threading

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and track API metrics"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.lock = threading.Lock()
        
        # Metrics storage
        self.total_predictions = 0
        self.total_errors = 0
        self.class_distribution = defaultdict(int)
        self.confidence_scores = deque(maxlen=max_history)
        self.processing_times = deque(maxlen=max_history)
        self.predictions_history = deque(maxlen=max_history)
        
        # Time-based metrics
        self.start_time = datetime.now()
        self.last_prediction_time = None
    
    def record_prediction(self, predicted_class: str, confidence: float, processing_time: float):
        """Record a successful prediction"""
        with self.lock:
            self.total_predictions += 1
            self.class_distribution[predicted_class] += 1
            self.confidence_scores.append(confidence)
            self.processing_times.append(processing_time)
            self.last_prediction_time = datetime.now()
            
            # Store prediction details
            self.predictions_history.append({
                'class': predicted_class,
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'timestamp': self.last_prediction_time.isoformat()
            })
    
    def record_error(self):
        """Record an error"""
        with self.lock:
            self.total_errors += 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics summary"""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            metrics = {
                'total_predictions': self.total_predictions,
                'total_errors': self.total_errors,
                'uptime_seconds': uptime,
                'predictions_per_minute': (self.total_predictions / uptime) * 60 if uptime > 0 else 0,
                'error_rate': self.total_errors / max(self.total_predictions, 1),
                'class_distribution': dict(self.class_distribution),
                'average_confidence': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
                'average_processing_time_ms': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                'min_processing_time_ms': min(self.processing_times) if self.processing_times else 0,
                'max_processing_time_ms': max(self.processing_times) if self.processing_times else 0,
                'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None
            }
            
            return metrics
    
    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Get n most recent predictions"""
        with self.lock:
            return list(self.predictions_history)[-n:]


def log_prediction(prediction_id: str, predicted_class: str, confidence: float, processing_time_ms: float):
    """
    Log prediction to file for monitoring and analysis
    
    Args:
        prediction_id: Unique prediction identifier
        predicted_class: Predicted tumor class
        confidence: Prediction confidence score
        processing_time_ms: Processing time in milliseconds
    """
    log_entry = {
        'prediction_id': prediction_id,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'processing_time_ms': processing_time_ms,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Log to file (create logs directory if needed)
        import os
        log_dir = '/app/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self):
        self.baseline_distribution = None
        self.current_distribution = defaultdict(int)
        self.drift_threshold = 0.15  # 15% drift threshold
        
    def set_baseline(self, distribution: Dict[str, int]):
        """Set baseline class distribution"""
        total = sum(distribution.values())
        self.baseline_distribution = {
            k: v / total for k, v in distribution.items()
        }
    
    def update_current(self, predicted_class: str):
        """Update current distribution"""
        self.current_distribution[predicted_class] += 1
    
    def check_drift(self) -> Dict:
        """Check for distribution drift"""
        if not self.baseline_distribution:
            return {'drift_detected': False, 'message': 'No baseline set'}
        
        total = sum(self.current_distribution.values())
        if total == 0:
            return {'drift_detected': False, 'message': 'No predictions yet'}
        
        current_dist = {
            k: v / total for k, v in self.current_distribution.items()
        }
        
        # Calculate drift using KL divergence or simple difference
        drift_scores = {}
        max_drift = 0
        
        for class_name in self.baseline_distribution:
            baseline_prob = self.baseline_distribution.get(class_name, 0)
            current_prob = current_dist.get(class_name, 0)
            drift = abs(baseline_prob - current_prob)
            drift_scores[class_name] = drift
            max_drift = max(max_drift, drift)
        
        drift_detected = max_drift > self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'max_drift': max_drift,
            'drift_threshold': self.drift_threshold,
            'drift_scores': drift_scores,
            'baseline_distribution': self.baseline_distribution,
            'current_distribution': current_dist
        }
