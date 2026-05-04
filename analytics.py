"""
Industrial Analytics Engine for PCB Defects
This module maps YOLO detections to industrial severity, costs, and repair processes.
"""
import cv2

# SPECIFIC INDUSTRIAL CONSTANTS (Mapped to norbertelter/pcb-defect-dataset classes)
DEFECT_METRICS = {
    'mouse_bite': {'severity': 'Moderate', 'machine': 'Manual Rework Station', 'risk': 'Medium'},
    'spur': {'severity': 'Low', 'machine': 'Optical Inspection / Manual Trimming', 'risk': 'Low'},
    'missing_hole': {'severity': 'Critical', 'machine': 'CNC Drilling Machine', 'risk': 'High'},
    'short': {'severity': 'Critical', 'machine': 'Rework Station (BGA/Laser)', 'risk': 'High'},
    'open_circuit': {'severity': 'Critical', 'machine': 'Soldering/Rework Station', 'risk': 'High'},
    'spurious_copper': {'severity': 'Low', 'machine': 'Optical Inspection / Cleanroom', 'risk': 'Low'}
}

def analyze_defects(yolo_results, diff_mask=None):
    """
    Processes YOLOv8 results to return industrial insights.
    """
    analysis = {
        'total_defects': 0,
        'ai_detections': 0,
        'defects_breakdown': [],
        'verdict': 'PASS',
        'risk_level': 'Low',
        'required_processes': set()
    }
    
    # 1. PROCESS AI DETECTIONS (YOLO)
    names = yolo_results[0].names
    for box in yolo_results[0].boxes:
        analysis['ai_detections'] += 1
        analysis['total_defects'] += 1
        class_id = int(box.cls)
        conf = float(box.conf)
        label = names[class_id]
        
        metrics = DEFECT_METRICS.get(label, {
            'severity': 'Unknown', 
            'machine': 'General Inspection', 
            'risk': 'Medium'
        })
        
        analysis['required_processes'].add(metrics['machine'])
        
        # Decision Logic: Any Critical defect = REJECT
        if metrics['severity'] == 'Critical':
            analysis['verdict'] = 'REJECT'
            analysis['risk_level'] = 'High'
        elif metrics['severity'] == 'Moderate' and analysis['verdict'] != 'REJECT':
            analysis['verdict'] = 'REPAIR'
            if analysis['risk_level'] != 'High':
                analysis['risk_level'] = 'Medium'
        
        analysis['defects_breakdown'].append({
            'source': 'AI (YOLOv8)',
            'type': label.replace('_', ' ').title(),
            'confidence': f"{conf:.2%}",
            'severity': metrics['severity'],
            'process': metrics['machine']
        })
        
    analysis['required_processes'] = list(analysis['required_processes'])
    return analysis
