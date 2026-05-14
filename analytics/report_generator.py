from datetime import datetime
from analytics.workout_analytics import WorkoutSession, WorkoutStatistics
from analytics.performance_analytics import PerformanceAnalytics
from analytics.fatigue_analytics import FatigueAnalytics
from  analytics.progress_prediction import ProgressPredictor

class ReportGenerator:
    def __init__(self, sessions: list[WorkoutSession]):
        self.sessions = sessions
        self.stats = WorkoutStatistics(sessions)
        self.performance = PerformanceAnalytics(sessions)
        self.fatigue = FatigueAnalytics(sessions)
        self.predictor = ProgressPredictor(sessions)

    def generate(self, exercises: list[str] = None) -> dict:
        exercises = exercises or list(set(s.exercise for s in self.sessions))
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.stats.summary(),
            "personal_records": self.performance.personal_records(),
            "consistency_score": self.performance.consistency_score(),
            "rest_analysis": self.fatigue.rest_days_analysis(),
            "rpe_trend": self.fatigue.rpe_trend(),
            "exercises": {},
        }
        for ex in exercises:
            report["exercises"][ex] = {
                "strength_trend": self.performance.strength_trend(ex),
                "volume_trend": self.performance.volume_trend(ex),
                "acwr": self.fatigue.acute_chronic_workload_ratio(ex),
                "plateau": self.predictor.plateau_detection(ex),
                "prediction_10_sessions": self.predictor.predict_max_weight(ex, 10),
            }
        return report

    def to_text(self, exercises: list[str] = None) -> str:
        data = self.generate(exercises)
        lines = [
            "=" * 60,
            "        WORKOUT ANALYTICS REPORT",
            f"  Generated: {data['generated_at']}",
            "=" * 60,
            "",
            "📊 SUMMARY",
            f"  Total Sessions     : {data['summary']['total_sessions']}",
            f"  Total Duration     : {data['summary']['total_duration_min']} min",
            f"  Total Calories     : {data['summary']['total_calories']} kcal",
            f"  Consistency Score  : {data['consistency_score']} / 100",
            f"  Avg RPE            : {data['rpe_trend'].get('avg_rpe', 'N/A')}",
            "",
            "😴 REST ANALYSIS",
        ]
        rest = data["rest_analysis"]
        lines += [
            f"  Avg Rest Days      : {rest.get('avg_rest_days')}",
            f"  Overtraining Flags : {rest.get('overtraining_flags')}",
            f"  Undertraining Flags: {rest.get('undertraining_flags')}",
            "",
        ]
        for ex, info in data["exercises"].items():
            lines += [
                f"🏋️  {ex.upper()}",
                f"  Strength Trend     : {info['strength_trend'].get('trend', 'N/A')} "
                f"(slope={info['strength_trend'].get('slope_per_session', 'N/A')})",
                f"  Volume Trend       : {info['volume_trend'].get('trend', 'N/A')}",
                f"  ACWR Status        : {info['acwr'].get('status', 'N/A')} "
                f"(ratio={info['acwr'].get('acwr', 'N/A')})",
                f"  Plateau            : {'YES ⚠️' if info['plateau'].get('plateau_detected') else 'No'}",
                f"  Predicted Max (10s): {info['prediction_10_sessions'].get('predicted_max', 'N/A')} kg",
                "",
            ]
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_html(self, exercises: list[str] = None, save_path: str = "report.html") -> str:
        data = self.generate(exercises)
        s = data["summary"]
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Workout Report</title>
<style>
  body {{ font-family: monospace; background: #0f0f0f; color: #eee; padding: 2rem; }}
  h1 {{ color: #00e5ff; }} h2 {{ color: #ff6b35; border-bottom: 1px solid #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th {{ background: #1a1a1a; color: #00e5ff; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #222; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }}
  .good {{ background: #1a3a1a; color: #69f0ae; }}
  .warn {{ background: #3a2a0a; color: #ffd740; }}
  .bad  {{ background: #3a0a0a; color: #ff5252; }}
</style>
</head>
<body>
<h1>🏋️ Workout Analytics Report</h1>
<p>Generated: {data['generated_at']}</p>
<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Sessions</td><td>{s['total_sessions']}</td></tr>
  <tr><td>Total Duration</td><td>{s['total_duration_min']} min</td></tr>
  <tr><td>Total Calories</td><td>{s['total_calories']} kcal</td></tr>
  <tr><td>Consistency Score</td><td>{data['consistency_score']} / 100</td></tr>
</table>
<h2>Exercise Breakdown</h2>
"""
        for ex, info in data["exercises"].items():
            trend = info['strength_trend'].get('trend', 'N/A')
            badge = "good" if trend == "improving" else "bad" if trend == "declining" else "warn"
            html += f"""
<h3>{ex}</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Strength Trend</td>
      <td><span class='badge {badge}'>{trend}</span></td></tr>
  <tr><td>ACWR Status</td><td>{info['acwr'].get('status','N/A')}</td></tr>
  <tr><td>Plateau Detected</td>
      <td>{'<span class="badge warn">YES</span>' if info['plateau'].get('plateau_detected') else 'No'}</td></tr>
  <tr><td>Predicted Max (10 sessions)</td>
      <td>{info['prediction_10_sessions'].get('predicted_max','N/A')} kg</td></tr>
</table>"""
        html += "</body></html>"
        with open(save_path, "w") as f:
            f.write(html)
        return html