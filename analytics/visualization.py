import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from analytics.workout_analytics import WorkoutSession, WorkoutStatistics

class WorkoutVisualizer:
    def __init__(self, sessions: list[WorkoutSession]):
        self.sessions = sorted(sessions, key=lambda s: s.date)
        self.stats = WorkoutStatistics(sessions)
        self._style()

    def _style(self):
        plt.rcParams.update({
            "figure.facecolor": "#0f0f0f",
            "axes.facecolor": "#1a1a1a",
            "axes.edgecolor": "#333",
            "axes.labelcolor": "#ccc",
            "text.color": "#eee",
            "xtick.color": "#999",
            "ytick.color": "#999",
            "grid.color": "#2a2a2a",
            "grid.linestyle": "--",
            "font.family": "monospace",
        })

    def plot_strength_progression(self, exercise: str, save_path: str = None):
        filtered = [s for s in self.sessions if s.exercise == exercise]
        dates = [s.date for s in filtered]
        weights = [self.stats.max_weight(s) for s in filtered]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, weights, color="#00e5ff", linewidth=2.5, marker="o",
                markersize=6, markerfacecolor="#ff6b35")
        ax.fill_between(dates, weights, alpha=0.15, color="#00e5ff")
        ax.set_title(f"Strength Progression — {exercise}", fontsize=14, pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Max Weight (kg)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_volume_heatmap(self, save_path: str = None):
        from collections import defaultdict
        exercise_weekly: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for s in self.sessions:
            week = s.date.strftime("%Y-W%U")
            exercise_weekly[s.exercise][week] += self.stats.total_volume(s)

        exercises = list(exercise_weekly.keys())
        all_weeks = sorted(set(w for ex in exercise_weekly.values() for w in ex))
        matrix = np.array([
            [exercise_weekly[ex].get(w, 0) for w in all_weeks]
            for ex in exercises
        ])

        fig, ax = plt.subplots(figsize=(max(10, len(all_weeks) * 1.2), max(4, len(exercises) * 0.8)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(all_weeks)))
        ax.set_xticklabels(all_weeks, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(exercises)))
        ax.set_yticklabels(exercises)
        ax.set_title("Weekly Volume Heatmap", fontsize=14, pad=15)
        plt.colorbar(im, ax=ax, label="Volume (kg·reps)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_session_summary(self, save_path: str = None):
        summary = self.stats.summary()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Sessions per week bar
        from collections import Counter
        weeks = Counter(s.date.strftime("%Y-W%U") for s in self.sessions)
        axes[0].bar(range(len(weeks)), list(weeks.values()), color="#00e5ff", edgecolor="#333")
        axes[0].set_xticks(range(len(weeks)))
        axes[0].set_xticklabels(list(weeks.keys()), rotation=45, ha="right", fontsize=7)
        axes[0].set_title("Sessions / Week")
        axes[0].set_ylabel("Count")

        # Duration distribution
        durations = [s.total_duration_min for s in self.sessions]
        axes[1].hist(durations, bins=10, color="#ff6b35", edgecolor="#333")
        axes[1].set_title("Session Duration Distribution")
        axes[1].set_xlabel("Minutes")

        # Calories per session
        calories = [s.calories_burned or 0 for s in self.sessions]
        axes[2].plot(range(len(calories)), calories, color="#b2ff59", linewidth=2)
        axes[2].set_title("Calories per Session")
        axes[2].set_xlabel("Session #")
        axes[2].set_ylabel("kcal")

        plt.suptitle("Workout Overview", fontsize=15, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()