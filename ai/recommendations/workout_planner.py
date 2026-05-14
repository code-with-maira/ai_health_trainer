class WorkoutPlanner:
    def __init__(self):
        self.exercises = {
            'weight_loss': [
                {'name': 'Jumping Jacks', 'duration': 60, 'calories': 10},
                {'name': 'Burpees',       'duration': 45, 'calories': 15},
                {'name': 'High Knees',    'duration': 60, 'calories': 12},
                {'name': 'Mountain Climbers', 'duration': 45, 'calories': 13},
                {'name': 'Jump Rope',     'duration': 60, 'calories': 14},
            ],
            'muscle_gain': [
                {'name': 'Push Ups',    'duration': 45, 'calories': 8},
                {'name': 'Squats',      'duration': 45, 'calories': 10},
                {'name': 'Lunges',      'duration': 45, 'calories': 9},
                {'name': 'Plank',       'duration': 60, 'calories': 5},
                {'name': 'Pull Ups',    'duration': 30, 'calories': 8},
            ],
            'endurance': [
                {'name': 'Jogging',     'duration': 300, 'calories': 40},
                {'name': 'Cycling',     'duration': 300, 'calories': 35},
                {'name': 'Swimming',    'duration': 300, 'calories': 45},
                {'name': 'Brisk Walk',  'duration': 300, 'calories': 25},
            ],
            'flexibility': [
                {'name': 'Yoga Sun Salutation', 'duration': 120, 'calories': 8},
                {'name': 'Hamstring Stretch',   'duration': 60,  'calories': 3},
                {'name': 'Hip Flexor Stretch',  'duration': 60,  'calories': 3},
                {'name': 'Shoulder Stretch',    'duration': 60,  'calories': 2},
            ]
        }

    def create_plan(self, user_profile):
        """
        user_profile: {
            goal: 'weight_loss' / 'muscle_gain' / 'endurance' / 'flexibility',
            level: 'beginner' / 'intermediate' / 'advanced',
            days_per_week: 3-6,
            available_mins: 30-60
        }
        """
        goal      = user_profile.get('goal', 'weight_loss')
        level     = user_profile.get('level', 'beginner')
        days      = user_profile.get('days_per_week', 3)
        available = user_profile.get('available_mins', 30)

        exercises = self.exercises.get(goal, self.exercises['weight_loss'])

        # Level ke hisaab se sets
        sets_map = {'beginner': 2, 'intermediate': 3, 'advanced': 4}
        sets = sets_map.get(level, 2)

        plan = []
        for day in range(1, days + 1):
            day_exercises = []
            total_time = 0
            for ex in exercises:
                if total_time + ex['duration'] > available * 60:
                    break
                day_exercises.append({
                    'exercise': ex['name'],
                    'duration': f"{ex['duration']}s",
                    'sets':     sets,
                    'calories': ex['calories'] * sets
                })
                total_time += ex['duration'] * sets

            plan.append({
                'day':        day,
                'exercises':  day_exercises,
                'total_cals': sum(e['calories'] for e in day_exercises)
            })

        return {
            'goal':         goal,
            'level':        level,
            'days_per_week': days,
            'weekly_plan':  plan
        }

    def print_plan(self, plan):
        print(f"\nWorkout Plan — {plan['goal'].upper()}")
        print(f"Level: {plan['level']} | Days: {plan['days_per_week']}/week")
        print("=" * 40)
        for day in plan['weekly_plan']:
            print(f"\nDay {day['day']} — Est. {day['total_cals']} calories")
            for ex in day['exercises']:
                print(f"  • {ex['exercise']}: {ex['sets']} sets x {ex['duration']}")


if __name__ == "__main__":
    planner = WorkoutPlanner()
    profile = {
        'goal': 'weight_loss',
        'level': 'intermediate',
        'days_per_week': 4,
        'available_mins': 45
    }
    plan = planner.create_plan(profile)
    planner.print_plan(plan)