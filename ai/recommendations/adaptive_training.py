import json, os
from datetime import datetime

class AdaptiveTraining:
    def __init__(self, user_id='default', data_dir='user_data'):
        self.user_id  = user_id
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.history  = self.load_history()

    def load_history(self):
        path = f"{self.data_dir}/{self.user_id}_history.json"
        if os.path.exists(path):
            return json.load(open(path))
        return {'sessions': [], 'level': 'beginner', 'streak': 0}

    def save_history(self):
        path = f"{self.data_dir}/{self.user_id}_history.json"
        json.dump(self.history, open(path, 'w'))

    def log_session(self, session_data):
        """Workout session log karo"""
        session = {
            'date':          datetime.now().strftime('%Y-%m-%d'),
            'exercises':     session_data.get('exercises', []),
            'duration_mins': session_data.get('duration_mins', 0),
            'calories':      session_data.get('calories', 0),
            'avg_form':      session_data.get('avg_form', 0.5),
            'fatigue_level': session_data.get('fatigue_level', 'fresh')
        }
        self.history['sessions'].append(session)
        self.history['streak'] += 1
        self._update_level()
        self.save_history()
        print(f"Session logged! Streak: {self.history['streak']} days")

    def _update_level(self):
        """Progress dekh ke level update karo"""
        sessions = self.history['sessions']
        if len(sessions) < 5:
            return

        recent = sessions[-5:]
        avg_form     = sum(s['avg_form'] for s in recent) / 5
        avg_duration = sum(s['duration_mins'] for s in recent) / 5

        current = self.history['level']

        if avg_form > 0.8 and avg_duration > 40 and current == 'beginner':
            self.history['level'] = 'intermediate'
            print("Level up! Intermediate ban gaye!")
        elif avg_form > 0.85 and avg_duration > 55 and current == 'intermediate':
            self.history['level'] = 'advanced'
            print("Level up! Advanced ban gaye!")

    def get_recommendation(self):
        """History dekh ke next workout recommend karo"""
        sessions  = self.history['sessions']
        level     = self.history['level']
        streak    = self.history['streak']

        if not sessions:
            return {'message': 'Pehla workout karo!', 'intensity': 'low'}

        last = sessions[-1]
        fatigue  = last.get('fatigue_level', 'fresh')
        avg_form = last.get('avg_form', 0.5)

        if fatigue in ['high_fatigue', 'exhausted']:
            return {
                'message':   'Kal rest karo — body recover karegi',
                'intensity': 'rest',
                'suggestion': 'Light stretching ya yoga'
            }
        elif avg_form < 0.6:
            return {
                'message':   'Form pe focus karo — weight/intensity kam karo',
                'intensity': 'low',
                'suggestion': 'Same exercises, better form'
            }
        elif streak % 7 == 0:
            return {
                'message':   f'{streak} din ho gaye — rest day lo!',
                'intensity': 'rest',
                'suggestion': 'Complete rest ya light walk'
            }
        else:
            return {
                'message':   f'Level: {level} | Streak: {streak} days 🔥',
                'intensity': 'normal',
                'suggestion': 'Normal workout continue karo'
            }


if __name__ == "__main__":
    trainer = AdaptiveTraining(user_id='user_123')

    # Session log karo
    trainer.log_session({
        'exercises':     ['squats', 'pushups', 'plank'],
        'duration_mins': 35,
        'calories':      280,
        'avg_form':      0.75,
        'fatigue_level': 'mild_fatigue'
    })

    # Recommendation lo
    rec = trainer.get_recommendation()
    print("\nRecommendation:", rec)