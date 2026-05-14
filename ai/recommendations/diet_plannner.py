import random


class DietPlanner:

    def __init__(self):

        # =====================================================
        # Meal Database
        # =====================================================
        self.meals = {

            'weight_loss': {

                'breakfast': [
                    'Oatmeal + fruits',
                    'Greek yogurt + berries',
                    'Eggs + vegetables'
                ],

                'lunch': [
                    'Grilled chicken salad',
                    'Lentil soup + bread',
                    'Tuna sandwich'
                ],

                'dinner': [
                    'Grilled fish + vegetables',
                    'Chicken curry + rice (small)',
                    'Dal + roti'
                ],

                'snacks': [
                    'Apple + peanut butter',
                    'Nuts (handful)',
                    'Green tea'
                ]
            },

            'muscle_gain': {

                'breakfast': [
                    'Eggs (4) + oats',
                    'Protein shake + banana',
                    'Chicken + rice'
                ],

                'lunch': [
                    'Chicken breast + rice + vegetables',
                    'Beef + potatoes',
                    'Tuna + pasta'
                ],

                'dinner': [
                    'Salmon + quinoa',
                    'Chicken + sweet potato',
                    'Beef + vegetables'
                ],

                'snacks': [
                    'Protein shake',
                    'Cottage cheese',
                    'Boiled eggs'
                ]
            },

            'endurance': {

                'breakfast': [
                    'Banana + oats + honey',
                    'Whole grain toast + eggs',
                    'Smoothie'
                ],

                'lunch': [
                    'Pasta + chicken',
                    'Rice + fish',
                    'Quinoa bowl'
                ],

                'dinner': [
                    'Carb-rich dinner + protein',
                    'Rice + dal + sabzi',
                    'Pasta + vegetables'
                ],

                'snacks': [
                    'Dates',
                    'Energy bar',
                    'Banana'
                ]
            }
        }

    # =========================================================
    # CALCULATE DAILY CALORIES
    # =========================================================
    def calculate_calories(self,
                           weight_kg,
                           height_cm,
                           age,
                           gender,
                           activity_level):

        """
        Calculate TDEE using Mifflin-St Jeor Formula
        """

        if gender.lower() == 'male':

            bmr = (
                10 * weight_kg +
                6.25 * height_cm -
                5 * age + 5
            )

        else:

            bmr = (
                10 * weight_kg +
                6.25 * height_cm -
                5 * age - 161
            )

        activity_map = {

            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }

        multiplier = activity_map.get(
            activity_level,
            1.55
        )

        tdee = bmr * multiplier

        return round(tdee)

    # =========================================================
    # CREATE DIET PLAN
    # =========================================================
    def create_diet_plan(self,
                         user_profile,
                         burned_calories=0):

        try:

            # ---------------------------------------------
            # User profile
            # ---------------------------------------------
            goal = user_profile.get(
                'goal',
                'weight_loss'
            )

            weight = float(
                user_profile.get(
                    'weight',
                    70
                )
            )

            height = float(
                user_profile.get(
                    'height',
                    170
                )
            )

            age = int(
                user_profile.get(
                    'age',
                    25
                )
            )

            gender = user_profile.get(
                'gender',
                'male'
            )

            activity_level = user_profile.get(
                'activity_level',
                'moderate'
            )

            # ---------------------------------------------
            # Calculate TDEE
            # ---------------------------------------------
            tdee = self.calculate_calories(
                weight,
                height,
                age,
                gender,
                activity_level
            )

            # ---------------------------------------------
            # Goal adjustment
            # ---------------------------------------------
            if goal == 'weight_loss':

                target_calories = tdee - 500

            elif goal == 'muscle_gain':

                target_calories = tdee + 300

            else:

                target_calories = tdee

            # Safety limit
            target_calories = max(
                target_calories,
                1200
            )

            # ---------------------------------------------
            # Meal selection
            # ---------------------------------------------
            meals = self.meals.get(
                goal,
                self.meals['weight_loss']
            )

            meal_plan = {

                'breakfast':
                    random.choice(
                        meals['breakfast']
                    ),

                'lunch':
                    random.choice(
                        meals['lunch']
                    ),

                'dinner':
                    random.choice(
                        meals['dinner']
                    ),

                'snacks':
                    random.choice(
                        meals['snacks']
                    )
            }

            # ---------------------------------------------
            # Water recommendation
            # ---------------------------------------------
            water = round(
                weight * 0.033,
                1
            )

            # ---------------------------------------------
            # Macronutrient estimation
            # ---------------------------------------------
            protein = round(
                weight * 1.8
            )

            fats = round(
                target_calories * 0.25 / 9
            )

            carbs = round(
                (
                    target_calories -
                    (protein * 4 + fats * 9)
                ) / 4
            )

            return {

                "goal":
                    goal,

                "daily_calories":
                    target_calories,

                "burned_calories":
                    burned_calories,

                "net_calories":
                    target_calories -
                    burned_calories,

                "macros": {

                    "protein_g":
                        protein,

                    "carbs_g":
                        carbs,

                    "fats_g":
                        fats
                },

                "meals":
                    meal_plan,

                "water_intake":
                    f"{water} liters/day"
            }

        except Exception as e:

            return {
                "error": str(e)
            }


# =============================================================
# TESTING
# =============================================================
if __name__ == "__main__":

    planner = DietPlanner()

    profile = {

        'weight': 75,
        'height': 175,
        'age': 25,
        'gender': 'male',
        'goal': 'weight_loss',
        'activity_level': 'moderate'
    }

    plan = planner.create_diet_plan(
        profile,
        burned_calories=400
    )

    print("\n DIET PLAN\n")

    for key, value in plan.items():

        print(f"{key}: {value}")