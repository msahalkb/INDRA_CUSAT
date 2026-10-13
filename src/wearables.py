"""Synthetic wearable generator (heart rate + step count) for GlucoTwin.

The Shanghai T2DM dataset has no wearable data, so heart rate and steps are
SIMULATED. Design rule to avoid leaking the answer: the simulator never reads
glucose. It only uses time of day, patient traits (age, BMI) and the logged meal
times (people tend to walk a bit after eating, and digestion raises HR slightly).
Any signal the model finds in these streams is therefore behavioural context,
not a copy of the label. This is stated in the README Limitations section.
"""
import numpy as np

STEP_MIN = 15  # one sample per 15 minutes, aligned with the CGM grid


def simulate(timestamps, meal_flags, age, bmi, seed):
    """timestamps: DatetimeIndex (15-min grid). meal_flags: 0/1 array, 1 where a meal was logged.
    Returns (hr_bpm, steps_per_15min) numpy arrays."""
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    hour = timestamps.hour.values + timestamps.minute.values / 60.0

    # patient traits
    activity = np.clip(rng.normal(1.0 - 0.006 * (age - 55), 0.25), 0.4, 1.6)
    rest_hr = np.clip(rng.normal(68 + 0.4 * (bmi - 24), 5), 52, 90)

    # hourly activity profile: asleep 23:00-06:30, peaks mid-morning and early evening
    awake = ((hour >= 6.5) & (hour < 23)).astype(float)
    profile = awake * (0.35 + 0.65 * np.exp(-((hour - 10) ** 2) / 8) + 0.6 * np.exp(-((hour - 17.5) ** 2) / 6))

    # walking episodes (Markov chain), more likely 15-60 min after a meal
    since_meal = np.full(n, 99.0)
    last = -999
    for i in range(n):
        if meal_flags[i]:
            last = i
        since_meal[i] = (i - last) * STEP_MIN if last > -999 else 99.0
    post_meal = ((since_meal >= 15) & (since_meal <= 60)).astype(float)

    walking = np.zeros(n)
    state = 0
    for i in range(n):
        p_start = 0.04 * profile[i] + 0.25 * post_meal[i] * awake[i]
        p_stop = 0.45
        state = (rng.random() < p_start) if state == 0 else (rng.random() > p_stop)
        walking[i] = float(state)

    base_steps = rng.poisson(25 * profile * activity)           # light movement
    walk_steps = rng.poisson(900 * walking * activity)           # ~60 steps/min for 15 min
    steps = (base_steps + walk_steps).astype(float)

    # heart rate: resting + circadian dip + exercise + digestion + AR(1) noise
    circ = -7 * np.cos(2 * np.pi * (hour - 15) / 24) * 0.5 - 4 * (1 - awake)
    exercise = 0.045 * np.minimum(steps, 1400)
    digest = 3.0 * (since_meal <= 60)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 1.8)
    hr = np.clip(rest_hr + circ + exercise + digest + noise, 45, 160)
    return hr.round(1), steps
