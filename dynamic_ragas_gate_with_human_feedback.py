## Dynamic RAGAS Gate with Human Feedback

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

initial_ragas_threshold = 0.7
ragas_threshold = initial_ragas_threshold

# Smoothing factor for exponential smoothing
# The alpha parameter controls how quickly the influence decays
# A larger alpha (closer to 1) would make the new_threshold respond quickly to recent scores
alpha = 0.1

# Initial uncertainty range and sampling parameters
certainty_low = 0.5
certainty_high = 0.7
recalibration_interval = 10  # Periodic recalibration interval
diverse_sample_rate = 0.2  # Probability of sampling confident answers


# Adaptive window size parameters
min_window_size = 3
max_window_size = 10
# Define a variance level where consistency is considered high, low aka more consistent
feedback_consistency_threshold = 0.2

# huamn feedback uncertainty bound
human_feedback_uncertainty_min_bound = 0.3  # Ensure min bound
human_feedback_uncertainty_max_bound = 0.9  # Ensure max bound
human_feedback_uncertainty_bound_buffer = 0.1


# Dynamically calculate window size based on feedback variance
def calculate_adaptive_window_size(
    variance, min_window_size, max_window_size, threshold
):
    # Interpolate window size between min and max based on variance relative to threshold
    # If stable enough (aka low var), then use min window
    if variance < threshold:
        return min_window_size
    # If not stable enough (aka high var), modify the window based on var
    elif variance >= threshold:
        # Map variance to a range between min and max window sizes
        # if not var is not low enough, the win upper bound will be max_window_size
        normalized_variance = min(1, variance / threshold)
        return int(
            min_window_size + normalized_variance * (max_window_size - min_window_size)
        )


# calculate feedback consistency as the variance within recent feedback
def calculate_feedback_consistency(feedback_history, max_window_size):
    # Calculate variance of the last `max_window_size` feedbacks to evaluate consistency
    recent_feedback = feedback_history[-max_window_size:]
    return np.var(recent_feedback) if len(recent_feedback) > 1 else 0


# adjust the uncertainty range dynamically
def dynamic_uncertainty_adjustment(
    scores,
    human_feedback_uncertainty_min_bound,
    human_feedback_uncertainty_max_bound,
    human_feedback_uncertainty_bound_buffer,
):
    # Recalculate uncertainty range based on recent scores (e.g., interquartile range)
    q1, q3 = np.percentile(scores, [25, 75])

    new_low = max(
        human_feedback_uncertainty_min_bound,
        q1 - human_feedback_uncertainty_bound_buffer,
    )
    new_high = min(
        human_feedback_uncertainty_max_bound,
        q3 + human_feedback_uncertainty_bound_buffer,
    )

    return new_low, new_high


# decide if human feedback is needed, with diverse sampling for confident answers
def needs_human_feedback_dynamic(
    ragas_score,
    scores,
    step,
    human_feedback_uncertainty_min_bound,
    human_feedback_uncertainty_max_bound,
    human_feedback_uncertainty_bound_buffer,
):

    # Recalibrate uncertainty range periodically
    if step % recalibration_interval == 0:
        global certainty_low, certainty_high
        certainty_low, certainty_high = dynamic_uncertainty_adjustment(
            scores,
            human_feedback_uncertainty_min_bound,
            human_feedback_uncertainty_max_bound,
            human_feedback_uncertainty_bound_buffer,
        )

    # Determine if feedback is needed within dynamic uncertainty range or by diverse sampling
    in_uncertainty_range = certainty_low <= ragas_score <= certainty_high

    # This sampling ensures not all confident answers (certainty high enough) are selected by diverse_sample_rate
    confident_sampling = (
        ragas_score > certainty_high and random.random() < diverse_sample_rate
    )

    return in_uncertainty_range or confident_sampling


# placeholder for simulate human feedback (1 for pass, 0 for fail)
def human_feedback_for_ragas():
    # Simulate human judgment for threshold accuracy
    # In real cases, replace with actual human feedback
    return np.random.choice([0, 1])


# Adaptive threshold adjustment with dynamic window size
def adaptive_exponential_smoothing(
    ragas_score, human_feedback, current_threshold, feedback_history, window_size
):
    # Only proceed with adjustment if we have positive human feedback
    if human_feedback != 1:
        return current_threshold

    # Calculate the weighted average based on window size
    weights = np.exp(np.linspace(-alpha * window_size, 0, len(feedback_history)))
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    # Apply weighted smoothing
    weighted_adjustment = np.sum(weights * (ragas_score - current_threshold))
    new_threshold = current_threshold + weighted_adjustment

    return new_threshold


# Track scores to recalibrate uncertainty range periodically
all_ragas_scores = []

# track feedback to calculate feedback consistency & update threshold
feedback_history = []

# for report
feedback_needed_count = 0
threshold_history = [initial_ragas_threshold]

df_res = pd.read_excel("my_custom_testset.xlsx")
df_res = df_res[
    [
        "question",
        "contexts",
        "answer",
        "ground_truth",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "kw",
        "kw_metric",
        "weighted_average_score",
    ]
]

# Calculate ragas_score as mean of the 4 metrics
df_res["ragas_score"] = df_res[
    ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
].mean(axis=1)

for step, ragas_score in enumerate(df_res["ragas_score"]):
    # can also put in contextual keyword score to get dynamic threshold
    all_ragas_scores.append(ragas_score)

    # if uncertainty is high enough, we need human feedback, aka active learning gate
    if needs_human_feedback_dynamic(
        ragas_score,
        all_ragas_scores,
        step,
        human_feedback_uncertainty_min_bound,
        human_feedback_uncertainty_max_bound,
        human_feedback_uncertainty_bound_buffer,
    ):
        feedback_needed_count += 1
        human_feedback = human_feedback_for_ragas()
        feedback_history.append(human_feedback)

        current_variance = calculate_feedback_consistency(
            feedback_history, max_window_size
        )
        adaptive_window_size = calculate_adaptive_window_size(
            current_variance,
            min_window_size,
            max_window_size,
            feedback_consistency_threshold,
        )

        # Update threshold
        ragas_threshold = adaptive_exponential_smoothing(
            ragas_score,
            human_feedback,
            ragas_threshold,
            feedback_history[-adaptive_window_size:],
            adaptive_window_size,
        )

        # Track the new threshold
        threshold_history.append(ragas_threshold)

print(f"Total Feedback Needed: {feedback_needed_count}")
print(f"Dynamic Uncertainty Range: {certainty_low:.2f} - {certainty_high:.2f}")
print(f"Final Adjusted RAGAS Threshold: {ragas_threshold:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(range(len(threshold_history)), threshold_history, marker="o")
plt.title("RAGAS Threshold Adjustment Over Time")
plt.xlabel("Feedback Iterations")
plt.ylabel("RAGAS Threshold")
plt.grid(True)
plt.show()
