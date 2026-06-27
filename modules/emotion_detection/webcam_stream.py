import cv2
import time
from modules.emotion_detection.emotion_model import detect_emotion_from_frame
from modules.emotion_detection.engagement_scorer import EngagementTracker

# Colours for the overlay text
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)


def get_color(label: str) -> tuple:
    return {
        "focused":     COLOR_GREEN,
        "distracted":  COLOR_YELLOW,
        "disengaged":  COLOR_RED,
    }.get(label, COLOR_WHITE)


def run_webcam_session(duration_seconds: int = 60) -> dict:
    """
    Opens webcam, tracks student engagement for the given duration.

    Returns a session summary with average engagement,
    dominant emotion, and intervention flag.

    Press Q to stop early.
    """
    cap     = cv2.VideoCapture(0)
    tracker = EngagementTracker(window_size=30)

    if not cap.isOpened():
        print("Could not open webcam.")
        return {}

    print(f"Tracking engagement for {duration_seconds}s — press Q to stop")

    start_time     = time.time()
    frame_count    = 0
    emotion_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed > duration_seconds:
            break

        # Only run detection every 2 seconds to save CPU
        if frame_count % 60 == 0:
            result = detect_emotion_from_frame(frame)
            tracker.update(result)

            if result["face_found"] and result["emotion"]:
                emotion = result["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                # Draw face box
                x, y, w, h = result["face_box"]
                engagement  = tracker.get_engagement()
                color       = get_color(engagement["label"])
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Draw labels
                cv2.putText(frame, f"Emotion: {emotion} ({result['confidence']:.0%})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Engagement: {engagement['label']} ({engagement['score']:.2f})",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Time: {int(elapsed)}s / {duration_seconds}s",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

        cv2.imshow("EduMentor — Engagement Tracker", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    engagement    = tracker.get_engagement()
    dominant      = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"

    summary = {
        "avg_engagement_score": engagement["score"],
        "engagement_label":     engagement["label"],
        "dominant_emotion":     dominant,
        "emotion_breakdown":    emotion_counts,
        "needs_intervention":   tracker.needs_intervention(),
        "session_duration_sec": int(time.time() - start_time),
    }

    print("\nSession Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    run_webcam_session(duration_seconds=30)