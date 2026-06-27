import time
from modules.adaptive_engine.difficulty_adjuster import DifficultyAdjuster
from modules.adaptive_engine.question_bank import QuestionBank
from modules.content_generator.quiz_generator import evaluate_answer


class AdaptiveSession:
    """
    Runs a complete adaptive quiz session.
    Combines QuestionBank + DifficultyAdjuster into one clean interface.

    Usage:
        session = AdaptiveSession(topic_text, "Photosynthesis")
        session.start()
        while not session.is_done():
            question = session.next_question()
            answer   = input("Your answer: ")
            result   = session.submit_answer(answer)
            print(result["feedback"])
        print(session.get_summary())
    """

    def __init__(self, topic_text: str, topic_name: str, max_questions: int = 10):
        self.topic_name      = topic_name
        self.max_questions   = max_questions
        self.adjuster        = DifficultyAdjuster(starting_level="medium")
        self.bank            = QuestionBank(topic_text, topic_name)
        self.current_question = None
        self.question_start   = None
        self.questions_done   = 0

    def start(self):
        print(f"\nStarting adaptive quiz on '{self.topic_name}'")
        print(f"Questions: {self.max_questions} | Starting difficulty: medium\n")

    def next_question(self) -> dict | None:
        if self.is_done():
            return None

        difficulty            = self.adjuster.recommended_next_difficulty()
        self.current_question = self.bank.get_question(difficulty)
        self.question_start   = time.time()

        if self.current_question:
            self.current_question["difficulty_level"] = difficulty

        return self.current_question

    def submit_answer(self, student_answer: str) -> dict:
        if not self.current_question:
            return {"error": "No active question"}

        time_taken = time.time() - self.question_start
        evaluation = evaluate_answer(self.current_question, student_answer)
        adjustment = self.adjuster.record_answer(
            evaluation["is_correct"],
            time_taken_sec=time_taken
        )

        self.questions_done += 1

        return {
            "is_correct":      evaluation["is_correct"],
            "correct_answer":  evaluation["correct_answer"],
            "explanation":     evaluation["explanation"],
            "feedback":        adjustment["feedback"],
            "new_difficulty":  adjustment["current_level"],
            "accuracy_so_far": adjustment["overall_accuracy"],
            "question_num":    self.questions_done,
        }

    def is_done(self) -> bool:
        return self.questions_done >= self.max_questions

    def get_summary(self) -> dict:
        return self.adjuster.get_session_summary()


if __name__ == "__main__":
    topic_text = """
    Photosynthesis is the process by which plants use sunlight, water, and CO2
    to produce glucose and oxygen. It occurs in chloroplasts using chlorophyll.
    The equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
    Two stages: light-dependent reactions (thylakoid) and Calvin cycle (stroma).
    """

    session = AdaptiveSession(topic_text, "Photosynthesis", max_questions=5)
    session.start()

    while not session.is_done():
        q = session.next_question()
        if not q:
            break

        print(f"\nQ{session.questions_done + 1} [{q['difficulty_level']}]: {q['question']}")
        for key, val in q["options"].items():
            print(f"  {key}) {val}")

        answer = input("Your answer (A/B/C/D): ").strip().upper()
        result = session.submit_answer(answer)

        print(f"{'✅ Correct!' if result['is_correct'] else '❌ Wrong!'}")
        print(f"Explanation: {result['explanation']}")
        print(f"Feedback: {result['feedback']}")
        print(f"New difficulty: {result['new_difficulty']} | Accuracy: {result['accuracy_so_far']:.0%}")

    summary = session.get_summary()
    print(f"\n{'='*40}")
    print(f"Session Complete!")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Correct: {summary['correct_answers']}")
    print(f"Accuracy: {summary['overall_accuracy']:.0%}")
    print(f"Final Difficulty: {summary['final_difficulty']}")