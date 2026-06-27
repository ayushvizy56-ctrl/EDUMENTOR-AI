from modules.content_generator.quiz_generator import generate_quiz
import random


class QuestionBank:
    """
    Manages a pool of questions at different difficulty levels.
    Generates new questions on demand using the AI quiz generator.
    """

    def __init__(self, topic_text: str, topic_name: str):
        self.topic_text  = topic_text
        self.topic_name  = topic_name
        self.bank        = {level: [] for level in ["very_easy", "easy", "medium", "hard", "very_hard"]}
        self.used        = set()   # track which questions have been asked

        # Difficulty → quiz generator difficulty mapping
        self.diff_map = {
            "very_easy": "easy",
            "easy":      "easy",
            "medium":    "medium",
            "hard":      "hard",
            "very_hard": "hard",
        }

    def get_question(self, difficulty: str) -> dict | None:
        """
        Returns a question at the requested difficulty level.
        Generates new ones if the bank is running low.
        """
        available = [
            q for i, q in enumerate(self.bank[difficulty])
            if i not in self.used
        ]

        # Refill if fewer than 2 questions left at this level
        if len(available) < 2:
            self._generate_questions(difficulty, count=5)
            available = self.bank[difficulty]

        if not available:
            return None

        question = random.choice(available)
        self.used.add(id(question))
        return question

    def _generate_questions(self, difficulty: str, count: int = 5):
        """Call the AI quiz generator to create new questions."""
        gen_difficulty = self.diff_map.get(difficulty, "medium")

        # Adjust prompt for very_easy and very_hard
        topic_text = self.topic_text
        if difficulty == "very_easy":
            topic_text = "Basic introduction: " + topic_text[:500]
        elif difficulty == "very_hard":
            topic_text = "Advanced analysis: " + topic_text

        new_questions = generate_quiz(
            topic_text,
            self.topic_name,
            num_questions=count,
            difficulty=gen_difficulty
        )
        self.bank[difficulty].extend(new_questions)
        print(f"Generated {len(new_questions)} {difficulty} questions")