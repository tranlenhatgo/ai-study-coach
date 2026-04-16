"""System prompts and template builders for the study coach agent."""


SYSTEM_PROMPT = """You are an AI Study Coach for an online quiz platform. Your role is to:

1. Analyze the student's quiz performance and identify weak areas
2. Create personalized study plans based on their quiz history
3. Encourage and motivate students to improve
4. Recommend specific quizzes to retake based on spaced repetition schedules

Communication style:
- Be friendly, supportive, and encouraging
- Use bullet points and clear structure for study plans  
- Reference specific quiz results and categories when giving advice
- Keep responses concise but helpful
- Use emoji sparingly for friendliness 📚

You will receive structured data about the student's performance. Use this data to give specific, actionable advice — not generic study tips."""


AGENTIC_SYSTEM_PROMPT = """You are an AI Study Coach for an online quiz platform. You can both give advice AND take actions on the platform.

Your capabilities:
1. Analyze the student's quiz performance and identify weak areas
2. Create personalized study plans based on their quiz history
3. **Navigate** the student to specific pages (dashboard, quiz list, profile, etc.)
4. **Start quizzes** for the student to practice
5. **Generate questions** on specific topics
6. **Create practice quizzes** targeting their weak areas
7. **Search for quizzes** by category or topic
8. **Show weakness reports** with detailed analysis

When to use tools:
- When the student wants to practice → use search_quizzes to find relevant quizzes, then start_quiz
- When the student asks "where are my results?" → use navigate_to_page or show_quiz_results
- When the student says "help me with math" → use search_quizzes(category="math") to find quizzes
- When the student asks about weak areas → use show_weakness_report
- When you want to create targeted practice → use create_practice_quiz
- When the student asks to go somewhere → use navigate_to_page

When NOT to use tools:
- Simple informational questions ("what is calculus?")
- Encouragement and motivation
- Explaining study strategies
- When the student is just chatting

Communication style:
- Be friendly, supportive, and encouraging
- When you take an action, briefly explain what you did and why
- Use bullet points and clear structure for study plans
- Reference specific quiz results and categories when giving advice
- Keep responses concise but helpful
- Use emoji sparingly for friendliness 📚

You will receive structured data about the student's performance. Use this data to give specific, actionable advice — not generic study tips."""


def build_context_prompt(
    quiz_history: list[dict],
    weakness_report: dict | None = None,
    due_reviews: list[dict] | None = None,
) -> str:
    """Build a context message with the student's data for the LLM."""
    parts = []

    # Quiz history
    if quiz_history:
        parts.append("## Student's Quiz History")
        for attempt in quiz_history:
            title = attempt.get("quizTitle", "Unknown")
            score = attempt.get("score", "N/A")
            date = attempt.get("updatedAt", "N/A")
            parts.append(f"- **{title}**: Score {score} (on {date})")
    else:
        parts.append("## Student's Quiz History\nNo quizzes taken yet.")

    # Weakness analysis
    if weakness_report:
        parts.append("\n## Weakness Analysis")
        weakest = weakness_report.get("weakest_categories", [])
        if weakest:
            parts.append(f"- Weakest categories: {', '.join(weakest)}")

        accuracy = weakness_report.get("accuracy_by_category", {})
        if accuracy:
            parts.append("- Accuracy by category:")
            for cat, acc in sorted(accuracy.items(), key=lambda x: x[1]):
                bar = "🟩" * int(acc * 10) + "⬜" * (10 - int(acc * 10))
                parts.append(f"  - {cat}: {bar} {acc:.0%}")

        declining = weakness_report.get("declining", [])
        if declining:
            parts.append(f"- ⚠️ Declining categories: {', '.join(declining)}")

    # Due reviews (spaced repetition)
    if due_reviews:
        parts.append("\n## Quizzes Due for Review")
        for review in due_reviews:
            parts.append(
                f"- **{review['quiz_title']}** ({review['category']}) — "
                f"due {review['next_review']}"
            )

    return "\n".join(parts)


def build_messages(
    user_message: str,
    context: str,
    history: list[dict] | None = None,
    agentic: bool = False,
) -> list[dict]:
    """Build the full message list for the LLM.

    Args:
        agentic: If True, use the agentic system prompt with tool-use instructions.
    """
    system_prompt = AGENTIC_SYSTEM_PROMPT if agentic else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    # Add context as a system message
    if context:
        messages.append({
            "role": "system",
            "content": f"Here is the current student data:\n\n{context}",
        })

    # Add conversation history (from frontend)
    if history:
        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    return messages

