import os
import base64
import json
from typing import List, Optional
from google.cloud import texttospeech
from google.cloud import speech
from google.api_core.client_options import ClientOptions

from config.ai_config import client, model_name
from config.database import SessionLocal
from utils.ai_tracker import log_gemini_usage


class FacultyAIController:
    def __init__(self):
        # 1. Voice to Text and Text to Voice Initialization
        gcp_key = os.getenv("GOOGLE_TTS_API_KEY")
        if not gcp_key:
            raise ValueError("GOOGLE_TTS_API_KEY is missing from environment variables.")

        client_options = ClientOptions(api_key=gcp_key)
        self.tts_client = texttospeech.TextToSpeechClient(client_options=client_options)
        self.stt_client = speech.SpeechClient(client_options=client_options)

        # 2. Premium Language mapping for Standard & Neural2 Google TTS voices
        self.advanced_language_map = {
            "english": {"code": "en-IN", "voice": "en-IN-Neural2-B"},
            "hindi": {"code": "hi-IN", "voice": "hi-IN-Neural2-A"},
            "telugu": {"code": "te-IN", "voice": "te-IN-Standard-A"},
            "kannada": {"code": "kn-IN", "voice": "kn-IN-Standard-A"},
            "tamil": {"code": "ta-IN", "voice": "ta-IN-Standard-A"},
            "malayalam": {"code": "ml-IN", "voice": "ml-IN-Standard-A"},
            "bengali": {"code": "bn-IN", "voice": "bn-IN-Standard-A"},
            "marathi": {"code": "mr-IN", "voice": "mr-IN-Standard-A"},
            "oriya": {"code": "or-IN", "voice": "or-IN-Standard-A"},
            "gujarati": {"code": "gu-IN", "voice": "gu-IN-Standard-A"},
            "gujrati": {"code": "gu-IN", "voice": "gu-IN-Standard-A"}
        }

    def _get_voice_config(self, language_name: str) -> dict:
        lang_lower = language_name.lower().strip() if language_name else "english"
        return self.advanced_language_map.get(lang_lower, {"code": "en-IN", "voice": "en-IN-Neural2-A"})

    # --- PYTHON MATH HELPER FOR DYNAMIC MARKS BREAKDOWN ---
    def _calculate_mixed_distribution(self, total_marks: int) -> dict:
        """Calculates exact question counts mathematically matching the total marks requested."""
        # Exact mapping for 50 Marks (Yields exactly 25 questions total)
        if total_marks == 50:
            return {
                "mcq": 5,               # 5 x 1m = 5
                "true_false": 5,        # 5 x 1m = 5
                "fill_in_the_blank": 5, # 5 x 1m = 5
                "short_answer": 5,      # 5 x 2m = 10
                "long_answer": 5        # 5 x 5m = 25
            }                           # Total  = 50

        # Exact mapping for 30 Marks (Yields exactly 21 questions total)
        elif total_marks == 30:
            return {
                "mcq": 5,               # 5 x 1m = 5
                "true_false": 5,        # 5 x 1m = 5
                "fill_in_the_blank": 5, # 5 x 1m = 5
                "short_answer": 5,      # 5 x 2m = 10
                "long_answer": 1        # 1 x 5m = 5
            }                           # Total  = 30

        # Exact mapping for 20 Marks (Yields exactly 15 questions total)
        elif total_marks == 20:
            return {
                "mcq": 4,               # 4 x 1m = 4
                "true_false": 3,        # 3 x 1m = 3
                "fill_in_the_blank": 3, # 3 x 1m = 3
                "short_answer": 5,      # 5 x 2m = 10
                "long_answer": 0        # 0 x 5m = 0
            }                           # Total  = 20

        # Dynamic calculation for any other odd marks (e.g., 40, 100)
        dist = {"mcq": 0, "true_false": 0, "fill_in_the_blank": 0, "short_answer": 0, "long_answer": 0}
        remaining = total_marks

        # Allocate Long Answers (5 marks each)
        if remaining >= 15:
            long_c = int(remaining * 0.4) // 5
            dist["long_answer"] = long_c
            remaining -= (long_c * 5)

        # Allocate Short Answers (2 marks each)
        if remaining >= 6:
            short_c = int(remaining * 0.4) // 2
            dist["short_answer"] = short_c
            remaining -= (short_c * 2)

        # Distribute remaining points equally among 1-mark questions
        cycle = ["mcq", "true_false", "fill_in_the_blank"]
        idx = 0
        while remaining > 0:
            dist[cycle[idx % 3]] += 1
            remaining -= 1
            idx += 1

        return dist

    # --- SCERT AP (CCE / CBA) BLUEPRINT & SKELETON HELPER ---
    def _is_ap_scert(self, syllabus: str) -> bool:
        s = (syllabus or "").strip().upper()
        return any(token in s for token in ["AP", "SCERT", "ANDHRA", "CCE", "CBA"])

    def _resolve_ap_blueprint(self, assessment_type: str, total_marks: int, grade_level: str) -> dict:
        """
        Resolves the SCERT AP CCE/CBA blueprint for FA (35M written / 50M CCE),
        SA (80M: 60M Descriptive + 20M MCQ), Class 8 split science (40M per paper),
        or Slip Test (3-5 questions).
        """
        kind = (assessment_type or "").strip().upper()
        grade_clean = str(grade_level or "").strip().lower()
        is_class_8 = "8" in grade_clean

        if "SLIP" in kind:
            return {
                "assessment_type": "SLIP_TEST",
                "duration": "1 Period (40 Mins)",
                "marks": max(5, total_marks),
                "cce_breakdown": {
                    "written_slip_test": max(5, total_marks),
                    "mode": "Informal lesson-ending check (3-5 competency questions)"
                },
                "sections": [
                    {
                        "section": "Slip Test",
                        "part": "Lesson-Ending Check",
                        "q_type": "short_answer",
                        "count": 4,
                        "marks_each": max(1, max(5, total_marks) // 4),
                        "has_internal_choice": False,
                        "omr_eligible": False
                    }
                ]
            }

        if kind in ["SA1", "SA2", "SUMMATIVE", "HALF-YEARLY", "ANNUAL", "CBA-3"] or total_marks >= 60:
            sa_label = "SA2 / CBA-3 (Annual Exam)" if kind in ["SA2", "ANNUAL", "CBA-3"] else "SA1 (Summative Assessment 1 — Half-Yearly)"
            if is_class_8 and total_marks == 40:
                # Class 8 Physical Science or Biological Science single 40M paper (30M Descriptive + 10M MCQ)
                return {
                    "assessment_type": sa_label,
                    "duration": "1 Hr 30 Mins",
                    "marks": 40,
                    "cce_breakdown": {
                        "descriptive_part_a": 30,
                        "objective_part_b_mcq": 10,
                        "note": "Class 8 Science splits into separate Physical Science & Biological Science papers (40M each = 80M combined)."
                    },
                    "sections": [
                        {"section": "Part B (Objective)", "part": "Part B", "q_type": "mcq", "count": 10, "marks_each": 1, "has_internal_choice": False, "omr_eligible": True},
                        {"section": "Section I (Very Short Answer)", "part": "Part A", "q_type": "very_short_answer", "count": 3, "marks_each": 2, "has_internal_choice": False, "omr_eligible": False},
                        {"section": "Section II (Short Answer)", "part": "Part A", "q_type": "short_answer", "count": 3, "marks_each": 4, "has_internal_choice": False, "omr_eligible": False},
                        {"section": "Section III (Essay / Long Answer)", "part": "Part A", "q_type": "long_answer", "count": 2, "marks_each": 6, "has_internal_choice": True, "omr_eligible": False},
                    ]
                }
            # Standard 80-mark SA1 / SA2 blueprint (60M Descriptive + 20M MCQs)
            return {
                "assessment_type": sa_label,
                "duration": "2 Hrs 45 Mins to 3 Hrs",
                "marks": 80,
                "cce_breakdown": {
                    "part_a_descriptive": 60,
                    "part_b_objective_mcq": 20,
                    "final_grading_formula": "20% FA average + 80% SA written exam (100 total; A1=91-100 down to E)"
                },
                "sections": [
                    {"section": "Part B — Objective (MCQs)", "part": "Part B", "q_type": "mcq", "count": 20, "marks_each": 1, "has_internal_choice": False, "omr_eligible": True},
                    {"section": "Part A — Section I (Very Short Answer)", "part": "Part A", "q_type": "very_short_answer", "count": 6, "marks_each": 2, "has_internal_choice": False, "omr_eligible": False},
                    {"section": "Part A — Section II (Short Answer)", "part": "Part A", "q_type": "short_answer", "count": 6, "marks_each": 4, "has_internal_choice": False, "omr_eligible": False},
                    {"section": "Part A — Section III (Essay / Long Answer)", "part": "Part A", "q_type": "long_answer", "count": 3, "marks_each": 8, "has_internal_choice": True, "omr_eligible": False},
                ]
            }

        # Default SCERT AP Formative Assessment (FA1–FA4 / CBA-1 / Self-Assessment 2 / CBA-2)
        # Written paper = 35M (1 Hr 15 Mins) + 15M CCE (5M Participation + 5M Written Work + 5M Project Work) = 50M
        fa_label = kind if kind else "FA1 / CBA-1 (Formative Assessment)"
        return {
            "assessment_type": fa_label,
            "duration": "1 Hr 15 Mins (Written Test)",
            "marks": 35,
            "cce_breakdown": {
                "written_test_cba": 35,
                "childrens_participation_reflection": 5,
                "written_work_notebook": 5,
                "project_work": 5,
                "total_fa_marks": 50,
                "scaled_to_term_weightage": 20
            },
            "sections": [
                {"section": "Section A (Multiple Choice Questions)", "part": "Written (35M)", "q_type": "mcq", "count": 15, "marks_each": 1, "has_internal_choice": False, "omr_eligible": False},
                {"section": "Section B (Very Short Answer Questions)", "part": "Written (35M)", "q_type": "very_short_answer", "count": 2, "marks_each": 2, "has_internal_choice": False, "omr_eligible": False},
                {"section": "Section C (Short Answer Questions)", "part": "Written (35M)", "q_type": "short_answer", "count": 2, "marks_each": 4, "has_internal_choice": False, "omr_eligible": False},
                {"section": "Section D (Essay / Long Answer Question)", "part": "Written (35M)", "q_type": "long_answer", "count": 1, "marks_each": 8, "has_internal_choice": True, "omr_eligible": False},
            ]
        }

    def _build_ap_skeleton_and_marks(self, assessment_type: str, total_marks: int, grade_level: str):
        bp = self._resolve_ap_blueprint(assessment_type, total_marks, grade_level)
        skeleton = []
        q_num = 1

        for sec in bp["sections"]:
            for _ in range(sec["count"]):
                marks_each = sec["marks_each"]
                q_type = sec["q_type"]
                item = {
                    "q_num": q_num,
                    "section": sec["section"],
                    "part": sec["part"],
                    "type": q_type,
                    "question": "[FILL_IN_QUESTION]",
                    "marks": marks_each,
                    "internal_choice": sec["has_internal_choice"],
                }
                if q_type == "mcq":
                    item["options"] = ["(A) Option 1", "(B) Option 2", "(C) Option 3", "(D) Option 4"]
                    item["answer"] = "[FILL_IN_ANSWER]"
                elif q_type == "very_short_answer":
                    item["max_characters"] = 180
                    item["expected_answer"] = "[FILL_IN_ANSWER]"
                elif q_type == "short_answer":
                    item["max_characters"] = 400
                    item["expected_answer"] = "[FILL_IN_ANSWER]"
                else:
                    item["max_characters"] = 2000
                    item["expected_answer"] = "[FILL_IN_ANSWER]"
                skeleton.append(item)
                q_num += 1

        # Compute exact sum of marks from skeleton so total_marks is strictly consistent
        computed_marks = sum(item["marks"] for item in skeleton)
        bp["marks"] = computed_marks
        return skeleton, computed_marks, bp

    # --- REAL: Language script translator ---
    def translate_text(self, text: str, target_language: str, user_email: str, client_name: str) -> str:
        prompt = f"Translate to {target_language} (use native script only). Return ONLY the translation. Text: {text}"
        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", "Translate Text")
        finally:
            db.close()

        return response.text.strip()

    # --- REAL: Text to Voice ---
    def generate_speech(self, text: str, language: str = "English") -> str:
        voice_config = self._get_voice_config(language)
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(language_code=voice_config["code"], name=voice_config["voice"])
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=0.75, sample_rate_hertz=24000)

        response = self.tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return base64.b64encode(response.audio_content).decode("utf-8")

    # --- REAL: Voice to Text ---
    def process_audio_to_text(self, audio_bytes: bytes, language: str = "English") -> str:
        voice_config = self._get_voice_config(language)
        audio = speech.RecognitionAudio(content=audio_bytes)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            language_code=voice_config["code"]
        )

        response = self.stt_client.recognize(config=config, audio=audio)
        if not response.results:
            return ""

        return " ".join([result.alternatives[0].transcript for result in response.results])

    # --- REAL: Audio language translator ---
    def audio_language_translator(self, audio_bytes: bytes, source_language: str, target_language: str, user_email: str, client_name: str) -> dict:
        original_text = self.process_audio_to_text(audio_bytes, source_language)
        if not original_text.strip():
            return {"original_text": "", "translated_text": "Please provide valid audio to translate.", "audio_base64": ""}

        translated_text = self.translate_text(original_text, target_language, user_email, client_name)
        translated_audio_base64 = self.generate_speech(translated_text, target_language)
        return {"original_text": original_text, "translated_text": translated_text, "audio_base64": translated_audio_base64}

    # --- Core Faculty Specific AI Methods ---
    # lesson planner

    def generate_teaching_material(
        self,
        topic: str,
        grade_level: str,
        user_email: str,
        client_name: str,
        syllabus: str = "CBSE",
        textbook_content: str = "",
        teacher_name: str = "Not specified",
        designation: str = "Teacher",
        section: str = "Not specified",
        subject: str = "Not specified",
        no_of_periods: str = "Not specified",
        date_of_commencement: str = "Not specified",
        expected_completion: str = "Not specified",
        actual_completion: str = "Not specified"
    ):
        board_label = syllabus.strip() if syllabus and syllabus.strip() else "CBSE"
        content_block = (
            f'You MUST base your entire lesson plan STRICTLY on the following textbook content:\n"""\n{textbook_content}\n"""'
            if textbook_content and textbook_content.strip()
            else f'Align strictly with the official {board_label} curriculum standards and textbook scope for Class {grade_level} {subject}.'
        )

        prompt = f"""
You are an expert, highly experienced school teacher following the {board_label} syllabus. Your task is to write a comprehensive and deeply detailed lesson plan for the single current topic "{topic}".
{content_block}

CRITICAL INSTRUCTIONS:
1. Act like a real teacher planning a detailed, practical lesson aligned with {board_label} pedagogical guidelines. NO generic filler. Write exact teaching strategies, specific real-world examples, and precise questions you will ask.
2. Focus strictly on the single current topic "{topic}". Do not drift into unrelated chapters or untaught topics.
3. Format the output EXACTLY matching the structure below. Do not add any extra headings, JSON, or markdown code blocks.
4. Fill in the bracketed areas with deep, faculty-level detail.

{teacher_name}
LESSON PLAN

Name of the teacher: {teacher_name}
Designation: {designation}
Syllabus / Board: {board_label}
Class & Section: {grade_level} {section}
Subject: {subject}
Chapter/Topic: {topic}
No. Of Periods: {no_of_periods}
Date of Commencement: {date_of_commencement}
Expected date of completion: {expected_completion}
Actual date of completion: {actual_completion}

Learning objectives:
[Write 3-4 specific, measurable objectives using action verbs aligned with {board_label} standards. E.g., "Students will be able to calculate...", "Students will identify..."]

Learning outcomes:
[Write 3-4 specific outcomes detailing exactly what students will demonstrate by the end of the lesson.]

Methodology:
[Provide a detailed, step-by-step teaching method. E.g., "1. Hook: Start by showing... 2. Direct Instruction: Explain the concept using [specific analogy]. 3. Guided Practice: Solve problem X together..."]

TLM:
[List specific Teaching Learning Materials. E.g., "Textbook page X, Smartboard presentation on Y, physical props like Z"]

Activities:
[Detail 1-2 specific classroom activities. E.g., "Think-Pair-Share: Students will pair up to solve...", or "Group Activity: Groups of 4 will analyze..."]

Assessment:
[List 2-3 exact questions you will ask to check understanding, or describe a specific exit ticket task.]

Home Work:
[Give a specific, actionable homework assignment related to the exact topic "{topic}".]

Sign. of the Teacher                                        Sign. of the Dean.
"""

        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            # Log the AI generation usage
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", "Generate Lesson Plan")

            # Note: You can add an INSERT statement here to log the lesson plan to your SSS DB
            # if you need to store it permanently, just like the SGS Node.js version does.

        finally:
            db.close()

        return response.text.replace("```text", "").replace("```", "").strip()

    # --- ACADEMIC SUBJECT PLANNER (CLASS 1 TO 12 - 4 EXAM MILESTONE GRID) ---

    def generate_academic_planner(
        self,
        grade_level: str,
        subject: str,
        chapters: Optional[List[str]] = None,
        total_chapters: int = 12,
        syllabus: str = "CBSE",
        academic_year: str = "2026-27",
        total_teaching_periods: int = 120,
        exam_config: Optional[List[dict]] = None,
        user_email: str = "",
        client_name: str = "SSS"
    ) -> dict:
        """
        Generates a 4-column Academic Subject Planner (Module/Term | Course Contents | Contact Periods | Mapped LO & Exam Pattern)
        for Classes 1 to 12 across the 4 school assessment milestones (20%, 50%, 70%, 100%).
        """
        board_label = syllabus.strip() if syllabus and syllabus.strip() else "CBSE"
        clean_chapters = [c.strip() for c in (chapters or []) if isinstance(c, str) and c.strip()]
        num_chapters = len(clean_chapters) if clean_chapters else max(4, int(total_chapters or 12))

        # Default 4-exam school structure if custom exam_config is not passed
        milestones = exam_config if (isinstance(exam_config, list) and len(exam_config) > 0) else [
            {"term_no": 1, "exam_name": "Sessional Exam 1", "cumulative_percent": 20, "marks": "20-30 Marks", "lo_code": "LO1"},
            {"term_no": 2, "exam_name": "Half-Yearly Exam", "cumulative_percent": 50, "marks": "50-70 Marks", "lo_code": "LO2"},
            {"term_no": 3, "exam_name": "Sessional Exam 2", "cumulative_percent": 70, "marks": "20-30 Marks", "lo_code": "LO3"},
            {"term_no": 4, "exam_name": "Annual Examination", "cumulative_percent": 100, "marks": "70 Marks", "lo_code": "LO4"},
        ]

        # Deterministic cumulative chapter slicing (e.g., 12 chapters -> 2, 6, 8, 12)
        chapter_labels = (
            clean_chapters
            if clean_chapters
            else [f"Chapter {i}" for i in range(1, num_chapters + 1)]
        )

        partition_plan = []
        prev_end = 0
        for idx, m in enumerate(milestones):
            pct = int(m.get("cumulative_percent", 100))
            if idx == len(milestones) - 1 or pct >= 100:
                end_idx = num_chapters
            else:
                end_idx = max(prev_end + 1, round((pct / 100.0) * num_chapters))
                end_idx = min(end_idx, num_chapters)

            newly_added = chapter_labels[prev_end:end_idx]
            cumulative_list = chapter_labels[:end_idx]
            weight_share = max(0.1, (end_idx - prev_end) / float(num_chapters))
            periods_allocated = max(4, round(total_teaching_periods * weight_share))

            partition_plan.append({
                "module_no": m.get("term_no", idx + 1),
                "exam_name": m.get("exam_name", f"Term {idx + 1}"),
                "cumulative_percent": pct,
                "marks": m.get("marks", "30 Marks"),
                "newly_taught_chapters": newly_added,
                "cumulative_exam_chapters": cumulative_list,
                "contact_periods": periods_allocated,
                "mapped_lo": m.get("lo_code", f"LO{idx + 1}")
            })
            prev_end = end_idx

        chapter_source_note = (
            f"Use the exact chapter titles provided: {json.dumps(clean_chapters)}."
            if clean_chapters
            else f"Use the standard official {board_label} textbook chapter titles and detailed sub-topics for {grade_level} {subject} (total {num_chapters} chapters)."
        )

        prompt = f"""
        You are an expert K-12 Academic Curriculum Planner for Indian schools ({board_label} Board).
        Generate a complete, structured Subject-Wise Academic Planner for:
- Class / Grade: {grade_level}
- Subject: {subject}
- Syllabus / Board: {board_label}
- Academic Year: {academic_year}
- Total Chapters: {num_chapters}
- Total Teaching Periods: {total_teaching_periods}

        {chapter_source_note}

        PRE-CALCULATED 4-EXAM CUMULATIVE SYLLABUS PARTITION (FOLLOW STRICTLY):
        {json.dumps(partition_plan, indent=2)}

        INSTRUCTIONS:
1. Create one structured row/module per exam milestone (Modules 1 to {len(partition_plan)}).
2. Under "course_contents", list every newly taught chapter for that module along with its comma-separated key sub-topics and concepts (formatted cleanly like an official university/school syllabus table), followed by a clear note stating the cumulative syllabus scope for the exam (e.g., "Exam Scope (50% Cumulative): Chapters 1 to 6").
3. Under "contact_periods", format as "<number> Periods" (matching the pre-calculated periods).
4. Under "mapped_outcome", specify the Learning Outcome code (LO1–LO4), exam name, syllabus percentage, marks weightage, and a concise 1-line competency outcome.
5. Also provide 3–4 standard textbook/reference readings under "suggested_readings".
6. Return ONLY valid JSON matching the exact schema below. Do NOT include markdown backticks.

        REQUIRED JSON SCHEMA:
        {{
          "header": {{
            "school_board": "{board_label}",
            "academic_year": "{academic_year}",
            "class_level": "{grade_level}",
            "subject": "{subject}",
            "total_chapters": {num_chapters},
            "total_periods": {total_teaching_periods}
          }},
          "columns": [
            "Module / Exam Term",
            "Course Contents (Chapters & Sub-Topics)",
            "Contact Periods",
            "Mapped LO & Exam Blueprint"
          ],
          "modules": [
            {{
              "module_no": 1,
              "exam_name": "Sessional Exam 1",
              "cumulative_percent": 20,
              "marks": "20-30 Marks",
              "newly_taught_chapters": ["Chapter 1...", "Chapter 2..."],
              "cumulative_exam_chapters": ["Chapter 1...", "Chapter 2..."],
              "course_contents": "Unit 1 — Chapter Title: Sub-topic 1, Sub-topic 2, Sub-topic 3...\\nUnit 2 — Chapter Title: Sub-topic 1, Sub-topic 2...\\n[Exam Scope: 20% Cumulative Syllabus — Chapters 1 to 2]",
              "contact_periods": "20 Periods",
              "mapped_lo": "LO1",
              "mapped_outcome_details": "LO1 | Sessional Exam 1 (20-30 Marks, 20% Syllabus): Foundational conceptual understanding and recall."
            }}
          ],
          "suggested_readings": [
            "Official {board_label} {grade_level} {subject} Textbook",
            "Exemplar Problems & Lab/Activity Manual for {grade_level} {subject}"
          ]
        }}
        """

        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(
                db,
                response,
                client_name,
                user_email,
                "Faculty Dashboard",
                f"Generate Academic Subject Planner ({grade_level} {subject})"
            )
        finally:
            db.close()

        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        return json.loads(cleaned)

    # --- BULLETPROOF QUESTION PAPER GENERATOR (SKELETON INJECTION + COVERED TOPICS + SCERT AP CCE/CBA) ---
    def generate_question_paper(
        self,
        topic: str = "",
        difficulty: str = "Medium",
        format_type: str = "all",
        num_questions: int = 0,
        user_email: str = "",
        client_name: str = "",
        total_marks: int = 50,
        topics: Optional[List[str]] = None,
        syllabus: str = "CBSE",
        assessment_type: str = "",
        grade_level: str = "",
        subject: str = ""
    ):
        actual_marks = int(num_questions) if (int(num_questions) > 0 and int(total_marks) == 50) else int(total_marks)
        if actual_marks <= 0:
            actual_marks = 50

        # Normalize covered-topics list (`topics: ["A Square and A Cube", ...]`) with fallback to single `topic`
        covered_topics = [t.strip() for t in (topics or []) if isinstance(t, str) and t.strip()]
        if not covered_topics and topic and topic.strip():
            covered_topics = [topic.strip()]

        topics_display = ", ".join(covered_topics) if covered_topics else "General Syllabus"
        board_label = syllabus.strip() if syllabus and syllabus.strip() else "CBSE"
        clean_client = client_name.strip() if client_name and client_name.strip() else "SSS"

        is_ap_mode = self._is_ap_scert(board_label)
        ap_blueprint = None

        if is_ap_mode:
            skeleton, actual_marks, ap_blueprint = self._build_ap_skeleton_and_marks(
                assessment_type=assessment_type,
                total_marks=actual_marks,
                grade_level=grade_level
            )
            format_label = f"AP SCERT CCE/CBA ({ap_blueprint['assessment_type']})"
            scoping_instruction = f"""
        SYLLABUS & COVERED TOPICS SCOPING ({board_label} — Continuous & Comprehensive Evaluation / CBA):
- Follow the official Andhra Pradesh State Syllabus (SCERT AP) CCE / CBA competency framework and academic standards (conceptual understanding, problem-solving, application, and diagram/observation skills).
- Assessment Type: {ap_blueprint['assessment_type']} | Duration: {ap_blueprint['duration']} | Total Marks: {actual_marks}.
- Grade / Class Context: {grade_level or 'Not specified'} | Subject: {subject or 'Not specified'}.
  * Note for Class 7 vs Class 8 Science: Class 7 uses combined General Science; Class 8 splits into Physical Science (PS) and Biological Science (BS).
- Covered Topics List: {json.dumps(covered_topics)}
- STRICT CONSTRAINT: You MUST generate questions ONLY from the covered topics listed above. NEVER include questions from untaught topics.
- Distribute questions evenly across all {len(covered_topics)} covered topic(s).
- Frame every question directly without meta-phrases like "According to the textbook" or "Based on the chapter".
        """
        else:
            scoping_instruction = f"""
        SYLLABUS & COVERED TOPICS SCOPING ({board_label}):
- Follow the {board_label} exam pattern and curriculum rigor.
- Covered Topics List: {json.dumps(covered_topics)}
- STRICT CONSTRAINT: You MUST generate questions ONLY from the covered topics listed above. NEVER include questions from untaught topics.
- Distribute questions evenly across all {len(covered_topics)} covered topic(s).
- Frame every question directly without meta-phrases like "According to the textbook" or "Based on the chapter".
        """

            format_clean = str(format_type).lower().strip() if format_type else "all"
            is_mixed_mode = format_clean in [
                "all", "all type", "all types", "mixed", "mixed type", "optional", "", "none"
            ]

            skeleton = []
            q_num = 1
            format_label = "Mixed"

            if is_mixed_mode:
                dist = self._calculate_mixed_distribution(actual_marks)
                format_label = "Mixed"

                for _ in range(dist["mcq"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "mcq",
                        "question": "[FILL_IN_QUESTION]",
                        "options": ["[A] Option", "[B] Option", "[C] Option", "[D] Option"],
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1
                for _ in range(dist["true_false"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "true_false",
                        "question": "[FILL_IN_QUESTION]",
                        "options": ["True", "False"],
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1
                for _ in range(dist["fill_in_the_blank"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "fill_in_the_blank",
                        "question": "[FILL_IN_QUESTION_WITH_BLANK]",
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1
                for _ in range(dist["short_answer"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "short_answer",
                        "question": "[FILL_IN_QUESTION]",
                        "marks": 2,
                        "max_characters": 180,
                        "expected_answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1
                for _ in range(dist["long_answer"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "long_answer",
                        "question": "[FILL_IN_QUESTION]",
                        "marks": 5,
                        "max_characters": 2000,
                        "expected_answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1

            elif "true" in format_clean or "false" in format_clean or "tf" in format_clean:
                count = max(15, actual_marks)
                actual_marks = count
                format_label = "True/False"
                for i in range(1, count + 1):
                    skeleton.append({
                        "q_num": i,
                        "type": "true_false",
                        "question": "[FILL_IN_QUESTION]",
                        "options": ["True", "False"],
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })

            elif "mcq" in format_clean or "quiz" in format_clean:
                count = max(15, actual_marks)
                actual_marks = count
                format_label = "MCQ"
                for i in range(1, count + 1):
                    skeleton.append({
                        "q_num": i,
                        "type": "mcq",
                        "question": "[FILL_IN_QUESTION]",
                        "options": ["[A] Option", "[B] Option", "[C] Option", "[D] Option"],
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })

            elif "short" in format_clean:
                count = max(15, actual_marks // 2)
                actual_marks = count * 2
                format_label = "Short Answers"
                for i in range(1, count + 1):
                    skeleton.append({
                        "q_num": i,
                        "type": "short_answer",
                        "question": "[FILL_IN_QUESTION]",
                        "marks": 2,
                        "max_characters": 180,
                        "expected_answer": "[FILL_IN_ANSWER]"
                    })

            elif "fill" in format_clean or "blank" in format_clean:
                count = max(15, actual_marks)
                actual_marks = count
                format_label = "Fill in the Blanks"
                for i in range(1, count + 1):
                    skeleton.append({
                        "q_num": i,
                        "type": "fill_in_the_blank",
                        "question": "[FILL_IN_QUESTION_WITH_BLANK]",
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })

            elif "long" in format_clean:
                count = max(5, actual_marks // 5)
                actual_marks = count * 5
                format_label = "Long Answers"
                for i in range(1, count + 1):
                    skeleton.append({
                        "q_num": i,
                        "type": "long_answer",
                        "question": "[FILL_IN_QUESTION]",
                        "marks": 5,
                        "max_characters": 2000,
                        "expected_answer": "[FILL_IN_ANSWER]"
                    })

            else:
                # Safe fallback to mixed mode so `prompt` is always initialized
                dist = self._calculate_mixed_distribution(actual_marks)
                format_label = "Mixed"
                for _ in range(dist["mcq"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "mcq",
                        "question": "[FILL_IN_QUESTION]",
                        "options": ["[A] Option", "[B] Option", "[C] Option", "[D] Option"],
                        "marks": 1,
                        "answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1
                for _ in range(dist["short_answer"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "short_answer",
                        "question": "[FILL_IN_QUESTION]",
                        "marks": 2,
                        "max_characters": 180,
                        "expected_answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1
                for _ in range(dist["long_answer"]):
                    skeleton.append({
                        "q_num": q_num,
                        "type": "long_answer",
                        "question": "[FILL_IN_QUESTION]",
                        "marks": 5,
                        "max_characters": 2000,
                        "expected_answer": "[FILL_IN_ANSWER]"
                    })
                    q_num += 1

        total_items = len(skeleton)
        extra_metadata_block = ""
        extra_json_fields = ""
        if is_ap_mode and ap_blueprint:
            extra_metadata_block = f"""
        AP SCERT BLUEPRINT & CCE METADATA (INCLUDE IN JSON):
        {json.dumps(ap_blueprint, indent=2)}
        """
            extra_json_fields = f',\n            "ap_blueprint": {json.dumps(ap_blueprint)}'

        prompt = f"""
        Act as an expert {board_label} school exam setter. Create a {difficulty} exam paper covering: '{topics_display}'.
        {scoping_instruction}
        {extra_metadata_block}

        CRITICAL ANTI-TRUNCATION INSTRUCTION:
        You MUST complete the following JSON structure EXACTLY as it is provided.
        Replace every "[FILL_IN_...]" placeholder with a real, high-quality question/answer.
        For items with "internal_choice": true, provide a clear "OR" alternative inside the question text or expected answer.
        DO NOT skip, remove, or shorten any part of the array. You must return all {total_items} questions.

        Return ONLY valid JSON. Do not include markdown blocks.

        {{
            "topic": "{topics_display}",
            "topics": {json.dumps(covered_topics)},
            "syllabus": "{board_label}",
            "difficulty": "{difficulty}",
            "total_marks": {actual_marks},
            "format": "{format_label}"{extra_json_fields},
            "questions": {json.dumps(skeleton, indent=4)}
        }}
        """

        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, clean_client, user_email, "Faculty Dashboard", "Generate Exam")
        finally:
            db.close()

        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        # Validate JSON structure before returning
        parsed = json.loads(cleaned)
        if is_ap_mode and ap_blueprint and "ap_blueprint" not in parsed:
            parsed["ap_blueprint"] = ap_blueprint
        return json.dumps(parsed)

    def auto_correct_answer(self, question: str, student_answer: str, rubric: str, user_email: str, client_name: str):
        prompt = f"""
        Grade this student answer based on the rubric.
        Question: {question} | Student Answer: {student_answer} | Rubric: {rubric}
        Return ONLY valid JSON format.
        Schema: {{"score": 85, "feedback": "...", "areas_for_improvement": ["...", "..."]}}
        """
        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", "Auto Correct")
        finally:
            db.close()

        return response.text.replace("```json", "").replace("```", "").strip()

    def generate_teacher_alert(self, alert_type: str, details: dict, user_email: str, client_name: str):
        prompt = f"Draft a short, professional alert to a teacher regarding {alert_type}. Details: {details}"
        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", "Generate Alert")
        finally:
            db.close()

        return response.text.strip()

    def _resolve_evaluation_mode(self, performance_data: dict, scope_description: str, report_type: str = "") -> str:
        """
        Determines whether the evaluation is for an 'individual_student' or 'whole_class'.
        Supports explicit report_type/mode keys or inference from scope_description and payload keys.
        """
        explicit = str(report_type or "").strip().lower()
        if not explicit and isinstance(performance_data, dict):
            explicit = str(
                performance_data.get("report_type")
                or performance_data.get("evaluation_type")
                or performance_data.get("mode")
                or ""
            ).strip().lower()

        if explicit in {"individual", "student", "individual_student", "student_report"}:
            return "individual_student"
        if explicit in {"class", "whole_class", "class_overview", "all_students"}:
            return "whole_class"

        scope_lower = (scope_description or "").lower()
        if any(k in scope_lower for k in ["whole class", "entire class", "class overview", "all students", "class teacher"]):
            return "whole_class"

        if isinstance(performance_data, dict):
            has_student_identity = any(
                performance_data.get(k) for k in ["student_name", "student_id", "roll_number", "student"]
            )
            has_students_list = isinstance(performance_data.get("students"), list) and len(performance_data.get("students")) > 0
            if has_students_list and not has_student_identity:
                return "whole_class"
            if has_student_identity and not has_students_list:
                return "individual_student"

        return "individual_student"

    def evaluate_performance(
        self,
        performance_data: dict,
        scope_description: str,
        user_email: str,
        client_name: str = "SSS",
        report_type: str = ""
    ):
        """
        Generates a balanced, encouraging, and confident performance evaluation report using
        student marks, assessment results, and assignment submissions.

        Supports two modes:
        1. Individual Student Report: Categorizes chapters into strong, average, and weak chapters
           based on marks/assessments/assignments, and highlights where to focus more.
        2. Whole Class Report (Class Teacher View across all subjects): Groups students into good,
           average, and needs-support ("bad"/struggling) tiers across ALL subjects (not a single
           subject), evaluating assessments, marks, and assignment submissions.
        """
        clean_client = client_name.strip() if client_name and client_name.strip() else "SSS"
        mode = self._resolve_evaluation_mode(performance_data, scope_description, report_type=report_type)
        serialized_data = json.dumps(performance_data, indent=2, default=str)

        tone_and_rules = """
        TONE & BALANCED REPORTING RULES (CRITICAL):
        - Write in a warm, soft, supportive, and confident teacher tone—never harsh, judgmental, or discouraging.
        - Always present a balanced picture covering ALL THREE categories (Strong/Good, Average, and Weak/Needs-Support). Never show only negative/weak points.
        - Base the evaluation strictly on the provided marks, assessment results, and assignment submissions (completion, timeliness, and quality).
        - Return ONLY valid JSON. Do NOT wrap the output in markdown code blocks or add extra text.
        """

        if mode == "whole_class":
            prompt = f"""
            You are a caring, experienced class teacher preparing a comprehensive whole-class academic report.
            As a class teacher, evaluate performance across ALL subjects present in the data (not limited to a single subject), combining assessment results, exam/test marks, and assignment submissions.

            Scope: {scope_description}
            Performance Data (marks, assessments, assignment submissions across subjects):
            {serialized_data}

            {tone_and_rules}

            CLASS EVALUATION REQUIREMENTS:
            1. Cover ALL subjects represented across assessments, marks, and assignment submissions.
            2. Categorize students into three balanced groups based on combined assessment marks and assignment submission performance:
               - "good_students": Students performing strongly across subjects and keeping up well with assignments.
               - "average_students": Students showing steady, moderate progress with room for growth.
               - "needs_support_students": Students currently facing challenges or lower scores/submissions (referred to gently and constructively, never harshly).
            3. Provide a subject-wise summary across all subjects and clear, encouraging next steps for the class teacher.

            REQUIRED JSON SCHEMA:
            {{
                "report_type": "whole_class",
                "scope": "{scope_description}",
                "subjects_covered": ["Subject 1", "Subject 2"],
                "overall_score": 0,
                "tone": "warm, soft, and confident",
                "student_categories": {{
                    "good_students": [
                        {{
                            "student_name": "...",
                            "summary": "Encouraging note on strong assessment marks and consistent assignment submissions across subjects"
                        }}
                    ],
                    "average_students": [
                        {{
                            "student_name": "...",
                            "summary": "Confident note on steady progress in assessments and assignments with clear growth potential"
                        }}
                    ],
                    "needs_support_students": [
                        {{
                            "student_name": "...",
                            "summary": "Gentle, supportive note on where extra guidance in assessments or assignments will help them grow"
                        }}
                    ]
                }},
                "subject_overview": [
                    {{
                        "subject": "...",
                        "highlight": "Balanced observation across assessments, marks, and assignment submissions"
                    }}
                ],
                "strengths": ["..."],
                "weaknesses": ["..."],
                "recommendation": "Actionable, encouraging class-teacher plan across all subjects, assessments, and assignments"
            }}
            """
        else:
            prompt = f"""
            You are a caring, encouraging teacher preparing an individual student academic report.
            Evaluate the student's performance using their marks, assessment results, and assignment submissions, breaking down chapters clearly into strong chapters, average chapters, and weak chapters, along with focused areas for improvement.

            Scope: {scope_description}
            Student Performance Data (marks, assessments, assignment submissions, chapters):
            {serialized_data}

            {tone_and_rules}

            INDIVIDUAL STUDENT EVALUATION REQUIREMENTS:
            1. Analyze performance across chapters using marks, assessment results, and assignment submissions.
            2. Clearly classify chapters into:
               - "strong_chapters": Chapters where the student scored high marks / performed well in assessments and assignments.
               - "average_chapters": Chapters with moderate/steady performance.
               - "weak_chapters": Chapters where the student got lower marks or needs more practice (presented softly and constructively).
            3. Highlight "focus_areas" explaining specifically where the student should focus more to turn weak and average chapters into strengths.

            REQUIRED JSON SCHEMA:
            {{
                "report_type": "individual_student",
                "scope": "{scope_description}",
                "student_name": "...",
                "overall_score": 0,
                "tone": "warm, soft, and confident",
                "chapter_breakdown": {{
                    "strong_chapters": [
                        {{
                            "chapter": "...",
                            "subject": "...",
                            "performance_note": "Why this chapter is strong based on marks, assessments, or assignments"
                        }}
                    ],
                    "average_chapters": [
                        {{
                            "chapter": "...",
                            "subject": "...",
                            "performance_note": "Steady progress note based on marks, assessments, or assignments"
                        }}
                    ],
                    "weak_chapters": [
                        {{
                            "chapter": "...",
                            "subject": "...",
                            "performance_note": "Gentle note on lower marks or gaps in assessments/assignments"
                        }}
                    ]
                }},
                "focus_areas": [
                    "Specific chapter, concept, or assignment habit to focus on next"
                ],
                "strengths": ["..."],
                "weaknesses": ["..."],
                "recommendation": "Warm, confident, and practical guidance for the student"
            }}
            """

        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, clean_client, user_email, "Faculty Dashboard", "Evaluate Performance")
        finally:
            db.close()

        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        return cleaned
    
    # --- Competitive Exam Prep Features ---

    def get_competitive_exam_structure(
        self,
        exam_type: str,
        class_level: str,
        user_email: str,
        client_name: str = "SSS",
        syllabus: str = "CBSE"
    ):
        board_label = syllabus.strip() if syllabus and syllabus.strip() else "CBSE"
        clean_client = client_name.strip() if client_name and client_name.strip() else "SSS"

        prompt = f"""
        You are an expert curriculum designer for Indian competitive exams aligned with school syllabi.
        A student in Class {class_level} under the {board_label} syllabus wants foundation preparation for the '{exam_type}' exam.
        List the relevant subjects for this exam and, for each subject, provide foundational topics that match BOTH:
        1. The Class {class_level} cognitive level and curriculum scope, AND
        2. The {board_label} syllabus textbook/chapter coverage for Class {class_level}.
        Do NOT include topics from higher classes or unrelated boards.

        Return ONLY valid JSON format. Do not use markdown blocks.
        Schema: {{"exam": "{exam_type}", "class": "{class_level}", "syllabus": "{board_label}", "subjects": [{{"subject_name": "Biology", "topics": ["Cell Structure", "Tissues"]}}]}}
        """
        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, clean_client, user_email, "Faculty Dashboard", "Fetch Exam Structure")
        finally:
            db.close()

        return response.text.replace("```json", "").replace("```", "").strip()

    def generate_competitive_content(
        self,
        exam_type: str,
        class_level: str,
        subject: str,
        topic: str,
        content_type: str,
        user_email: str,
        client_name: str = "SSS",
        syllabus: str = "CBSE"
    ):
        board_label = syllabus.strip() if syllabus and syllabus.strip() else "CBSE"
        clean_client = client_name.strip() if client_name and client_name.strip() else "SSS"

        prompt = f"""
        Act as an expert coaching instructor for the {exam_type} exam in India.
        Create '{content_type}' for a student in Class {class_level} following the {board_label} syllabus, focusing on the subject '{subject}' and the topic '{topic}'.
        Strict rules:
        - Calibrate concepts, terminology, and question depth specifically to Class {class_level} foundation level.
        - Align strictly with the {board_label} syllabus standards and textbook scope for Class {class_level} {subject}.
        - Do NOT introduce out-of-syllabus or higher-grade concepts beyond Class {class_level} {board_label}.

        Return ONLY valid JSON format. Do not use markdown blocks.
        """

        if content_type.lower() == "quiz":
            prompt += f"""
            Schema: {{"exam": "{exam_type}", "class": "{class_level}", "syllabus": "{board_label}", "topic": "{topic}", "questions": [{{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_answer": "...", "explanation": "..."}}]}}
            """
        else:
            prompt += f"""
            Schema: {{"exam": "{exam_type}", "class": "{class_level}", "syllabus": "{board_label}", "topic": "{topic}", "core_concepts": ["...", "..."], "detailed_notes": "...", "important_formulas_or_facts": ["...", "..."]}}
            """

        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, clean_client, user_email, "Faculty Dashboard", f"Generate Competitive {content_type}")
        finally:
            db.close()

        return response.text.replace("```json", "").replace("```", "").strip()
        prompt = f"""
        Act as an expert coaching instructor for the {exam_type} exam in India.
        Create '{content_type}' for a student in Class {class_level} focusing on the subject '{subject}' and the topic '{topic}'.
        Since the student is in Class {class_level}, ensure the difficulty is at a 'Foundation' level—building core concepts that will help them later in their actual {exam_type} exam.

        Return ONLY valid JSON format. Do not use markdown blocks.
        """

        if content_type.lower() == "quiz":
            prompt += f"""
            Schema: {{"exam": "{exam_type}", "topic": "{topic}", "questions": [{{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_answer": "...", "explanation": "..."}}]}}
            """
        else:
            prompt += f"""
            Schema: {{"exam": "{exam_type}", "topic": "{topic}", "core_concepts": ["...", "..."], "detailed_notes": "...", "important_formulas_or_facts": ["...", "..."]}}
            """

        response = client.models.generate_content(model=model_name, contents=prompt)

        db = SessionLocal()
        try:
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", f"Generate Competitive {content_type}")
        finally:
            db.close()

        return response.text.replace("```json", "").replace("```", "").strip()