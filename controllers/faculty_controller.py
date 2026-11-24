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
                "mcq": 5,              # 5 x 1m = 5
                "true_false": 5,       # 5 x 1m = 5
                "fill_in_the_blank": 5,# 5 x 1m = 5
                "short_answer": 5,     # 5 x 2m = 10
                "long_answer": 5       # 5 x 5m = 25
            }                          # Total  = 50
            
        # Exact mapping for 30 Marks (Yields exactly 21 questions total)
        elif total_marks == 30:
            return {
                "mcq": 5,              # 5 x 1m = 5
                "true_false": 5,       # 5 x 1m = 5
                "fill_in_the_blank": 5,# 5 x 1m = 5
                "short_answer": 5,     # 5 x 2m = 10
                "long_answer": 1       # 1 x 5m = 5
            }                          # Total  = 30
            
        # Exact mapping for 20 Marks (Yields exactly 15 questions total)
        elif total_marks == 20:
            return {
                "mcq": 4,              # 4 x 1m = 4
                "true_false": 3,       # 3 x 1m = 3
                "fill_in_the_blank": 3,# 3 x 1m = 3
                "short_answer": 5,     # 5 x 2m = 10
                "long_answer": 0       # 0 x 5m = 0
            }                          # Total  = 20
            
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
    #lesson planner

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

    # --- BULLETPROOF QUESTION PAPER GENERATOR (SKELETON INJECTION + COVERED TOPICS) ---
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
        syllabus: str = "CBSE"
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

        scoping_instruction = f"""
        SYLLABUS & COVERED TOPICS SCOPING ({board_label}):
        - Follow the {board_label} exam pattern and curriculum rigor.
        - Covered Topics List: {json.dumps(covered_topics)}
        - STRICT CONSTRAINT: You MUST generate questions ONLY from the covered topics listed above. NEVER include questions from untaught topics.
        - Distribute questions evenly across all {len(covered_topics)} covered topic(s).
        - Frame every question directly without meta-phrases like "According to the textbook" or "Based on the chapter".
        """
            
        format_clean = str(format_type).lower().strip() if format_type else "all"
        is_mixed_mode = format_clean in ["all", "all type", "all types", "mixed", "mixed type", "optional", "", "none"]
        
        skeleton = []
        q_num = 1

        if is_mixed_mode:
            dist = self._calculate_mixed_distribution(actual_marks)
            
            # Build an unbreakable JSON skeleton that the AI is FORCED to fill out
            for _ in range(dist['mcq']):
                skeleton.append({"q_num": q_num, "type": "mcq", "question": "[FILL_IN_QUESTION]", "options": ["[A] Option", "[B] Option", "[C] Option", "[D] Option"], "marks": 1, "answer": "[FILL_IN_ANSWER]"})
                q_num += 1
            for _ in range(dist['true_false']):
                skeleton.append({"q_num": q_num, "type": "true_false", "question": "[FILL_IN_QUESTION]", "options": ["True", "False"], "marks": 1, "answer": "[FILL_IN_ANSWER]"})
                q_num += 1
            for _ in range(dist['fill_in_the_blank']):
                skeleton.append({"q_num": q_num, "type": "fill_in_the_blank", "question": "[FILL_IN_QUESTION_WITH_BLANK]", "marks": 1, "answer": "[FILL_IN_ANSWER]"})
                q_num += 1
            for _ in range(dist['short_answer']):
                skeleton.append({"q_num": q_num, "type": "short_answer", "question": "[FILL_IN_QUESTION]", "marks": 2, "max_characters": 180, "expected_answer": "[FILL_IN_ANSWER]"})
                q_num += 1
            for _ in range(dist['long_answer']):
                skeleton.append({"q_num": q_num, "type": "long_answer", "question": "[FILL_IN_QUESTION]", "marks": 5, "max_characters": 2000, "expected_answer": "[FILL_IN_ANSWER]"})
                q_num += 1
            
            prompt = f"""
            Act as an expert {board_label} school exam setter. Create a {difficulty} exam paper covering: '{topics_display}'.
            {scoping_instruction}

            CRITICAL ANTI-TRUNCATION INSTRUCTION:
            You MUST complete the following JSON structure EXACTLY as it is provided. 
            Replace every "[FILL_IN_...]" placeholder with a real, high-quality question/answer.
            DO NOT skip, remove, or shorten any part of the array. You must return all {q_num - 1} questions.
            
            Return ONLY valid JSON. Do not include markdown blocks.
            
            {{
                "topic": "{topics_display}",
                "topics": {json.dumps(covered_topics)},
                "syllabus": "{board_label}",
                "difficulty": "{difficulty}",
                "total_marks": {actual_marks},
                "format": "Mixed",
                "questions": {json.dumps(skeleton, indent=4)}
            }}
            """

        elif "true" in format_clean or "false" in format_clean or "tf" in format_clean:
            # Force minimum 15 questions for single formats to ensure proper test length
            count = max(15, actual_marks)
            actual_marks = count
            for i in range(1, count + 1):
                skeleton.append({"q_num": i, "type": "true_false", "question": "[FILL_IN_QUESTION]", "options": ["True", "False"], "marks": 1, "answer": "[FILL_IN_ANSWER]"})
            
            prompt = f"""
            Create a {difficulty} {board_label} exam covering: '{topics_display}'.
            {scoping_instruction}
            
            CRITICAL ANTI-TRUNCATION INSTRUCTION:
            You MUST complete the following JSON structure EXACTLY. Replace all placeholders.
            DO NOT skip or remove items. Return ONLY valid JSON.
            
            {{
                "topic": "{topics_display}",
                "topics": {json.dumps(covered_topics)},
                "syllabus": "{board_label}",
                "difficulty": "{difficulty}",
                "total_marks": {actual_marks},
                "format": "True/False",
                "questions": {json.dumps(skeleton, indent=4)}
            }}
            """

        elif "mcq" in format_clean or "quiz" in format_clean:
            count = max(15, actual_marks)
            actual_marks = count
            for i in range(1, count + 1):
                skeleton.append({"q_num": i, "type": "mcq", "question": "[FILL_IN_QUESTION]", "options": ["[A] Option", "[B] Option", "[C] Option", "[D] Option"], "marks": 1, "answer": "[FILL_IN_ANSWER]"})
                
            prompt = f"""
            Create a {difficulty} {board_label} exam covering: '{topics_display}'.
            {scoping_instruction}
            
            CRITICAL ANTI-TRUNCATION INSTRUCTION:
            You MUST complete the following JSON structure EXACTLY. Replace all placeholders.
            DO NOT skip or remove items. Return ONLY valid JSON.
            
            {{
                "topic": "{topics_display}",
                "topics": {json.dumps(covered_topics)},
                "syllabus": "{board_label}",
                "difficulty": "{difficulty}",
                "total_marks": {actual_marks},
                "format": "MCQ",
                "questions": {json.dumps(skeleton, indent=4)}
            }}
            """

        elif "short" in format_clean:
            count = max(15, actual_marks // 2)
            actual_marks = count * 2
            for i in range(1, count + 1):
                skeleton.append({"q_num": i, "type": "short_answer", "question": "[FILL_IN_QUESTION]", "marks": 2, "max_characters": 180, "expected_answer": "[FILL_IN_ANSWER]"})
                
            prompt = f"""
            Create a {difficulty} {board_label} exam covering: '{topics_display}'.
            {scoping_instruction}
            
            CRITICAL ANTI-TRUNCATION INSTRUCTION:
            You MUST complete the following JSON structure EXACTLY. Replace all placeholders.
            DO NOT skip or remove items. Return ONLY valid JSON.
            
            {{
                "topic": "{topics_display}",
                "topics": {json.dumps(covered_topics)},
                "syllabus": "{board_label}",
                "difficulty": "{difficulty}",
                "total_marks": {actual_marks},
                "format": "Short Answers",
                "questions": {json.dumps(skeleton, indent=4)}
            }}
            """

        elif "fill" in format_clean or "blank" in format_clean:
            count = max(15, actual_marks)
            actual_marks = count
            for i in range(1, count + 1):
                skeleton.append({"q_num": i, "type": "fill_in_the_blank", "question": "[FILL_IN_QUESTION_WITH_BLANK]", "marks": 1, "answer": "[FILL_IN_ANSWER]"})
                
            prompt = f"""
            Create a {difficulty} {board_label} exam covering: '{topics_display}'.
            {scoping_instruction}
            
            CRITICAL ANTI-TRUNCATION INSTRUCTION:
            You MUST complete the following JSON structure EXACTLY. Replace all placeholders.
            DO NOT skip or remove items. Return ONLY valid JSON.
            
            {{
                "topic": "{topics_display}",
                "topics": {json.dumps(covered_topics)},
                "syllabus": "{board_label}",
                "difficulty": "{difficulty}",
                "total_marks": {actual_marks},
                "format": "Fill in the Blanks",
                "questions": {json.dumps(skeleton, indent=4)}
            }}
            """

        response = client.models.generate_content(model=model_name, contents=prompt)
        
        db = SessionLocal()
        try:
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", "Generate Exam")
        finally:
            db.close()
            
        return response.text.replace("```json", "").replace("```", "").strip()

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

    def evaluate_performance(self, performance_data: dict, scope_description: str, user_email: str, client_name: str):
        prompt = f"""
        Evaluate this classroom performance data ({scope_description}): {performance_data}
        Return ONLY valid JSON format.
        Schema: {{"scope": "...", "overall_score": 90, "strengths": ["..."], "weaknesses": ["..."], "recommendation": "..."}}
        """
        response = client.models.generate_content(model=model_name, contents=prompt)
        
        db = SessionLocal()
        try:
            log_gemini_usage(db, response, client_name, user_email, "Faculty Dashboard", "Evaluate Performance")
        finally:
            db.close()
            
        return response.text.replace("```json", "").replace("```", "").strip()

    # --- Competitive Exam Prep Features ---

    def get_competitive_exam_structure(self, exam_type: str, class_level: str, user_email: str, client_name: str):
        prompt = f"""
        You are an expert curriculum designer for Indian competitive exams.
        A student in Class {class_level} wants to start foundation preparation for the '{exam_type}' exam.
        List the required subjects for this exam. For each subject, provide a list of foundational topics appropriate for a Class {class_level} cognitive level.
        
        Return ONLY valid JSON format. Do not use markdown blocks.
        Schema: {{"exam": "{exam_type}", "class": "{class_level}", "subjects": [{{"subject_name": "Biology", "topics": ["Cell Structure", "Tissues"]}}]}}
        """
        response = client.models.generate_content(model=model_name, contents=prompt)
        
        db = SessionLocal()
        try:
            log_gemini_usage(db, response, user_email, "Faculty Dashboard", "Fetch Exam Structure")
        finally:
            db.close()
            
        return response.text.replace("```json", "").replace("```", "").strip()

    def generate_competitive_content(self, exam_type: str, class_level: str, subject: str, topic: str, content_type: str, user_email: str, client_name: str):
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