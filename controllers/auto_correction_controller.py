import json
import re
from typing import List, Optional, Tuple
from google.genai import types

from config.ai_config import client, model_name
from config.database import SessionLocal
from utils.ai_tracker import log_gemini_usage


class AutoCorrectionController:
    """
    Multimodal OCR & AI Answer Sheet Auto-Correction / Verification Controller.
    Supports:
    1. Direct AI Grading ('direct_grade'): Scans raw student answer sheets + question paper/rubric and awards marks.
    2. Teacher Evaluation Audit ('verify_teacher'): Scans teacher-checked answer sheets, verifies teacher's marks & totaling, and flags discrepancies.
    """

    def __init__(self):
        self.supported_mime_types = {
            "application/pdf",
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }
        # Keywords identifying language-specific subjects where cross-language answers are disallowed
        self.language_subject_keywords = {
            "english",
            "hindi",
            "telugu",
            "sanskrit",
            "urdu",
            "tamil",
            "kannada",
            "malayalam",
            "marathi",
            "bengali",
            "gujarati",
            "punjabi",
            "odia",
            "oriya",
            "french",
            "german",
            "spanish",
            "arabic",
            "language",
            "literature",
        }

    def _validate_file(self, file_bytes: bytes, mime_type: str, label: str = "Answer sheet") -> types.Part:
        if not file_bytes:
            raise ValueError(f"{label} file is empty.")
        clean_mime = (mime_type or "application/pdf").lower().strip()
        if clean_mime == "image/jpg":
            clean_mime = "image/jpeg"
        if clean_mime not in self.supported_mime_types:
            raise ValueError(
                f"Unsupported file format '{clean_mime}' for {label}. Allowed: PDF, JPG, PNG, WEBP."
            )
        return types.Part.from_bytes(data=file_bytes, mime_type=clean_mime)

    def _clean_json_response(self, raw_text: str) -> str:
        cleaned = (
            raw_text.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    def _is_language_subject(self, subject: str) -> bool:
        subj_lower = (subject or "").lower().strip()
        if not subj_lower:
            return False
        tokens = set(re.findall(r"[a-z]+", subj_lower))
        return bool(tokens & self.language_subject_keywords)

    def _classify_question_length_tier(self, question_item: dict) -> Optional[dict]:
        """
        Determines minimum character threshold based on question type / max_marks:
        - Short Answer (e.g. 2-3 marks or type='short_answer'): min 200 characters (target 200-250+)
        - Long Answer (e.g. 4-6 marks or type='long_answer'): min 900 characters
        - Very Long / Essay Answer (e.g. >= 8 marks or type='very_long_answer'/'essay'): min 1800 characters
        Objective questions (MCQ, True/False, Fill in the blank, 1-mark) are exempt.
        """
        q_type = str(question_item.get("question_type") or "").lower().strip()
        try:
            q_max = float(question_item.get("max_marks") or 0)
        except (TypeError, ValueError):
            q_max = 0.0

        # Exempt objective / 1-mark items
        if q_type in {"mcq", "true_false", "fill_in_the_blank", "objective", "one_word"} or q_max <= 1.0:
            return None

        if q_type in {"very_long_answer", "essay"} or q_max >= 8.0:
            return {"tier": "very_long_answer", "min_chars": 1800}
        if q_type in {"long_answer"} or q_max >= 4.0:
            return {"tier": "long_answer", "min_chars": 900}
        if q_type in {"short_answer", "very_short_answer"} or (1.0 < q_max < 4.0):
            return {"tier": "short_answer", "min_chars": 200}

        return None

    def _apply_post_grading_rules(
        self,
        questions_list: list,
        subject: str,
        is_language_subj: bool,
        norm_mode: str,
    ) -> list:
        """
        Deterministically enforces:
        1. Duplicate/repeated question attempts: keeps the first uncrossed attempt and ignores subsequent duplicates.
        2. Cross-language rejection for language subjects (0 marks + explicit reason).
        3. Minimum character length caps (halves the mark if below threshold: 200 for short, 900 for long, 1800 for very long).
        """
        seen_questions = set()
        processed_questions = []

        for q in questions_list:
            q_num_raw = str(q.get("question_number") or "").strip()
            q_num_norm = re.sub(r"[^a-zA-Z0-9]", "", q_num_raw).lower()

            # Rule 3a: Duplicate attempt guard — if the model still emitted a second entry for the same question number
            if q_num_norm and q_num_norm in seen_questions:
                q["duplicate_attempt_ignored"] = True
                q["ai_evaluated_marks"] = 0.0
                existing_fb = (q.get("feedback") or "").strip()
                dup_note = (
                    f"Duplicate attempt for Question {q_num_raw} ignored: only the first valid (uncrossed) attempt is evaluated."
                )
                q["feedback"] = f"{dup_note} {existing_fb}".strip()
                if norm_mode == "verify_teacher" and q.get("teacher_awarded_marks") is not None:
                    if float(q.get("teacher_awarded_marks") or 0) > 0:
                        q["discrepancy_flag"] = True
                        q["discrepancy_reason"] = (
                            q.get("discrepancy_reason")
                            or "Teacher awarded marks on a duplicate attempt of an already-answered question."
                        )
                processed_questions.append(q)
                continue

            if q_num_norm:
                seen_questions.add(q_num_norm)

            try:
                max_m = float(q.get("max_marks") or 0)
            except (TypeError, ValueError):
                max_m = 0.0

            try:
                ai_m = float(q.get("ai_evaluated_marks") or 0)
            except (TypeError, ValueError):
                ai_m = 0.0

            transcribed = (q.get("transcribed_answer") or "").strip()
            language_violation = bool(q.get("language_violation"))

            # Rule 1: Language subject enforcement — writing in a different language gives 0 marks
            if is_language_subj and language_violation:
                ai_m = 0.0
                q["ai_evaluated_marks"] = 0.0
                detected_lang = (q.get("detected_answer_language") or "another language").strip()
                violation_msg = (
                    f"Marked incorrect (0 marks): {subject} paper answered in {detected_lang} is not allowed."
                )
                existing_fb = (q.get("feedback") or "").strip()
                if violation_msg.lower() not in existing_fb.lower():
                    q["feedback"] = f"{violation_msg} {existing_fb}".strip()
                q["step_wise_breakdown"] = f"0/{max_m:g} — {violation_msg}"
                if norm_mode == "verify_teacher" and q.get("teacher_awarded_marks") is not None:
                    if float(q.get("teacher_awarded_marks") or 0) > 0:
                        q["discrepancy_flag"] = True
                        q["discrepancy_reason"] = violation_msg
                processed_questions.append(q)
                continue

            # Rule 2: Minimum character length enforcement (Short: 200, Long: 900, Very Long: 1800)
            # Only apply when student actually attempted the answer (ai_m > 0 and non-empty text)
            tier_info = self._classify_question_length_tier(q)
            char_count = len(transcribed)
            q["answer_character_count"] = char_count

            if tier_info and ai_m > 0 and max_m > 0:
                min_chars = tier_info["min_chars"]
                tier_label = tier_info["tier"].replace("_", " ").title()
                if char_count < min_chars:
                    half_cap = round(max_m * 0.5, 2)
                    if ai_m > half_cap:
                        ai_m = half_cap
                        q["ai_evaluated_marks"] = ai_m
                    length_note = (
                        f"Length penalty applied ({tier_label}): answer length is {char_count} characters, "
                        f"which is below the minimum {min_chars}-character requirement; capped at half marks ({half_cap:g}/{max_m:g})."
                    )
                    existing_fb = (q.get("feedback") or "").strip()
                    if "below the minimum" not in existing_fb.lower():
                        q["feedback"] = f"{existing_fb} {length_note}".strip()
                    existing_bd = (q.get("step_wise_breakdown") or "").strip()
                    if "capped at half marks" not in existing_bd.lower():
                        q["step_wise_breakdown"] = (
                            f"{existing_bd} | {length_note}" if existing_bd else length_note
                        )
                    if norm_mode == "verify_teacher" and q.get("teacher_awarded_marks") is not None:
                        t_marks = float(q.get("teacher_awarded_marks") or 0)
                        if abs(t_marks - ai_m) >= 0.5:
                            q["discrepancy_flag"] = True
                            q["discrepancy_reason"] = q.get("discrepancy_reason") or length_note

            processed_questions.append(q)

        return processed_questions

    def evaluate_answer_sheet(
        self,
        answer_sheet_files: List[Tuple[bytes, str]],
        mode: str = "direct_grade",
        question_paper_file: Optional[Tuple[bytes, str]] = None,
        question_paper_json: Optional[str] = None,
        rubric_or_answer_key: Optional[str] = None,
        subject: str = "General",
        class_level: str = "Not specified",
        max_marks: int = 50,
        student_name: str = "Unknown Student",
        roll_number: str = "",
        user_email: str = "",
        client_name: str = ""
    ) -> dict:
        """
        Main entry point for OCR scanning, grading, and teacher-audit verification.
        """
        if not answer_sheet_files:
            raise ValueError("At least one scanned answer sheet file (PDF or image) is required.")

        norm_mode = (mode or "direct_grade").lower().strip()
        if norm_mode not in {"direct_grade", "verify_teacher"}:
            norm_mode = "direct_grade"

        is_language_subj = self._is_language_subject(subject)

        contents = []

        # 1. Attach Question Paper File if provided as PDF/Image
        if question_paper_file and question_paper_file[0]:
            qp_part = self._validate_file(
                question_paper_file[0], question_paper_file[1], label="Question paper"
            )
            contents.append("REFERENCE DOCUMENT 1 — OFFICIAL QUESTION PAPER:")
            contents.append(qp_part)

        # 2. Attach all scanned Student Answer Sheet pages/PDFs
        contents.append("REFERENCE DOCUMENT 2 — SCANNED STUDENT ANSWER SHEET(S):")
        for idx, (sheet_bytes, sheet_mime) in enumerate(answer_sheet_files, start=1):
            sheet_part = self._validate_file(
                sheet_bytes, sheet_mime, label=f"Answer sheet page {idx}"
            )
            contents.append(f"--- Answer Sheet File/Page {idx} ---")
            contents.append(sheet_part)

        # 3. Build Reference Text Block (Question Paper JSON + Rubric)
        reference_text_sections = []
        if question_paper_json and str(question_paper_json).strip():
            reference_text_sections.append(
                f"OFFICIAL QUESTION PAPER & ANSWER KEY DATA:\n{str(question_paper_json).strip()}"
            )
        if rubric_or_answer_key and str(rubric_or_answer_key).strip():
            reference_text_sections.append(
                f"TEACHER MARKING RUBRIC / GUIDELINES:\n{str(rubric_or_answer_key).strip()}"
            )

        reference_context = (
            "\n\n".join(reference_text_sections)
            if reference_text_sections
            else "No separate text rubric provided; infer question text and standard academic rubric directly from the attached question paper and answer sheet."
        )

        # 4. Language Policy Block (Language Subject vs. Content Subject)
        if is_language_subj:
            language_policy_instructions = f"""
            LANGUAGE POLICY — STRICT LANGUAGE SUBJECT ('{subject}'):
- '{subject}' is a language subject. The student MUST write answers strictly in {subject} (except where a question explicitly asks to translate from/to another language).
- If the student writes an answer in a DIFFERENT language (for example, answering an English paper in Telugu/Hindi, or a Hindi/Telugu/Sanskrit paper in English or another script):
  1. Set `"language_violation": true` and `"detected_answer_language": "<language used>"`.
  2. Mark that answer completely INCORRECT and award `0` marks (`"ai_evaluated_marks": 0.0`).
  3. Explicitly state in `feedback` why it was rejected (e.g., `"{subject} paper answered in <detected_language> is not allowed."`).
            """
        else:
            language_policy_instructions = f"""
            LANGUAGE POLICY — CONTENT / NON-LANGUAGE SUBJECT ('{subject}'):
- '{subject}' is a non-language subject (e.g., Science, Mathematics, Social Science).
- Bilingual or code-mixed answers (such as Telugu + English, Hindi + English, or regional script mixed with English technical terms) ARE ALLOWED.
- Scan and transcribe the mixed-language handwriting accurately, evaluate the conceptual and factual correctness across both languages, and award full/partial marks strictly on academic merit without penalizing mixed-language usage.
- Set `"language_violation": false`.
            """

        # 5. Answer Length & Duplicate Attempt Rules
        grading_policy_instructions = """
        MANDATORY ANSWER LENGTH & DUPLICATE ATTEMPT RULES:
        A. Minimum Character Length Rule (for descriptive answers):
           - Classify each question into `"question_type"`: `"objective"` (1M MCQ/T-F/Fill), `"short_answer"` (2–3M), `"long_answer"` (4–6M), or `"very_long_answer"` (8M+ / Essay).
           - Short Answer (`"short_answer"`): minimum 200–250 characters (enforce at least 200 characters). If the student's transcribed answer has fewer than 200 characters, award at most HALF of `max_marks` (e.g., for a 2-mark question, give at most 1 mark) even if the content is accurate.
           - Long Answer (`"long_answer"`): minimum 900 characters. If fewer than 900 characters, award at most HALF of `max_marks`.
           - Very Long Answer / Essay (`"very_long_answer"`): minimum 1800 characters. If fewer than 1800 characters, award at most HALF of `max_marks`.
           - Objective / 1-mark questions have no minimum character requirement.

        B. Repeated / Multiple Attempts for the Same Question Number:
           - Scan the entire answer sheet carefully for cases where the student wrote an answer for the same question number more than once.
           - If the FIRST written attempt is NOT crossed/struck out: evaluate ONLY the FIRST attempt and completely ignore the second/subsequent attempt (do NOT add marks from the second attempt, and do NOT emit a second object for that question number). Note in `feedback` that the repeated second attempt was ignored.
           - If the FIRST written attempt IS crossed out / struck through by the student: ignore the crossed-out first attempt, evaluate the SECOND (uncrossed) attempt normally, and award marks based on that second attempt. Set `"attempted_multiple_times": true` and `"evaluated_attempt_note"` explaining which attempt was graded.
        """

        # 6. Mode-Specific Evaluation Instructions
        if norm_mode == "verify_teacher":
            mode_instructions = """
            MODE: TEACHER EVALUATION AUDIT & RE-VERIFICATION ('verify_teacher')
            The attached student answer sheet has ALREADY been checked by a human teacher (look for red/colored pen ticks, crosses, marginal marks per question, and total marks written on the cover/top page).
            Your job is to perform a thorough OCR + academic audit:
1. Transcribe the student's handwritten answer for every attempted question accurately (including mathematical steps, equations, mixed-language text, and diagram labels), honoring the crossed-out / first-attempt rule above.
2. Detect the marks awarded by the teacher for each question (`teacher_awarded_marks`) and the teacher's written grand total (`teacher_written_total`).
3. Independently evaluate the student's written answer against the question paper/rubric and all mandatory policies above to determine the fair score (`ai_evaluated_marks`).
4. Compare `teacher_awarded_marks` vs `ai_evaluated_marks` for each question:
  - Mark `discrepancy_flag` as true if the difference is >= 0.5 marks, if a correct step was marked wrong, if an incorrect/wrong-language/short-length answer was given full credit, or if a question was left unchecked.
5. Check the arithmetic addition of the teacher's marks (`totaling_error_detected`: true if the sum of individual teacher marks does not match the grand total written on the front page).
6. Set overall `audit_status` to one of: "accurate", "under_marked", "over_marked", or "totaling_error".
            """
        else:
            mode_instructions = """
            MODE: DIRECT AI OCR GRADING ('direct_grade')
            The attached student answer sheet is unchecked (or needs fresh AI grading).
            Your job is to:
1. Perform high-accuracy handwriting OCR to transcribe the student's written answer for every question attempted, honoring the crossed-out / first-attempt rule above.
2. Match each answer to the corresponding question number and maximum marks from the question paper.
3. Evaluate fairly using step-wise marking while strictly applying the Language Policy and Minimum Character Length rules above.
4. Provide clear, constructive feedback for every question explaining why marks were awarded or deducted.
5. Leave `teacher_awarded_marks` and `teacher_written_total` as null, `discrepancy_flag` as false, and `audit_status` as "direct_graded".
            """

        prompt = f"""
        You are an expert school examiner and OCR grading engine for Indian school examinations.
        Subject: {subject}
        Class / Grade: {class_level}
        Maximum Marks: {max_marks}
        Student Name (Provided): {student_name}
        Roll Number (Provided): {roll_number}

        {reference_context}

        {language_policy_instructions}

        {grading_policy_instructions}

        {mode_instructions}

        CRITICAL OUTPUT RULES:
- Read every page carefully. If a student continued a single answer on a later page, combine the full answer under that question number.
- Emit at most ONE entry per question number in `questions`.
- If handwriting is partially illegible, note "[illegible word]" in `transcribed_answer` and grade based on readable context.
- Ensure `ai_total_marks` equals the exact sum of `ai_evaluated_marks` across all questions.
- Return ONLY valid JSON matching the exact schema below. Do NOT include markdown backticks or extra text.

        REQUIRED JSON SCHEMA:
        {{
            "mode": "{norm_mode}",
            "student_details": {{
                "detected_name": "Name read from sheet or '{student_name}'",
                "detected_roll_no": "Roll number read from sheet or '{roll_number}'",
                "subject": "{subject}",
                "class_level": "{class_level}"
            }},
            "summary": {{
                "max_marks": {max_marks},
                "ai_total_marks": 0.0,
                "teacher_written_total": null,
                "calculated_teacher_sum": null,
                "score_difference": 0.0,
                "percentage": 0.0,
                "audit_status": "direct_graded | accurate | under_marked | over_marked | totaling_error",
                "totaling_error_detected": false,
                "overall_feedback": "Concise summary of student performance and grading observations"
            }},
            "questions": [
                {{
                    "question_number": "1",
                    "question_type": "objective | short_answer | long_answer | very_long_answer",
                    "question_text": "Matched question text from paper",
                    "transcribed_answer": "Exact OCR transcription of the evaluated attempt",
                    "detected_answer_language": "English | Telugu | Hindi | Mixed (Telugu + English) | etc.",
                    "language_violation": false,
                    "attempted_multiple_times": false,
                    "evaluated_attempt_note": "Single attempt | Graded 1st attempt (2nd attempt ignored) | 1st attempt crossed out, graded 2nd attempt",
                    "max_marks": 2.0,
                    "ai_evaluated_marks": 1.0,
                    "teacher_awarded_marks": null,
                    "discrepancy_flag": false,
                    "discrepancy_reason": null,
                    "step_wise_breakdown": "Breakdown of marks awarded and any length/language deductions",
                    "feedback": "Constructive feedback for the student/teacher"
                }}
            ]
        }}
        """

        contents.append(prompt)

        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )

        db = SessionLocal()
        try:
            action_label = (
                "Verify Checked Answer Sheet"
                if norm_mode == "verify_teacher"
                else "Auto-Grade Answer Sheet"
            )
            log_gemini_usage(
                db,
                response,
                client_name,
                user_email,
                "Faculty Dashboard",
                action_label
            )
        finally:
            db.close()

        cleaned_json = self._clean_json_response(response.text)
        parsed_result = json.loads(cleaned_json)

        # Apply deterministic post-grading rules (duplicate guard, language rejection, character length caps)
        questions_list = parsed_result.get("questions", [])
        if isinstance(questions_list, list) and questions_list:
            questions_list = self._apply_post_grading_rules(
                questions_list=questions_list,
                subject=subject,
                is_language_subj=is_language_subj,
                norm_mode=norm_mode,
            )
            parsed_result["questions"] = questions_list

            exact_ai_sum = round(
                sum(float(q.get("ai_evaluated_marks") or 0) for q in questions_list), 2
            )
            summary = parsed_result.setdefault("summary", {})
            summary["ai_total_marks"] = exact_ai_sum
            if max_marks > 0:
                summary["percentage"] = round((exact_ai_sum / float(max_marks)) * 100, 2)

            if norm_mode == "verify_teacher":
                teacher_marks_list = [
                    float(q["teacher_awarded_marks"])
                    for q in questions_list
                    if q.get("teacher_awarded_marks") is not None
                ]
                if teacher_marks_list:
                    calc_teacher_sum = round(sum(teacher_marks_list), 2)
                    summary["calculated_teacher_sum"] = calc_teacher_sum
                    written_total = summary.get("teacher_written_total")
                    if written_total is not None and abs(float(written_total) - calc_teacher_sum) >= 0.5:
                        summary["totaling_error_detected"] = True
                        summary["audit_status"] = "totaling_error"
                    summary["score_difference"] = round(exact_ai_sum - calc_teacher_sum, 2)

        return parsed_result