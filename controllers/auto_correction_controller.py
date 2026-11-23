import json
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

        Parameters:
            answer_sheet_files: List of (file_bytes, mime_type) tuples for scanned answer sheet pages/PDF.
            mode: "direct_grade" (AI grades from scratch) OR "verify_teacher" (AI audits teacher's marks & totaling).
            question_paper_file: Optional (file_bytes, mime_type) tuple of the scanned/PDF question paper.
            question_paper_json: Optional JSON/text string of the question paper & answer key (e.g., from Auto Test).
            rubric_or_answer_key: Optional text marking scheme / rubric instructions.
            subject: Subject name (e.g., "Mathematics", "Science").
            class_level: Student grade/class (e.g., "Class 8").
            max_marks: Total maximum marks of the exam.
            student_name: Student's full name (also cross-checked via OCR from cover page).
            roll_number: Student's roll number.
            user_email: Logged-in faculty email for token usage tracking.
            client_name: Tenant/school identifier for usage tracking.
        """
        if not answer_sheet_files:
            raise ValueError("At least one scanned answer sheet file (PDF or image) is required.")

        norm_mode = (mode or "direct_grade").lower().strip()
        if norm_mode not in {"direct_grade", "verify_teacher"}:
            norm_mode = "direct_grade"

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

        # 4. Mode-Specific Evaluation Instructions
        if norm_mode == "verify_teacher":
            mode_instructions = """
            MODE: TEACHER EVALUATION AUDIT & RE-VERIFICATION ('verify_teacher')
            The attached student answer sheet has ALREADY been checked by a human teacher (look for red/colored pen ticks, crosses, marginal marks per question, and total marks written on the cover/top page).
            Your job is to perform a thorough OCR + academic audit:
            1. Transcribe the student's handwritten answer for every attempted question accurately (including mathematical steps, equations, and diagram labels).
            2. Detect the marks awarded by the teacher for each question (`teacher_awarded_marks`) and the teacher's written grand total (`teacher_written_total`).
            3. Independently evaluate the student's written answer against the question paper/rubric and determine the fair score (`ai_evaluated_marks`).
            4. Compare `teacher_awarded_marks` vs `ai_evaluated_marks` for each question:
               - Mark `discrepancy_flag` as true if the difference is >= 0.5 marks, if a correct step was marked wrong, if an incorrect answer was given full credit, or if a question was left unchecked.
            5. Check the arithmetic addition of the teacher's marks (`totaling_error_detected`: true if the sum of individual teacher marks does not match the grand total written on the front page).
            6. Set overall `audit_status` to one of: "accurate", "under_marked", "over_marked", or "totaling_error".
            """
        else:
            mode_instructions = """
            MODE: DIRECT AI OCR GRADING ('direct_grade')
            The attached student answer sheet is unchecked (or needs fresh AI grading).
            Your job is to:
            1. Perform high-accuracy handwriting OCR to transcribe the student's written answer for every question attempted.
            2. Match each answer to the corresponding question number and maximum marks from the question paper.
            3. Evaluate fairly using step-wise marking (award partial marks for correct formulas, intermediate steps, or key points even if the final answer has a minor calculation slip).
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

        {mode_instructions}

        CRITICAL OUTPUT RULES:
        - Read every page carefully. If a student continued an answer on a later page, combine the full answer under that question number.
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
                    "question_text": "Matched question text from paper",
                    "transcribed_answer": "Exact OCR transcription of student's handwritten answer",
                    "max_marks": 2.0,
                    "ai_evaluated_marks": 1.5,
                    "teacher_awarded_marks": null,
                    "discrepancy_flag": false,
                    "discrepancy_reason": null,
                    "step_wise_breakdown": "1 mark for correct formula, 0.5 mark for substitution, -0.5 for final unit missing",
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

        # Enforce exact mathematical sum of AI marks in Python so totals never drift
        questions_list = parsed_result.get("questions", [])
        if isinstance(questions_list, list) and questions_list:
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