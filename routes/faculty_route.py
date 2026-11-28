import json
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from controllers.faculty_controller import FacultyAIController
from controllers.auto_correction_controller import AutoCorrectionController

router = APIRouter()
faculty_controller = FacultyAIController()
auto_correction_controller = AutoCorrectionController()

# --- Schemas (Updated with SCERT AP CCE/CBA, Syllabus-Aligned Competitive Prep & Performance Report Modes) ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str
    user_email: str
    client_name: str = "SSS"

class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    user_email: str
    client_name: str = "SSS"

class MaterialRequest(BaseModel):
    topic: str
    grade_level: str
    user_email: str
    client_name: str = "SSS"
    syllabus: str = "CBSE"  # "State" | "CBSE" | "SCERT AP"
    textbook_content: str = ""
    teacher_name: str = "Not specified"
    designation: str = "Teacher"
    section: str = "Not specified"
    subject: str = "Not specified"
    no_of_periods: str = "Not specified"
    date_of_commencement: str = "Not specified"
    expected_completion: str = "Not specified"
    actual_completion: str = "Not specified"

class QuestionPaperRequest(BaseModel):
    topics: List[str] = []           # Covered topics list e.g. ["A Square and A Cube", ...]
    topic: Optional[str] = ""        # Optional fallback if single topic is sent
    syllabus: str = "CBSE"           # "State" | "CBSE" | "SCERT AP"
    difficulty: str = "Medium"
    format_type: str = "all"
    num_questions: int = 0
    total_marks: int = 50
    assessment_type: str = ""        # e.g., "FA1", "FA2", "SA1", "FA3", "FA4", "SA2", "SLIP_TEST"
    grade_level: str = ""            # e.g., "Class 7", "Class 8"
    subject: str = ""                # e.g., "Physical Science", "Biological Science", "General Science"
    user_email: str
    client_name: str = "SSS"

class AutoCorrectRequest(BaseModel):
    question: str
    student_answer: str
    rubric: str
    user_email: str
    client_name: str = "SSS"

class AlertRequest(BaseModel):
    alert_type: str 
    details: dict 
    user_email: str
    client_name: str = "SSS"

class AssessmentRequest(BaseModel):
    scope_description: str 
    performance_data: dict
    report_type: str = ""            # Optional: "individual_student" | "whole_class" (auto-detected if omitted)
    user_email: str
    client_name: str = "SSS"

class CompetitiveStructureRequest(BaseModel):
    exam_type: str
    class_level: str
    syllabus: str = "CBSE"           # "CBSE" | "SCERT AP" | "State"
    user_email: str
    client_name: str = "SSS"

class CompetitiveContentRequest(BaseModel):
    exam_type: str
    class_level: str
    subject: str
    topic: str
    content_type: str                # 'Quiz' or 'Study Material'
    syllabus: str = "CBSE"           # "CBSE" | "SCERT AP" | "State"
    user_email: str
    client_name: str = "SSS"

class AcademicPlannerRequest(BaseModel):
    grade_level: str                          # e.g., "Class 8" or "Class 10" (Class 1 to 12)
    subject: str                              # e.g., "Social Studies"
    chapters: List[str] = []                  # Optional list of exact chapter names; if empty, AI uses standard board chapters
    total_chapters: int = 12                  # Total chapters in subject (default 12)
    syllabus: str = "CBSE"                    # "State" | "CBSE"
    academic_year: str = "2026-27"
    total_teaching_periods: int = 120
    exam_config: Optional[List[dict]] = None  # Optional custom exam percentages/marks override
    user_email: str
    client_name: str = "SSS"

# --- Endpoints ---

@router.post("/translate")
async def translate_script(req: TranslationRequest):
    try:
        translated = faculty_controller.translate_text(req.text, req.target_language, req.user_email, req.client_name)
        return {"status": "success", "translated_text": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-to-voice")
async def text_to_voice(req: TTSRequest):
    try:
        return {"status": "success", "audio_base64": faculty_controller.generate_speech(req.text, req.language)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice-to-text")
async def voice_to_text(
    file: UploadFile = File(...), 
    language: str = Form("English"),
    user_email: str = Form(...),
    client_name: str = Form("SSS")
):
    try:
        audio_bytes = await file.read()
        return {"status": "success", "transcription": faculty_controller.process_audio_to_text(audio_bytes, language)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio-translator")
async def audio_translator(
    file: UploadFile = File(...), 
    source_language: str = Form("English"), 
    target_language: str = Form(...),
    user_email: str = Form(...),
    client_name: str = Form("SSS")
):
    try:
        audio_bytes = await file.read()
        result = faculty_controller.audio_language_translator(audio_bytes, source_language, target_language, user_email, client_name)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-material")
async def generate_material(req: MaterialRequest):
    try:
        material_str = faculty_controller.generate_teaching_material(
            topic=req.topic, 
            grade_level=req.grade_level, 
            user_email=req.user_email, 
            client_name=req.client_name,
            syllabus=req.syllabus,
            textbook_content=req.textbook_content,
            teacher_name=req.teacher_name,
            designation=req.designation,
            section=req.section,
            subject=req.subject,
            no_of_periods=req.no_of_periods,
            date_of_commencement=req.date_of_commencement,
            expected_completion=req.expected_completion,
            actual_completion=req.actual_completion
        )
        return {"status": "success", "teaching_material": material_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-exam")
async def generate_exam(req: QuestionPaperRequest):
    try:
        exam_str = faculty_controller.generate_question_paper(
            topic=req.topic or "",
            difficulty=req.difficulty,
            format_type=req.format_type,
            num_questions=req.num_questions,
            user_email=req.user_email,
            client_name=req.client_name,
            total_marks=req.total_marks,
            topics=req.topics,
            syllabus=req.syllabus,
            assessment_type=req.assessment_type,
            grade_level=req.grade_level,
            subject=req.subject
        )
        return {"status": "success", "question_paper": json.loads(exam_str)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auto-correct")
async def auto_correct(req: AutoCorrectRequest):
    try:
        feedback_str = faculty_controller.auto_correct_answer(req.question, req.student_answer, req.rubric, req.user_email, req.client_name)
        return {"status": "success", "grading_feedback": json.loads(feedback_str)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auto-correct/scan")
async def scan_and_evaluate_answer_sheet(
    answer_sheets: List[UploadFile] = File(..., description="Scanned student answer sheet PDF or image pages"),
    question_paper_file: Optional[UploadFile] = File(None, description="Optional scanned/PDF question paper"),
    question_paper_json: Optional[str] = Form(None, description="Optional JSON/text of question paper & answer key"),
    rubric_or_answer_key: Optional[str] = Form(None, description="Optional marking rubric or answer key"),
    mode: str = Form("direct_grade", description="'direct_grade' or 'verify_teacher'"),
    subject: str = Form("General"),
    class_level: str = Form("Not specified"),
    max_marks: int = Form(50),
    student_name: str = Form("Unknown Student"),
    roll_number: str = Form(""),
    user_email: str = Form(...),
    client_name: str = Form("SSS")
):
    try:
        sheet_payloads = []
        for sheet in answer_sheets:
            file_bytes = await sheet.read()
            sheet_payloads.append((file_bytes, sheet.content_type or "application/pdf"))

        qp_payload = None
        if question_paper_file:
            qp_bytes = await question_paper_file.read()
            if qp_bytes:
                qp_payload = (qp_bytes, question_paper_file.content_type or "application/pdf")

        evaluation = auto_correction_controller.evaluate_answer_sheet(
            answer_sheet_files=sheet_payloads,
            mode=mode,
            question_paper_file=qp_payload,
            question_paper_json=question_paper_json,
            rubric_or_answer_key=rubric_or_answer_key,
            subject=subject,
            class_level=class_level,
            max_marks=max_marks,
            student_name=student_name,
            roll_number=roll_number,
            user_email=user_email,
            client_name=client_name
        )
        return {"status": "success", "evaluation": evaluation}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts")
async def generate_alert(req: AlertRequest):
    try:
        alert = faculty_controller.generate_teacher_alert(req.alert_type, req.details, req.user_email, req.client_name)
        return {"status": "success", "alert_message": alert}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assess")
async def evaluate_performance(req: AssessmentRequest):
    try:
        assessment_str = faculty_controller.evaluate_performance(
            performance_data=req.performance_data,
            scope_description=req.scope_description,
            user_email=req.user_email,
            client_name=req.client_name,
            report_type=req.report_type
        )
        return {"status": "success", "assessment_report": json.loads(assessment_str)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/competitive/structure")
async def get_competitive_structure(req: CompetitiveStructureRequest):
    try:
        structure_str = faculty_controller.get_competitive_exam_structure(
            exam_type=req.exam_type,
            class_level=req.class_level,
            user_email=req.user_email,
            client_name=req.client_name,
            syllabus=req.syllabus
        )
        return {"status": "success", "exam_structure": json.loads(structure_str)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/competitive/generate")
async def generate_competitive_content(req: CompetitiveContentRequest):
    try:
        content_str = faculty_controller.generate_competitive_content(
            exam_type=req.exam_type,
            class_level=req.class_level,
            subject=req.subject,
            topic=req.topic,
            content_type=req.content_type,
            user_email=req.user_email,
            client_name=req.client_name,
            syllabus=req.syllabus
        )
        return {"status": "success", "competitive_content": json.loads(content_str)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/academic-planner")
async def generate_academic_subject_planner(req: AcademicPlannerRequest):
    try:
        planner_data = faculty_controller.generate_academic_planner(
            grade_level=req.grade_level,
            subject=req.subject,
            chapters=req.chapters,
            total_chapters=req.total_chapters,
            syllabus=req.syllabus,
            academic_year=req.academic_year,
            total_teaching_periods=req.total_teaching_periods,
            exam_config=req.exam_config,
            user_email=req.user_email,
            client_name=req.client_name
        )
        return {"status": "success", "academic_planner": planner_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))