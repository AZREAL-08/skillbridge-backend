import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app, DB_JDS, DB_SESSIONS

@pytest.mark.anyio
async def test_full_pathway_flow():
    # 1. Setup Mock Data in "Database"
    jd_id = "jd_test_123"
    DB_JDS[jd_id] = {
        "jd_id": jd_id,
        "role_title": "Software Engineer",
        "company": "TestCorp",
        "department": "Engineering",
        "domain": "technical",
        "raw_text": "Looking for a Python developer with SQL knowledge.",
        "required_skills": [
            {"taxonomy_id": "python-001", "label": "Python", "mastery_score": 0.0, "confidence_score": 1.0, "taxonomy_source": "emsi"},
            {"taxonomy_id": "sql-001", "label": "SQL", "mastery_score": 0.0, "confidence_score": 1.0, "taxonomy_source": "emsi"}
        ],
        "created_at": "2026-03-21T00:00:00"
    }

    session_id = "sess_test_456"
    DB_SESSIONS[session_id] = {
        "raw_resume_text": "I am a developer with Python skills but no SQL.",
        "extracted_skills": [
            {"taxonomy_id": "python-001", "label": "Python", "mastery_score": 0.9, "confidence_score": 0.9, "taxonomy_source": "emsi"},
            {"taxonomy_id": "sql-001", "label": "SQL", "mastery_score": 0.1, "confidence_score": 0.9, "taxonomy_source": "emsi"}
        ]
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 2. Test Questions Endpoint
        questions_resp = await ac.post("/api/pathway/questions", json={
            "current_state_id": session_id,
            "jd_id": jd_id
        })
        assert questions_resp.status_code == 200
        questions_data = questions_resp.json()
        assert "questions" in questions_data
        
        # Should have baseline questions + 1 dynamic technical question
        q_ids = [q["id"] for q in questions_data["questions"]]
        assert "weekly_hours" in q_ids
        assert "learning_style" in q_ids
        assert "preferred_os" in q_ids # Dynamic because domain is technical and gap exists

        # 3. Test Generate Endpoint
        generate_resp = await ac.post("/api/pathway/generate", json={
            "current_state_id": session_id,
            "jd_id": jd_id,
            "preferences": {
                "weekly_hours": "4-6 hours",
                "learning_style": "Hands-on projects",
                "preferred_os": "Linux/Unix"
            }
        })
        assert generate_resp.status_code == 200
        pipeline_state = generate_resp.json()
        
        # Verify PipelineState structure
        assert "current" in pipeline_state
        assert "target" in pipeline_state
        assert "final_pathway" in pipeline_state
        assert "metrics" in pipeline_state
        
        # Verify specific logic: Python (0.9) should be skipped, SQL (0.1) should be assigned/prerequisite
        # Note: In our tiny mock catalog in pathing.py, SQL Basics is C-SQL-101
        pathway = pipeline_state["final_pathway"]
        course_ids = [c["course_id"] for c in pathway]
        assert any(c["course_id"] == "C-SQL-101" and c["node_state"] != "skipped" for c in pathway)

if __name__ == "__main__":
    pytest.main([__file__])
