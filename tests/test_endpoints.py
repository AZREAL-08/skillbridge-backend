import pytest
import io
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
        "created_at": "2026-03-21T00:00:00",
        "is_deleted": False
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
        assert "preferred_os" in q_ids 

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
        
        # Verify PipelineState structure (v2.0)
        assert "current" in pipeline_state
        assert "target" in pipeline_state
        assert "final_pathway" in pipeline_state
        assert "metrics" in pipeline_state
        assert "preference_questions" in pipeline_state
        
        # Verify Metrics (v2.0)
        metrics = pipeline_state["metrics"]
        assert "total_hours" in metrics
        assert "saved_hours" in metrics
        
        # Verify specific logic
        pathway = pipeline_state["final_pathway"]
        assert any(c["course_id"] == "C-SQL-101" and c["node_state"] != "skipped" for c in pathway)

@pytest.mark.anyio
async def test_jd_management():
    """Verify JD list, get, and soft-delete."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create a JD via confirm (direct mock in DB for speed)
        jd_id = "jd_mgmt_test"
        DB_JDS[jd_id] = {
            "jd_id": jd_id,
            "role_title": "Test Role",
            "company": "Test Co",
            "department": "IT",
            "domain": "technical",
            "raw_text": "text",
            "required_skills": [],
            "created_at": "now",
            "is_deleted": False
        }
        
        # 1. Test Get
        resp = await ac.get(f"/api/jd/{jd_id}")
        assert resp.status_code == 200
        assert resp.json()["jd_id"] == jd_id
        
        # 2. Test List
        resp = await ac.get("/api/jd/list")
        assert any(j["jd_id"] == jd_id for j in resp.json())
        
        # 3. Test Delete
        resp = await ac.delete(f"/api/jd/{jd_id}")
        assert resp.status_code == 200
        
        # 4. Verify hidden from List
        resp = await ac.get("/api/jd/list")
        assert not any(j["jd_id"] == jd_id for j in resp.json())
        
        # 5. Verify 404 from Get
        resp = await ac.get(f"/api/jd/{jd_id}")
        assert resp.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__])
