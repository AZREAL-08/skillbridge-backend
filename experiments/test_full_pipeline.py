from app.pathing.pathing import run_pipeline
import json

def test_pipeline():
    resume_text = "Experienced developer with Python and Git."
    jd_text = "Looking for a Python developer with SQL skills."
    
    print("Running pipeline...")
    result = run_pipeline(resume_text, jd_text)
    
    print("\n--- Pipeline State ---")
    print(json.dumps(result, indent=2))
    
    # Assertions
    assert "current" in result
    assert "target" in result
    assert "skill_gap" in result
    assert "final_pathway" in result
    assert "reasoning_trace" in result
    assert "metrics" in result
    
    print("\nTest passed!")

if __name__ == "__main__":
    test_pipeline()
