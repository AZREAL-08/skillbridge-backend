import time
import json
from app.extractor.extractor import extract_skills

# The 5 personas to prove Cross-Domain Scalability
TEST_RESUMES = {
    "Persona A (Senior Tech)": "Over 8 years of experience in JavaScript and Node.js backend architecture. Migrated legacy databases to PostgreSQL. Mentored junior developers and managed agile sprints.",
    
    "Persona B (Warehouse Ops)": "Warehouse shift supervisor with 6 years of experience managing supply chain logistics and forklift operations. Implemented safety protocols that reduced incidents by 20%. Proficient in Microsoft Excel.",
    
    "Persona C (Marketing)": "Creative marketing strategist with a decade of experience in SEO, content creation, and social media campaigns. Managed a $50k monthly ad spend on Google Ads. Basic knowledge of HTML for email templates.",
    
    "Persona D (Junior Analyst)": "Recent bootcamp graduate with hands-on projects in Pandas, SQL, and data visualization using Tableau. Familiar with basic statistical modeling and A/B testing. Looking for an entry-level analytics role.",
    
    "Persona E (Retail Manager)": "Dedicated retail manager focused on customer service excellence and conflict resolution. Led a team of 15 sales associates, consistently exceeding quarterly sales targets. Handled payroll and staff scheduling."
}

def run_integration_tests():
    print("=" * 70)
    print("🚀 INITIATING FULL PIPELINE INTEGRATION TEST")
    print("=" * 70)

    total_start_time = time.time()

    for persona_name, resume_text in TEST_RESUMES.items():
        print(f"\n[ RUNNING ] {persona_name}")
        print("-" * 50)
        
        try:
            step_start = time.time()
            
            # --- THE CORE CALL ---
            results = extract_skills(resume_text)
            
            latency = time.time() - step_start
            
            # Print the results cleanly
            print(f"Latency: {latency:.2f} seconds")
            if not results:
                print("  ⚠ No skills extracted. Check model thresholds.")
            else:
                for skill in results:
                    # using json.dumps to ensure dictionary prints on a single clean line
                    print(f"  → {json.dumps(skill)}")
                    
        except Exception as e:
            print(f"  [!] PIPELINE FATAL ERROR: {str(e)}")

    total_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"✅ ALL TESTS COMPLETED IN {total_time:.2f} SECONDS")
    print("=" * 70)

if __name__ == "__main__":
    run_integration_tests()