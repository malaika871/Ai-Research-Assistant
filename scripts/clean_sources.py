import csv, hashlib, os, shutil, time
from pathlib import Path
import fitz  # PyMuPDF

CORPUS = Path("uploads")
OUT = CORPUS / "sources.csv"
BASE = "https://www.hec.gov.pk/english/policies/Pages/"

COLS = ["doc_id", "title", "doc_type", "version", "effective_date", "status",
        "supersedes", "source_url", "local_path", "sha256", "pages",
        "pdf_type", "language", "issuing_body", "notes"]

# file, doc_id, title, doc_type, version, effective_date, status, supersedes, source_url, notes
DOCS = [
 ("UGE-Policy.pdf", "ugep_v1_1", "Undergraduate Education Policy", "policy", "1.1 (Fall 2023)", "", "current", "", BASE + "UGEP.aspx", ""),
 ("Graduate-Education-Policy.pdf", "gep_2023", "Graduate Education Policy", "policy", "2023", "", "current", "phd_policy_2021", BASE + "GEP.aspx", ""),
 ("HEC-PhD-Policy.pdf", "phd_policy_2021", "Policy on PhD Degree Programs", "policy", "2021", "2021-01-01", "superseded", "", "", "Replaced by GEP-2023"),
 ("DDJDP.pdf", "ddjdp_2026", "Policy on Dual, Double and Joint Degree Programmes", "policy", "2026", "", "current", "", "", ""),
 ("FRAMEWORK FOR RECOGNITION OF LIBERAL ARTS UNDERGRADUATE PROGRAMS.pdf", "liberal_arts_framework_2026", "Framework for Recognition of Liberal Arts Undergraduate Programs", "framework", "2026", "", "current", "", "", ""),
 ("Guidelines for Engagement of Visiting Faculty in Higher Education Institutions.pdf", "visiting_faculty_2026", "Guidelines for Engagement of Visiting Faculty in HEIs", "guideline", "2026", "", "current", "", BASE + "Visiting-Faculty.aspx", ""),
 ("Framework-PE.pdf", "prof_emeritus_2024", "Framework for Appointment of Professor Emeritus in HEIs", "framework", "2024", "", "current", "", "", ""),
 ("HEC-Foreign-Collaboratin-Policy.pdf", "tne_policy_2024", "Transnational Education Policy (TNE)", "policy", "2024", "", "current", "", "", ""),
 ("Refund-Policy.pdf", "refund_policy_2024", "HEC National Fee Refund Policy", "policy", "Revised 2024", "", "current", "", "", ""),
 ("HEC - POLICY FOR STUDENTS WITH DISABILITIES AT HEIs IN PAKISTAN.pdf", "disabilities_2021", "Policy for Students with Disabilities at HEIs in Pakistan", "policy", "2021 (Amended)", "", "current", "", "", ""),
 ("Policy on drugs and tobacco control at HEIs.pdf", "drugs_tobacco_2021", "Policy on Drug and Tobacco Abuse in Higher Education Institutions", "policy", "2021", "", "current", "", "", ""),
 ("Campus-Security-surveillance-policy.pdf", "campus_security", "Campus Security and Surveillance Policy", "policy", "", "", "current", "", "", "Date not stated in PDF"),
 ("Final Examination Policy Guidelines.pdf", "uniform_semester", "Policy Guidelines for Implementation of Uniform Semester System in HEIs of Pakistan", "guideline", "", "", "current", "", "", "Date not stated in PDF"),
 ("IPFP-Policy.pdf", "ipfp_phase2_2025", "Interim Placement of Fresh PhDs (IPFP) Policy, Phase II", "policy", "2025", "", "current", "", "", ""),
 ("ORICs_Policy.pdf", "orics_2021", "Policy on Office of Research, Innovation and Commercialization", "policy", "2021", "2021-01-01", "current", "", "", ""),
 ("ASIP-Policy.pdf", "asip_policy", "Access to Scientific Instrumentation Program Policy", "policy", "", "", "current", "", "", "Date not stated in PDF"),
 ("Financial Rules Research Grants_Policy.pdf", "financial_rules_grants_2021", "Financial Rules Governing Competitive Research Grants", "policy", "Revised 2021", "", "current", "", "", ""),
 ("Travel Grant for Pakistani Researchers Revised_Policy.pdf", "tgpr_2021", "Travel Grant for Pakistani Researchers (TGPR), Revised Policy", "policy", "Revised 2021", "2022-07-01", "current", "", "", ""),
 ("Library Support Program_Policy.pdf", "library_support_2019", "Library Support Program", "policy", "2019", "2019-12-18", "current", "", "", "Dated 18 December 2019"),
 ("Textbook Writing Support Program_Policy.pdf", "twsp_policy", "HEC Textbook Writing Support Program (TWSP), Revised Policy", "policy", "", "", "current", "", "", "Date not stated in PDF"),
 ("PERIDOT Researchers Mobility Grant Program_Policy.pdf", "peridot_mobility", "PERIDOT Researchers Mobility Grant Program", "policy", "", "", "current", "", "", "Date not stated in PDF"),
 ("Scientific Reveiw Panels_Policy.pdf", "scientific_review_panels", "Scientific Review Panels: Assessment Framework", "policy", "", "", "current", "", "", "Date not stated in PDF"),
 ("ODL-Policy-Circulated.pdf", "odl_policy", "Open and Distance Learning (ODL) Policy", "policy", "", "", "current", "", "", "TITLE AND VERSION UNVERIFIED: check the PDF"),
 ("887_HEC2_Criteria_of_university_institutions.pdf", "criteria_institutions", "HEC Criteria for Universities and Institutions", "criteria", "", "", "current", "", "", "TITLE AND VERSION UNVERIFIED: check the PDF"),
 ("Time bound SOPs for Plagiarism Cases.pdf", "plagiarism_sops", "Time-bound SOPs for Plagiarism Cases", "sop", "", "", "current", "", "", ""),
 ("Little Book of Plagiarism.pdf", "plagiarism_little_book", "The Little Book of Plagiarism", "guide", "", "", "current", "", "", "HEC adaptation of a Leeds Metropolitan University booklet"),
]

rows, missing, scanned = [], [], []
for (fname, doc_id, title, doc_type, version, eff, status, supersedes, url, notes) in DOCS:
    p = CORPUS / fname
    if not p.is_file():
        missing.append(fname)
        continue
    doc = fitz.open(p)
    pages = len(doc)
    text_pages = sum(1 for pg in doc if len(pg.get_text().strip()) > 50)
    pdf_type = "text" if pages and text_pages / pages > 0.8 else "scanned"
    if pdf_type == "scanned":
        scanned.append(fname)
    rows.append({
        "doc_id": doc_id, "title": title, "doc_type": doc_type, "version": version,
        "effective_date": eff, "status": status, "supersedes": supersedes,
        "source_url": url, "local_path": f"uploads/{fname}",
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "pages": pages, "pdf_type": pdf_type, "language": "en",
        "issuing_body": "HEC", "notes": notes,
    })

if OUT.exists():
    os.makedirs("backups", exist_ok=True)
    shutil.copy(OUT, f"backups/sources_{int(time.time())}.csv")   # never overwritten

tmp = str(OUT) + ".tmp"
with open(tmp, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(rows)
os.replace(tmp, OUT)

print(f"{len(rows)} rows written to {OUT}")
print("Missing files (fix the name in DOCS):", missing)
print("Scanned PDFs (need OCR):", scanned)