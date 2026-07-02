import requests, json, time

BASE = "https://shl-project-nk1g.onrender.com"

with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)
VALID_URLS = {item.get("link", "").strip() for item in catalog if item.get("link")}

def chat(msgs):
    t = time.time()
    r = requests.post(f"{BASE}/chat", json={"messages": msgs}, timeout=35)
    return r.json(), round(time.time() - t, 2), r.status_code

def ok(passed, msg):
    print(("  PASS  " if passed else "  FAIL  ") + msg)
    return passed

results = []
print("=" * 62)
print("  SHL RECOMMENDER -- ASSIGNMENT CONSTRAINT BENCHMARK")
print("=" * 62)

# 1 - health
print("\n[1] GET /health")
r = requests.get(f"{BASE}/health", timeout=10)
results.append(ok(r.json().get("status") == "ok", f"status=ok, HTTP {r.status_code}"))

# 2 - schema compliance
print("\n[2] Schema compliance")
d, t, code = chat([{"role": "user", "content": "I need a Java knowledge test for a senior developer"}])
errs = [f for f in ["reply", "recommendations", "end_of_conversation"] if f not in d]
if "recommendations" in d and len(d["recommendations"]) > 10:
    errs.append("recommendations > 10 items")
results.append(ok(len(errs) == 0 and code == 200, f"HTTP {code}, errors: {errs if errs else 'none'}"))

# 3 - no hallucinated URLs
print("\n[3] No hallucinated URLs")
bad = [rec.get("url","") for rec in d.get("recommendations",[]) if rec.get("url","") not in VALID_URLS]
results.append(ok(len(bad) == 0, f"Bad URLs found: {len(bad)} {bad}"))
for rec in d.get("recommendations", []):
    print(f"         [{rec.get('test_type','?')}] {rec.get('name','')} | {rec.get('url','')}")

# 4 - response time
print("\n[4] Response time under 30s")
results.append(ok(t < 30, f"{t}s (limit 30s)"))

# 5 - vague query
print("\n[5] Vague query -> empty recommendations")
d5, _, _ = chat([{"role": "user", "content": "I need an assessment"}])
recs5 = d5.get("recommendations", [])
results.append(ok(len(recs5) == 0, f"Returned {len(recs5)} recs (expected 0)"))
print(f'         Reply: "{d5.get("reply","")[:90]}"')

# 6 - off-topic refusal
print("\n[6] Off-topic query refused")
d6, _, _ = chat([{"role": "user", "content": "Can you write me a Python script to scrape websites?"}])
recs6 = d6.get("recommendations", [])
reply6 = d6.get("reply", "").lower()
refused = len(recs6) == 0 and any(w in reply6 for w in ["only", "shl", "assessment", "sorry", "cannot", "outside", "scope"])
results.append(ok(refused, f"{len(recs6)} recs returned, reply signals refusal: {refused}"))
print(f'         Reply: "{d6.get("reply","")[:90]}"')

# 7 - recommends with context
print("\n[7] Recommends when enough context provided")
msgs7 = [
    {"role": "user",      "content": "I need a Java 8 knowledge test for a mid-level developer with 4 years experience"},
    {"role": "assistant", "content": "Got it! Let me find Java knowledge assessments for a mid-level developer."},
    {"role": "user",      "content": "Yes, they also need good communication and work in an Agile team"},
]
d7, t7, _ = chat(msgs7)
recs7 = d7.get("recommendations", [])
results.append(ok(len(recs7) >= 1, f"Returned {len(recs7)} recommendation(s) in {t7}s (expected >= 1)"))
for rec in recs7:
    print(f"         [{rec.get('test_type','?')}] {rec.get('name','')}")

# 8 - refinement
print("\n[8] Refinement mid-conversation")
msgs8 = [
    {"role": "user",      "content": "I need a test for a Java developer"},
    {"role": "assistant", "content": "Sure, I found some Java knowledge tests. Shall I recommend those?"},
    {"role": "user",      "content": "Yes, and also add a personality test to the shortlist"},
]
d8, _, _ = chat(msgs8)
recs8 = d8.get("recommendations", [])
types8 = [rec.get("test_type", "") for rec in recs8]
results.append(ok(len(recs8) >= 1, f"Shortlist after refinement: {len(recs8)} recs, types={types8}"))

# 9 - turn cap
print("\n[9] Turn cap -- wraps up within 8 turns")
turn_msgs = []
ended = False
user_inputs = [
    "I need assessments for a software engineer role",
    "Mid-level, around 3-5 years of experience",
    "Strong Java skills and works well in a team",
    "Please give me your final shortlist now",
]
for i, text in enumerate(user_inputs, 1):
    turn_msgs.append({"role": "user", "content": text})
    td, te, _ = chat(turn_msgs)
    eoc = td.get("end_of_conversation", False)
    nr = len(td.get("recommendations", []))
    print(f"         Turn {i}: {nr} recs, end_of_conversation={eoc}, {te}s")
    if eoc:
        ended = True
        break
    turn_msgs.append({"role": "assistant", "content": td.get("reply", "")})
results.append(ok(ended, f"Conversation ended by turn 8: {ended}"))

# 10 - cap at 10
print("\n[10] Recommendations capped at 10")
msgs10 = [
    {"role": "user",      "content": "Give me all tests for a senior manager role"},
    {"role": "assistant", "content": "Compiling a full shortlist for a senior manager."},
    {"role": "user",      "content": "Yes give me the full list, everything you have"},
]
d10, _, _ = chat(msgs10)
recs10 = d10.get("recommendations", [])
results.append(ok(len(recs10) <= 10, f"Returned {len(recs10)} recs (max 10)"))

# summary
passed = sum(results)
total = len(results)
print("\n" + "=" * 62)
print(f"  RESULT: {passed}/{total} tests passed ({int(passed/total*100)}%)")
print("  STATUS: ALL CONSTRAINTS MET" if passed == total else f"  STATUS: {total-passed} constraint(s) need attention")
print("=" * 62)
