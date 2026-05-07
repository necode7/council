"""
Demo runner — exercises the full council.py pipeline (anonymization, peer review,
chairman, terminal print, PDF) without making any network calls. The OpenRouter
client is monkey-patched to return pre-written, persona-flavoured responses to
the example CA-vs-RL question.

Use this only to preview the output format. Real verdicts come from `python3 council.py`.
"""

import asyncio
import council

# ---- canned content keyed by the `label` argument that council passes ----

ADVISOR_TEXT = {
    "Contrarian": """Both. That's the answer you're avoiding. 'Split time' isn't a strategy — it's what someone says when they haven't accepted they're already underwater on time. CA finals + ACCA PM + ACCA AAA + an RL project with team deadlines, all in one 8-week window? The fatal flaw isn't picking A or B. It's that the math doesn't work for any of them.

What aren't you telling me? You said the RL team has 'firm commitments' — what's the actual cost of bailing? Reputation? A grade? You said CA finals are 'critical' — critical to pass, or critical to top? Those are very different problems with very different time budgets. And the ACCA exams: firm dates or yours to defer?

The question 'how do I split time?' assumes a budget that doesn't exist. Pick the goal with the least-reversible deadline, ruthlessly downgrade everything else, or accept up front that one of these four tracks will fail. Choose which one consciously, instead of finding out in week 6.""",

    "First Principles Thinker": """You're asking 'how do I divide time?' — but the real question is 'what am I optimizing for over the next 5 years?' Not the next 2 months. Time allocation is downstream of that, and you haven't answered it yet.

Strip the assumptions. Is CA finals genuinely critical to your career, or is it critical because you've already sunk three years into it? Is the RL project critical, or is it the shiny thing your peers are excited about? ACCA — does it map to a job you actually want, or did you start it as a hedge against not knowing? Until you can answer those three honestly, any time-budget you draw is theatre.

Here's the test that bypasses the deliberation. Imagine you're 30, looking back. Which of these three failures hurts most: 'I almost passed CA finals but split focus,' 'I dropped the RL team mid-project,' 'I deferred ACCA by six months'? Whichever stings hardest is your priority. Most people answer in thirty seconds and realize they already knew.""",

    "Expansionist": """The framing 'CA vs RL' is too small. The upside nobody's pricing is the combination. Quant finance, algo trading, risk analytics, fintech infrastructure, derivatives modelling — the people who win in those fields have both the credential AND the technical chops. Most CAs can't ship code. Most ML engineers can't read a balance sheet. You are eight weeks from being one of the rare hybrids.

The Mario Kart RL project is a portfolio piece in disguise. If it ships and you can articulate it cleanly — 'trained an RL agent while studying for CA finals' — that's interview gold. It's a Twitter thread. It's the difference between 'CA looking to break into finance' and 'CA who prototypes trading systems.' That single bullet on your CV opens doors that 'CA + ACCA' alone won't.

Don't optimize for survival of the next 8 weeks. Optimize for the version of yourself sitting in a hedge fund, fintech, or your own startup in 18 months. That version of you needs both. Not equally — but both.""",

    "Outsider": """I have no idea what CA finals are or what ACCA stands for, and I'm reading 'Mario Kart RL project' as a video-game machine-learning hobby project. From outside, here's what I see: one clearly-credentialed track (two formal qualifications with fixed exam dates) and one uncredentialed track (a research/coding project with peer commitments).

Insiders in your field probably weight the credentials heavily — because credentials shaped their own careers. But: what does a real job listing in your target field actually demand? If it's the qualification, the project is a luxury. If it's demonstrated ability, the qualification alone won't get you past the first round.

One thing that's invisible from inside but obvious from outside: 'firm team commitments' on a hobby project is a social obligation, not a career obligation. You can apologise and step back. You cannot apologise your way to a passing CA mark. Weigh those two failure modes correctly.""",

    "Executor": """Stop deliberating. Here's a schedule. Six days a week, CA + ACCA from 6am to 9pm with breaks. That's your job for the next 8 weeks. Sundays plus 90 minutes after dinner on Mon/Wed/Fri = your RL window. That's about 12 hours/week on the project. Tell your team this is the budget. If they need more, they take the deltas. Send that message tonight.

Tomorrow morning: open the CA syllabus, list the six highest-weight topics, sort by your current marks-per-hour ratio. Start with the worst. Same exercise for ACCA PM and AAA on Wednesday. By Sunday you should have a one-page study calendar with daily targets, not weekly ones — daily targets are the only ones humans hit.

The people who pass these exams aren't smarter than you. They sat down earlier. Sit down today, not Monday. Monday is when most people lose two weeks pretending they'll start 'soon.'""",
}

# Reviews are written as if the reviewer doesn't know which letter is which persona.
# (The chairman sees the deanonymized version anyway.)
REVIEW_TEXT = {
    "Contrarian": """Strongest: Response D. It refuses to accept the premise of the question and forces a values-level choice (which failure stings most in 5 years), which is the only way to break out of false-equivalence between four obligations. Everything downstream gets easier once that's settled.

Biggest blind spot: Response B. It assumes the user can will themselves into a 6am-to-9pm grind for 8 weeks. That schedule has a 70%+ failure rate at week 3. It's prescriptive without being realistic about energy, motivation, or the cost of falling behind on it.

What all five missed: nobody asked about the user's family/financial situation or whether the RL project has a paying outcome (collaborators getting hired? a paper?). 'Just drop it' or 'just do it' both ignore that hobby projects sometimes have stakes the user hasn't articulated.""",

    "First Principles Thinker": """Strongest: Response A. It names the actual problem — that the time budget mathematically doesn't exist — instead of pretending optimization is possible inside an impossible constraint. Most useful framing of the five.

Biggest blind spot: Response C. The 'be the rare hybrid' pitch is romantic but unfalsifiable; it doesn't tell the user what to actually do this week. It optimizes for an 18-month story without addressing the 8-week deadline that's the entire point of the question.

What all five missed: not one of them asked about sleep, baseline health, or whether the user is already burned out. A 22-year-old running CA + ACCA + a project at full intensity for 8 weeks isn't a time problem, it's a recovery problem. Without sleep, none of these plans work.""",

    "Expansionist": """Strongest: Response C — it reframes the question from 'how do I survive the next 8 weeks' to 'what's the version of me 18 months from now,' which is the only frame in which the answer is genuinely 'do both.' That's the one that opens up the largest option space.

Biggest blind spot: Response E. The Monday-morning checklist is concrete but assumes the answer ('CA + ACCA dominate, RL gets scraps'). It's executing on a strategy that hasn't been chosen yet. Wrong order of operations.

What all five missed: the option of pushing the RL project's deadline by 4 weeks. None of them seriously interrogated whether the 'firm commitments' are actually firm or just unexamined. A frank conversation with the team often unlocks the dilemma entirely.""",

    "Outsider": """Strongest: Response D. It cuts through framing by asking which failure the user would regret most in 5 years. That's a question anyone — insider or outsider — can answer honestly, and once answered the rest of the choice is mechanical.

Biggest blind spot: Response B. It dismisses the RL project as 'a hobby project' and 'social obligation,' but a peer-shipped ML project with collaborators is closer to early career capital than to a hobby. Underestimates what's at stake on that side of the ledger.

What all five missed: the user's own track record. None asked: 'How well are you currently doing in CA mocks? Are you on pass-track or borderline?' If the user is already passing CA mocks comfortably, the time math changes completely — they should ship the RL project. If borderline, they shouldn't. The advisors all assumed parity.""",

    "Executor": """Strongest: Response E. It's the only one that turns advice into a Tuesday-morning action. Time on calendar beats elegance of framing. The others are essays; this one is a plan.

Biggest blind spot: Response A. Saying 'one of these will fail, choose which' is honest but actionably useless — it doesn't tell the user how to decide which to drop. Diagnosis without prescription.

What all five missed: nobody mentioned that ACCA PM and AAA can usually be deferred one cycle (3-4 months) at the cost of a small fee. That single move halves the conflict. Multiple advisors talked about prioritization without naming the most obvious lever — moving a deadline.""",
}

CHAIRMAN_TEXT = """## Where the Council Agrees

Four of the five advisors converged on one point: **the time budget for all four tracks (CA + ACCA PM + ACCA AAA + RL project) at full intensity does not exist**. Whether they framed it as a fatal-flaw problem, a values problem, or an execution problem, none of them believed the user can hit all four targets without compromise. The Contrarian, First Principles Thinker, Outsider, and Executor all implicitly told the user to stop pretending otherwise.

There was also strong agreement that **CA finals carry the heaviest, least-reversible weight** in the 8-week window. Even the Expansionist — who pushed hardest for keeping the RL project alive — never suggested CA finals should be downgraded.

## Where the Council Clashes

The real disagreement is about **what the RL project is worth**. The Outsider and Executor frame it as a hobby project / social obligation that can be apologised away. The Expansionist frames it as career capital — the rare hybrid signal that gets you into quant finance roles a pure-CA candidate can't reach. The Contrarian and First Principles Thinker stay agnostic and push the user to decide based on which failure they'd regret most at 30.

Both sides are reasonable. The Outsider/Executor are right that a hobby ML project does not, on its own, beat a CA pass. The Expansionist is right that combined credentials are increasingly how juniors break into quant/fintech roles where pure-CA candidates plateau. **Which view is correct depends on the user's target career, not on the project itself** — a fact none of the five advisors named directly enough.

## Blind Spots the Council Caught

The peer review surfaced three things the advisors as a group missed:

- **Track record.** Nobody asked how the user is currently performing in CA mocks. If already pass-track, the math is generous and the RL project should ship. If borderline, the project must yield. The advisors assumed equivalence.
- **The deadline lever.** ACCA PM and AAA can typically be deferred one cycle for a modest fee. Moving even one of those exams out of the window roughly halves the conflict. Several advisors prescribed schedules without considering whether the schedule itself is fixed.
- **Recovery, not just time.** An 8-week sprint at full load isn't a time-allocation problem; it's a sleep and burnout problem. The Executor's "6am-to-9pm" schedule has a high failure rate by week three.

## The Recommendation

**CA finals are the priority. Defer ACCA PM/AAA by one cycle. Keep the RL project alive but capped at 8–10 hours per week, with the team's deadline pushed back by 3–4 weeks.**

This isn't a compromise — it's a sequence. CA finals are a fixed, high-stakes, hard-to-reverse credential window that you've already invested years into; protect it. ACCA exams are deferrable and the cost of moving them is low; move them. The RL project is your differentiator for finance + ML roles 12–24 months from now; do not abandon it, but renegotiate its scope today, not in week 5.

The version of you in 18 months interviewing at a quant fund or fintech needs both the CA and a shippable ML project to point at. The version of you in 8 weeks needs to pass CA. These are not in conflict if you cut ACCA out of the window and shrink the RL commitment to a sustainable weekly budget. They are in conflict only if you refuse to move any of the deadlines you treat as fixed.

## The One Thing to Do First

**Tonight, send one message to the RL team:** "I need to push our deadline by 4 weeks and cap my contribution at 8–10 hours per week through CA finals. Can we make that work?" Their answer determines everything else. Until you've sent it, every study plan you build is fiction.
"""


# ---- monkey-patch: replace the OpenRouter call with canned lookups ----

async def fake_call(client, api_key, model, system, user, temperature, max_tokens, label):
    if label in ADVISOR_TEXT:
        return ADVISOR_TEXT[label]
    if label.endswith("(review)"):
        name = label.replace(" (review)", "")
        return REVIEW_TEXT[name]
    if label == "Chairman":
        return CHAIRMAN_TEXT
    return f"[no canned text for label={label!r}]"


council.call_openrouter = fake_call

if __name__ == "__main__":
    asyncio.run(council.main())
