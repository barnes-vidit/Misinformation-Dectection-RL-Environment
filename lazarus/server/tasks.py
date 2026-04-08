"""
Task definitions, dataset, and grading logic for the Misinformation Detection Environment.

Dataset: 18 fictional-but-realistic claims across 3 difficulty tiers.
All claims are invented for the environment — they do NOT reference real events.

Grader is fully deterministic: same inputs always produce the same reward float in [0.0, 1.0].
"""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import FactCheckAction

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

EASY_CLAIMS: list[dict] = [
    {
        "id": "easy_001",
        "claim": "The Eiffel Tower is located in Berlin, Germany.",
        "article_snippet": (
            "The Eiffel Tower, built between 1887 and 1889 as the centrepiece of the 1889 "
            "World's Fair, stands on the Champ de Mars in Paris, France. The iron lattice "
            "tower was designed by engineer Gustave Eiffel and stands approximately 330 metres "
            "tall including its broadcast antenna. It attracted fierce criticism from prominent "
            "French artists and intellectuals during its construction, who dubbed it an 'eyesore'. "
            "Despite early opposition, the tower became the most-visited paid monument in the world, "
            "drawing nearly seven million tourists annually. The structure was originally intended "
            "as a temporary installation and was supposed to be dismantled in 1909, but its utility "
            "as a radio transmission tower saved it. Today it serves as a global icon of France "
            "and is recognised immediately by most people worldwide. Berlin, the capital of Germany, "
            "does not contain any part of the Eiffel Tower and never has."
        ),
        "source_metadata": {"outlet": "Fictitious Times", "date": "2024-03-12", "topic": "geography"},
        "ground_truth_verdict": "FALSE",
        "key_evidence_phrases": ["Paris, France", "Champ de Mars"],
        "explanation": (
            "The claim is FALSE. The article clearly states the tower stands on the Champ de Mars "
            "in Paris, France — not in Berlin, Germany."
        ),
    },
    {
        "id": "easy_002",
        "claim": "Water boils at 100 °C at standard atmospheric pressure.",
        "article_snippet": (
            "The boiling point of water depends on the surrounding atmospheric pressure. Under "
            "standard atmospheric pressure — defined as 101.325 kilopascals (kPa) at sea level — "
            "pure water boils at exactly 100 degrees Celsius (212 degrees Fahrenheit). This "
            "fundamental physical constant has been confirmed countless times in laboratory "
            "settings and forms the basis for the Celsius temperature scale itself, where "
            "100 °C was originally defined as water's boiling point. At higher altitudes, "
            "atmospheric pressure decreases, causing water to boil at lower temperatures — "
            "for example, at 3,000 metres above sea level, water boils at roughly 90 °C. "
            "Dissolved salts raise the boiling point slightly through boiling-point elevation, "
            "but this effect is negligible for small quantities. Overall, the statement that "
            "water boils at 100 °C at standard atmospheric pressure is a well-established "
            "scientific fact taught in every introductory chemistry course."
        ),
        "source_metadata": {"outlet": "ScienceDaily Digest", "date": "2024-01-09", "topic": "science"},
        "ground_truth_verdict": "TRUE",
        "key_evidence_phrases": ["100 degrees Celsius", "standard atmospheric pressure"],
        "explanation": (
            "The claim is TRUE. The article confirms water boils at 100 °C under standard "
            "atmospheric pressure (101.325 kPa)."
        ),
    },
    {
        "id": "easy_003",
        "claim": "Mount Olympia is the tallest mountain on Earth, standing at 12,000 metres.",
        "article_snippet": (
            "Mount Everest, located on the border of Nepal and Tibet, is the tallest mountain "
            "on Earth above sea level, reaching a height of 8,848.86 metres (29,031.7 feet) "
            "as measured by a 2020 survey conducted jointly by China and Nepal. The mountain "
            "is part of the Himalayan range and was first successfully summited by Sir Edmund "
            "Hillary and Tenzing Norgay on 29 May 1953. No mountain on Earth exceeds 9,000 "
            "metres above sea level. Mount Olympia does not appear in any credible geographical "
            "record as a significant peak, and the figure of 12,000 metres would exceed the "
            "theoretical maximum height for a mountain on this planet given tectonic and "
            "gravitational constraints. The claim therefore contains two separate factual errors: "
            "the mountain name and the stated elevation."
        ),
        "source_metadata": {"outlet": "Geography Now Weekly", "date": "2024-06-22", "topic": "geography"},
        "ground_truth_verdict": "FALSE",
        "key_evidence_phrases": ["Mount Everest", "8,848.86 metres"],
        "explanation": (
            "The claim is FALSE on two counts: the tallest mountain is Mount Everest (not Mount "
            "Olympia), and it stands at 8,848.86 m — not 12,000 m."
        ),
    },
    {
        "id": "easy_004",
        "claim": "Photosynthesis produces oxygen as a by-product.",
        "article_snippet": (
            "Photosynthesis is the biological process by which green plants, algae, and some "
            "bacteria convert light energy — typically from the sun — into chemical energy stored "
            "as glucose. The overall chemical equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ "
            "+ 6O₂. In this reaction, carbon dioxide and water are used to produce glucose and "
            "oxygen. The oxygen released is a direct by-product of the splitting of water molecules "
            "during the light-dependent reactions stage. This oxygen is expelled through tiny pores "
            "called stomata on the leaf surface. The process is fundamental to life on Earth: it "
            "is the primary source of atmospheric oxygen and forms the base of most food chains. "
            "Without photosynthesis, atmospheric oxygen levels would plummet and most aerobic life "
            "would perish."
        ),
        "source_metadata": {"outlet": "Biology Basics Journal", "date": "2024-02-14", "topic": "biology"},
        "ground_truth_verdict": "TRUE",
        "key_evidence_phrases": ["oxygen released is a direct by-product", "splitting of water molecules"],
        "explanation": (
            "The claim is TRUE. The article explicitly states oxygen is released as a by-product "
            "of photosynthesis during the splitting of water molecules."
        ),
    },
    {
        "id": "easy_005",
        "claim": "The human body has 206 bones in adulthood.",
        "article_snippet": (
            "The adult human skeleton consists of exactly 206 bones, a figure that is universally "
            "agreed upon by anatomists worldwide. Interestingly, newborns have approximately 270 "
            "to 300 bones at birth; many of these fuse together during childhood and adolescence. "
            "By the time a person reaches adulthood — typically around age 25 — the count "
            "stabilises at 206. The skeleton is divided into two main sections: the axial skeleton "
            "(80 bones, including the skull, vertebral column, and rib cage) and the appendicular "
            "skeleton (126 bones, covering the limbs and girdles). Bones serve critical functions "
            "including structural support, protection of internal organs, mineral storage, and the "
            "production of blood cells in bone marrow. Medical students memorise this figure early "
            "in training as a foundational anatomical fact."
        ),
        "source_metadata": {"outlet": "Medical Facts Quarterly", "date": "2024-04-01", "topic": "anatomy"},
        "ground_truth_verdict": "TRUE",
        "key_evidence_phrases": ["exactly 206 bones", "adult human skeleton"],
        "explanation": (
            "The claim is TRUE. The article confirms the adult human skeleton has exactly 206 bones."
        ),
    },
    {
        "id": "easy_006",
        "claim": "The speed of light in a vacuum is approximately 3 million kilometres per second.",
        "article_snippet": (
            "The speed of light in a vacuum, denoted by the symbol c, is one of the most "
            "fundamental constants in physics. Its exact value is 299,792,458 metres per second, "
            "which is approximately 300,000 kilometres per second — or about 186,000 miles per "
            "second. The claim that it equals 3 million kilometres per second is off by a factor "
            "of ten: the correct figure is roughly 300,000 km/s, not 3,000,000 km/s. This constant "
            "forms the cornerstone of Einstein's theory of special relativity and sets the ultimate "
            "speed limit for the transfer of information in the universe. Light from the Sun takes "
            "approximately 8 minutes and 20 seconds to reach Earth, travelling across roughly "
            "150 million kilometres. The precise value of c has been fixed by international "
            "convention since 1983 and is used to define the metre."
        ),
        "source_metadata": {"outlet": "Physics Today", "date": "2024-05-18", "topic": "physics"},
        "ground_truth_verdict": "FALSE",
        "key_evidence_phrases": ["299,792,458 metres per second", "off by a factor of ten"],
        "explanation": (
            "The claim is FALSE. The speed of light is ~300,000 km/s, not 3 million km/s — "
            "the article notes the claim is off by a factor of ten."
        ),
    },
]

MEDIUM_CLAIMS: list[dict] = [
    {
        "id": "medium_001",
        "claim": "Greenland's GDP grew by 40% last year, proving its economic miracle.",
        "article_snippet": (
            "A recent economic report shows Greenland recorded a 40% increase in GDP last year, "
            "driven almost entirely by a single large-scale offshore oil exploration contract "
            "worth USD 2.8 billion signed with a foreign consortium. Greenland's total GDP before "
            "this contract was approximately USD 3.1 billion. Economists caution that the figure "
            "is highly misleading: the contract revenue is a one-time inflow and most profits will "
            "repatriate to foreign shareholders. Unemployment in Greenland actually rose slightly "
            "from 8.1% to 8.9% over the same period, and household income remained flat when "
            "adjusted for inflation. The government's own advisors warned that presenting this as "
            "evidence of an 'economic miracle' ignores structural dependencies on Danish subsidies "
            "and the non-renewable nature of the windfall. Analysts from the Nordic Economic "
            "Institute described the growth figure as a 'statistical artefact' rather than "
            "a sign of sustainable development."
        ),
        "source_metadata": {"outlet": "Nordic Finance Review", "date": "2024-07-30", "topic": "economics"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["one-time inflow", "statistical artefact"],
        "explanation": (
            "The claim is MISLEADING. The 40% GDP figure is technically accurate but reflects a "
            "one-time foreign contract, not sustained economic growth. Unemployment rose and "
            "household income stagnated."
        ),
    },
    {
        "id": "medium_002",
        "claim": "Dr Priya Kalan said vaccines cause childhood obesity.",
        "article_snippet": (
            "At a nutritional health conference in Helsinki, Dr Priya Kalan, a paediatric "
            "dietitian at the Fictitious Institute of Child Health, presented a paper on the "
            "rising rates of childhood obesity in Nordic countries. When asked about potential "
            "contributing factors during the Q&A session, Dr Kalan listed sedentary screen time, "
            "ultra-processed food marketing, reduced physical education in schools, and disrupted "
            "sleep patterns. A journalist in attendance later published a headline claiming Dr Kalan "
            "had stated that 'vaccines cause childhood obesity.' Dr Kalan promptly issued a "
            "correction stating she never mentioned vaccines during her talk or in the Q&A and "
            "that the claim is scientifically unfounded. The article misattributes a statement "
            "she never made. No peer-reviewed study supports a causal link between childhood "
            "vaccination schedules and obesity rates."
        ),
        "source_metadata": {"outlet": "Health Matters Blog", "date": "2024-09-05", "topic": "health"},
        "ground_truth_verdict": "FALSE",
        "key_evidence_phrases": ["never mentioned vaccines", "misattributes a statement she never made"],
        "explanation": (
            "The claim is FALSE. Dr Kalan never said vaccines cause obesity — the article "
            "documents that the quote was fabricated by a journalist."
        ),
    },
    {
        "id": "medium_003",
        "claim": "Drinking eight glasses of water per day is a scientifically proven health requirement.",
        "article_snippet": (
            "The popular advice to drink 'eight glasses of water a day' has been repeated in "
            "health campaigns for decades, but its scientific basis is questionable. A 2002 review "
            "by Dr Heinz Valtin published in the American Journal of Physiology found no scientific "
            "evidence supporting the '8×8' rule (eight 8-ounce glasses). Actual hydration needs "
            "vary enormously by body weight, activity level, climate, and dietary water intake from "
            "food and other beverages. The figure of eight glasses appears to have originated from "
            "a 1945 US Food and Nutrition Board recommendation that was widely misread: the original "
            "text stated that most of the water requirement is already supplied by food. Doctors "
            "generally advise drinking when thirsty and monitoring urine colour as a hydration "
            "indicator. The claim is therefore partly true (hydration is important) but the specific "
            "eight-glasses figure is not scientifically mandated."
        ),
        "source_metadata": {"outlet": "MedCheck Weekly", "date": "2024-03-20", "topic": "health"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["no scientific evidence supporting the '8×8' rule", "most of the water requirement is already supplied by food"],
        "explanation": (
            "The claim is MISLEADING. Hydration is real but the '8 glasses' figure is not "
            "scientifically mandated and ignores individual variation and dietary water intake."
        ),
    },
    {
        "id": "medium_004",
        "claim": "The Fictonia city council voted unanimously to ban all fossil fuel vehicles by 2030.",
        "article_snippet": (
            "After a contentious three-hour debate, the Fictonia city council passed a resolution "
            "on Tuesday calling for a transition away from fossil fuel vehicles by 2030. The motion "
            "passed 9 votes to 3, with the dissenting councillors arguing the timeline was "
            "unworkable given the city's limited public transport infrastructure. The resolution "
            "is non-binding and requires state government approval before any enforcement mechanism "
            "can be introduced. Legal analysts noted that the city lacks constitutional authority "
            "to unilaterally impose a vehicle ban, and the state transport minister indicated she "
            "would review the proposal 'in due course.' Supporters celebrated the symbolic gesture "
            "while environmental groups cautioned that without state backing, the target remains "
            "aspirational at best."
        ),
        "source_metadata": {"outlet": "Fictonia Gazette", "date": "2024-10-15", "topic": "politics"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["9 votes to 3", "non-binding"],
        "explanation": (
            "The claim is MISLEADING. The vote was not unanimous (9-3) and the resolution is "
            "non-binding — so the 'ban' has no legal force yet."
        ),
    },
    {
        "id": "medium_005",
        "claim": "Electric cars produce zero emissions.",
        "article_snippet": (
            "Electric vehicles (EVs) produce zero direct tailpipe emissions while driving, which "
            "represents a meaningful improvement over internal combustion engine vehicles in urban "
            "air quality. However, the full lifecycle emissions picture is more complex. The "
            "electricity used to charge EVs may be generated from coal, natural gas, or other "
            "fossil fuels depending on the regional energy grid. Manufacturing EV batteries — "
            "particularly mining lithium, cobalt, and nickel — generates significant carbon "
            "emissions. A 2023 lifecycle analysis by the Fictitious Institute of Sustainable "
            "Transport found that EVs charged on coal-heavy grids produce only 20% fewer lifecycle "
            "emissions than comparable petrol vehicles. On renewable-powered grids the saving "
            "rises to around 70%. The claim of 'zero emissions' therefore applies only to direct "
            "tailpipe output, not to manufacturing or electricity generation."
        ),
        "source_metadata": {"outlet": "Clean Tech Report", "date": "2024-08-27", "topic": "environment"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["zero direct tailpipe emissions", "manufacturing EV batteries"],
        "explanation": (
            "The claim is MISLEADING. EVs have zero tailpipe emissions but lifecycle emissions "
            "from battery manufacturing and grid electricity are significant."
        ),
    },
    {
        "id": "medium_006",
        "claim": "The Valoria stock exchange rose 15% in Q2, signalling strong investor confidence.",
        "article_snippet": (
            "The Valoria Securities Exchange (VSE) posted a 15% gain over the second quarter, "
            "driven primarily by a single sector: speculative mining stocks linked to a newly "
            "discovered rare-earth deposit in northern Valoria. Technology, banking, and "
            "consumer goods stocks — which represent 72% of the exchange by market capitalisation "
            "— were broadly flat or negative over the same period. The VSE has a thin float with "
            "fewer than 120 listed companies, making it susceptible to outsized moves from a "
            "small number of stocks. Foreign portfolio investors actually reduced their holdings "
            "in Valoria by 8% during Q2, citing political uncertainty ahead of the upcoming "
            "election. The headline gain masks a selective rally that does not reflect economy-wide "
            "investor sentiment."
        ),
        "source_metadata": {"outlet": "Valoria Financial Post", "date": "2024-07-05", "topic": "finance"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["driven primarily by a single sector", "Foreign portfolio investors actually reduced their holdings"],
        "explanation": (
            "The claim is MISLEADING. The 15% rise is real but driven by one speculative sector; "
            "foreign investors withdrew and broad market performance was flat-to-negative."
        ),
    },
]

HARD_CLAIMS: list[dict] = [
    {
        "id": "hard_001",
        "claim": "Professor Alana Merced's study proves that social media use causes depression in teenagers.",
        "article_snippet": (
            "A longitudinal study led by Professor Alana Merced at the Fictitious University of "
            "Westbrook tracked 1,200 teenagers aged 13-17 over three years, measuring social media "
            "usage and self-reported depression scores. The study found a statistically significant "
            "positive correlation (r = 0.34, p < 0.01) between hours of social media use per day "
            "and depression scores. However, the study's own methodology section notes multiple "
            "limitations: depression scores were self-reported using a non-validated questionnaire; "
            "the sample was drawn exclusively from high-income urban schools, limiting "
            "generalisability; and critically, the study design was cross-sectional-at-follow-up "
            "rather than fully longitudinal, preventing causal inference. Professor Merced stated "
            "in the conclusion: 'We observe an association, not causation. Reverse causality — "
            "depressed individuals seeking social connection online — cannot be excluded.' The "
            "study has not yet been independently replicated."
        ),
        "source_metadata": {"outlet": "Westbrook Journal of Psychology", "date": "2024-11-03", "topic": "mental health"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["association, not causation", "preventing causal inference"],
        "explanation": (
            "The claim is MISLEADING. The study found correlation, not proven causation. "
            "Professor Merced herself explicitly stated the study cannot establish causality, "
            "and reverse causality remains possible."
        ),
    },
    {
        "id": "hard_002",
        "claim": "The Rekova Dam project will displace 200,000 people.",
        "article_snippet": (
            "The proposed Rekova Hydroelectric Dam, if approved, would flood an estimated "
            "valley area of 340 square kilometres in the Rekova River basin. Government "
            "environmental impact assessments from 2022 estimated that between 80,000 and "
            "130,000 people currently reside in the affected zone, although civil society "
            "groups contest this range, arguing that seasonal and undocumented populations "
            "could push the true figure closer to 180,000. No official body has published "
            "an estimate of 200,000. The project is currently suspended pending judicial "
            "review. Final displacement figures will depend on the exact flood zone "
            "boundaries, which are still under negotiation, and on whether resettlement "
            "exclusion zones are enforced. The dam has not yet received final approval "
            "from the National Infrastructure Authority."
        ),
        "source_metadata": {"outlet": "Rekova Tribune", "date": "2024-02-28", "topic": "infrastructure"},
        "ground_truth_verdict": "UNVERIFIABLE",
        "key_evidence_phrases": ["between 80,000 and 130,000", "No official body has published an estimate of 200,000"],
        "explanation": (
            "The claim is UNVERIFIABLE. The article gives contested ranges (80k-180k) but no "
            "source supports the 200,000 figure, and final figures depend on unapproved boundaries."
        ),
    },
    {
        "id": "hard_003",
        "claim": "Novacor Industries reduced its carbon emissions by 60% between 2018 and 2023.",
        "article_snippet": (
            "Novacor Industries published its annual sustainability report last week, claiming "
            "a 60% reduction in carbon emissions between 2018 and 2023. The company's own "
            "footnotes reveal that this figure uses a market-based accounting method and "
            "includes the purchase of carbon offset credits rather than actual operational "
            "reductions. On a location-based accounting method — which measures actual "
            "physical emissions from operations — the reduction was 18%. Novacor also "
            "changed its reporting baseline year in 2021 from a high-emission year (2015) "
            "to 2018, when emissions were already 22% lower. Independent auditors from "
            "VeraCert LLP noted that scope 3 supply chain emissions, which account for "
            "an estimated 65% of Novacor's total carbon footprint, are excluded from all "
            "reported figures. The 60% figure therefore cannot be verified against physical "
            "emission reductions without further disclosure."
        ),
        "source_metadata": {"outlet": "Corporate Climate Watch", "date": "2024-12-01", "topic": "environment"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["market-based accounting method", "location-based accounting method — which measures actual physical emissions — the reduction was 18%"],
        "explanation": (
            "The claim is MISLEADING. The 60% figure relies on offset credits and a changed "
            "baseline year; actual physical emission reduction was only 18%, and scope 3 "
            "emissions (65% of footprint) are excluded entirely."
        ),
    },
    {
        "id": "hard_004",
        "claim": "The ancient city of Keldara was founded in 3000 BC.",
        "article_snippet": (
            "Archaeological excavations at the Keldara site, conducted between 2019 and 2023 "
            "by a joint team from three universities, have unearthed pottery shards, foundation "
            "stones, and charcoal samples. Radiocarbon dating of the charcoal samples yielded "
            "dates ranging from approximately 3,400 BC to 2,600 BC at two-sigma confidence "
            "intervals, suggesting the site was occupied during this broad period. Researchers "
            "note that the oldest strata excavated so far cover only 12% of the estimated "
            "site area, and deeper layers may contain earlier remains. The lead archaeologist "
            "stated: 'We cannot yet determine a precise founding date; 3000 BC falls within "
            "our current date range but is neither confirmed nor ruled out.' No written founding "
            "records exist for Keldara. The claim of a 3000 BC founding date cannot be "
            "confirmed or denied with current evidence."
        ),
        "source_metadata": {"outlet": "Archaeology Today", "date": "2024-04-17", "topic": "history"},
        "ground_truth_verdict": "UNVERIFIABLE",
        "key_evidence_phrases": ["3000 BC falls within our current date range but is neither confirmed nor ruled out", "cannot yet determine a precise founding date"],
        "explanation": (
            "The claim is UNVERIFIABLE. 3000 BC falls within the radiocarbon date range but "
            "cannot be confirmed as the founding date; only 12% of the site has been excavated."
        ),
    },
    {
        "id": "hard_005",
        "claim": "The Talmira River flood in 2023 killed more than 1,000 people.",
        "article_snippet": (
            "The Talmira River flood event of June 2023 caused widespread devastation across "
            "three provinces. Government agencies reported 847 confirmed fatalities within "
            "the first month. A UN humanitarian report issued three months later raised the "
            "estimate to 912 deaths after additional search operations concluded. Local NGOs "
            "stated that unofficial tallies from community leaders suggest the death toll "
            "could exceed 1,000 when accounting for remote villages where documentation is "
            "poor and some bodies may never be recovered. However, no official body has "
            "confirmed a figure above 912 as of the date of this article. The discrepancy "
            "between official and community estimates reflects the challenges of data "
            "collection in disaster-affected regions with limited state capacity."
        ),
        "source_metadata": {"outlet": "Disaster Watch Network", "date": "2023-10-09", "topic": "disaster"},
        "ground_truth_verdict": "UNVERIFIABLE",
        "key_evidence_phrases": ["no official body has confirmed a figure above 912", "could exceed 1,000 when accounting for remote villages"],
        "explanation": (
            "The claim is UNVERIFIABLE. Official confirmed count is 912; the >1,000 figure "
            "comes from unverifiable community estimates in areas with poor documentation."
        ),
    },
    {
        "id": "hard_006",
        "claim": "Quantum computers have already made all current encryption obsolete.",
        "article_snippet": (
            "Quantum computing research is progressing rapidly, with some experimental systems "
            "now demonstrating capabilities that would have seemed impossible a decade ago. "
            "A paper published in Fictitious Science Letters in late 2023 described a "
            "quantum processor that factored a 1,024-bit RSA key — a significant milestone. "
            "However, experts caution that the system required near-absolute-zero temperatures, "
            "13,000 physical qubits to maintain error correction for just 100 logical qubits, "
            "and hours of computation to factor a single key. Commercial RSA deployments "
            "typically use 2,048-bit or 4,096-bit keys, which would require quantum machines "
            "orders of magnitude more powerful than anything currently in existence. The "
            "National Institute of Fictitious Standards has already begun transitioning to "
            "post-quantum cryptographic standards precisely to prepare for future threats. "
            "Cryptographers largely agree that current encryption is not yet broken in practice, "
            "but the threat is real on a 10-15 year horizon."
        ),
        "source_metadata": {"outlet": "Tech Frontier Digest", "date": "2024-01-22", "topic": "cybersecurity"},
        "ground_truth_verdict": "FALSE",
        "key_evidence_phrases": ["current encryption is not yet broken in practice", "orders of magnitude more powerful than anything currently in existence"],
        "explanation": (
            "The claim is FALSE. Quantum computers have demonstrated progress but cannot yet "
            "break commercial-grade 2048/4096-bit encryption in practice; experts say the "
            "threat is 10-15 years away."
        ),
    },
    {
        "id": "hard_007",
        "claim": "The Veldara pharmaceutical trial showed Zenivax cured 95% of patients with Stage III Lumaris disease.",
        "article_snippet": (
            "Results from the Phase II Veldara trial of Zenivax, published in the Fictitious "
            "Medical Lancet, reported that 95 of 100 patients with Stage III Lumaris disease "
            "showed 'complete response' at the 12-week mark. However, the trial has multiple "
            "significant limitations noted by peer reviewers: the 100-patient sample is small "
            "for Phase II; 'complete response' was defined as a 50% reduction in biomarker "
            "levels, not symptom resolution or cure; there was no placebo control group; "
            "and the follow-up period was only 12 weeks — insufficient to assess long-term "
            "outcomes or relapse rates. An independent review panel noted that three patients "
            "who did not respond were excluded from the final analysis due to 'protocol "
            "deviations,' which critics argue inflated the response rate. Phase III trials "
            "are not yet planned. The word 'cured' does not appear anywhere in the published "
            "paper."
        ),
        "source_metadata": {"outlet": "Fictitious Medical Lancet Review", "date": "2024-06-10", "topic": "medicine"},
        "ground_truth_verdict": "MISLEADING",
        "key_evidence_phrases": ["'complete response' was defined as a 50% reduction in biomarker levels", "The word 'cured' does not appear anywhere in the published paper"],
        "explanation": (
            "The claim is MISLEADING. The 95% figure is from a small, uncontrolled Phase II "
            "trial measuring biomarker reduction, not cure. The word 'cured' does not appear "
            "in the actual paper."
        ),
    },
]

CLAIMS_BY_TASK: dict[str, list[dict]] = {
    "easy": EASY_CLAIMS,
    "medium": MEDIUM_CLAIMS,
    "hard": HARD_CLAIMS,
}

# ---------------------------------------------------------------------------
# GRADER
# ---------------------------------------------------------------------------

VALID_VERDICTS = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"}


def grade_action(action: "FactCheckAction", claim: dict, task_id: str) -> float:
    """
    Deterministic grader for a FactCheckAction against a ground-truth claim dict.

    Returns a float in [0.0, 1.0].  Same inputs always produce the same output.

    Grading rules by task:
    -----------------------
    EASY
      +1.0  correct verdict
      +0.3  reasoning includes a key_evidence_phrase (capped so total ≤ 1.0)
      -0.2  overconfidence penalty: confidence > 0.9 AND verdict is wrong (capped ≥ 0.0)

    MEDIUM
      +1.0  exact verdict match
      +0.3  exact verdict + reasoning mentions a key_evidence_phrase
      +0.6  wrong verdict but reasoning explicitly acknowledges complexity
             (e.g. contains "complex", "nuanced", "partial", "context", "misleading")
      +0.3  wrong verdict, evidence cited, no complexity acknowledgement

    HARD
      +1.0  exact verdict match
      +0.3  exact verdict + evidence_cited contains a key_evidence_phrase (bonus)
      -0.3  overconfidence penalty: verdict is UNVERIFIABLE AND confidence > 0.8
      +0.3  wrong verdict but reasoned engagement (evidence cited + reasoning ≥ 30 chars)
      evidence_cited scoring: any key_evidence_phrase substring found in any cited string
    """
    verdict = (action.verdict or "").strip().upper()
    ground_truth = claim["ground_truth_verdict"].upper()
    reasoning_lower = (action.reasoning or "").lower()
    evidence_list = [e.lower() for e in (action.evidence_cited or [])]
    key_phrases = [kp.lower() for kp in claim.get("key_evidence_phrases", [])]

    correct = verdict == ground_truth

    # --- Helper functions (all pure / deterministic) ---

    def _evidence_cited_hit() -> bool:
        """True if any key_evidence_phrase appears as substring in any evidence_cited entry."""
        for kp in key_phrases:
            for cited in evidence_list:
                if kp in cited:
                    return True
        return False

    def _reasoning_evidence_hit() -> bool:
        """True if any key_evidence_phrase appears as substring in reasoning."""
        for kp in key_phrases:
            if kp in reasoning_lower:
                return True
        return False

    def _reasoning_acknowledges_complexity() -> bool:
        """True if reasoning contains words that show nuanced thinking."""
        markers = [
            "misleading", "nuanced", "complex", "partial", "context",
            "however", "but", "although", "cherry", "cherry-picked",
            "not entirely", "not fully", "not completely", "oversimplified",
        ]
        for m in markers:
            if m in reasoning_lower:
                return True
        return False

    # --- Per-task grading ---

    if task_id == "easy":
        if correct:
            score = 1.0
            if _reasoning_evidence_hit():
                score = min(1.0, score + 0.3)
            return score
        else:
            penalty = 0.2 if action.confidence > 0.9 else 0.0
            return max(0.0, 0.0 - penalty)

    elif task_id == "medium":
        if correct:
            score = 1.0
            if _reasoning_evidence_hit():
                score = min(1.0, score + 0.3)  # already at 1.0, no-op but explicit
            return score
        else:
            if _reasoning_acknowledges_complexity() and (_evidence_cited_hit() or _reasoning_evidence_hit()):
                return 0.6
            elif _reasoning_acknowledges_complexity():
                return 0.3
            elif _evidence_cited_hit() or _reasoning_evidence_hit():
                return 0.3
            else:
                return 0.0

    elif task_id == "hard":
        is_unverifiable_claim = ground_truth == "UNVERIFIABLE"
        overconfidence_penalty = (
            0.3 if is_unverifiable_claim and action.confidence > 0.8 else 0.0
        )

        if correct:
            score = 1.0 - overconfidence_penalty
            if _evidence_cited_hit():
                score = min(1.0, score + 0.3)
            return max(0.0, score)
        else:
            # partial credit for genuine reasoning engagement
            has_cited = _evidence_cited_hit() or _reasoning_evidence_hit()
            has_reasoning = len(action.reasoning or "") >= 30
            if has_cited and has_reasoning:
                return max(0.0, 0.3 - overconfidence_penalty)
            return 0.0

    # Unknown task — conservative fallback
    return 0.0


def get_claims_for_task(task_id: str) -> list[dict]:
    """Return a deep copy of the claim list for the given task (safe to shuffle in-place)."""
    claims = CLAIMS_BY_TASK.get(task_id, EASY_CLAIMS)
    return copy.deepcopy(claims)
