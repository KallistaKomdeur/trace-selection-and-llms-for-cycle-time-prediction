# Synthetic Hospital Benchmark — patient journey with a known concept drift

This folder contains a **fully synthetic** event log of a hospital patient journey, *from admission to
discharge*, in two versions — an undrifted **baseline** and a **drifted** copy carrying a single, exactly
known concept drift. It is meant as a controlled benchmark for concept-drift detection and, more broadly,
for predictive process monitoring, business-process simulation, and prescriptive process analytics, where
methods must be evaluated against realistic behaviour with a **gold-standard ground truth**.

Unlike log collections obtained by editing real logs, this process is **invented end-to-end**: activities,
control-flow, durations, resources, arrivals and case attributes are all generated procedurally, so nothing
is tied to a real organisation and the ground truth is exact by construction.

## The process
A patient flows through an emergency/inpatient pathway with a triage-driven branching structure:

```
Admission → Triage → Initial Assessment
          → [Lab Tests] → [Imaging] → [Specialist Consultation]
          → Diagnosis
          → ( Surgery → Post-op Observation → Medication Administration* )   # inpatient / surgical branch
            or ( Treatment → Medication Administration* )                    # non-surgical branch
          → Discharge → Billing & Payment
```

Optional steps (in brackets) and the surgical vs. non-surgical branch are drawn per case; the probability of
imaging and of surgery depends on the patient's **triage level**, so more urgent patients follow longer,
more complex paths. This yields many distinct trace variants over **13 activities**.

## Perspectives modelled
- **Control-flow** — the branching pathway above (mandatory core + optional/conditional steps + a repeatable
  `Medication Administration`).
- **Time** — every activity has a realistic **duration** sampled from a log-normal distribution with a
  domain-plausible median (triage minutes, lab hours, post-op observation days), separated by **waiting
  times**; patient **arrivals** follow a Poisson process.
- **Resource** — a deliberate **mix**: single-tasking units (15 doctors, 20 nurses, 4 imaging machines — one
  patient at a time) and multitasking shared units (laboratory teams, wards, admissions desk, billing) that
  legitimately handle work concurrently.
- **Data** — each case carries the attributes `Age` (numeric), `Department`
  (Cardiology / Orthopedics / Neurology / General / Pediatrics) and `TriageLevel`
  (Emergency / Urgent / Routine).

## Generation summary
- **2,500 cases**, ~23,160 events, 13 activities; seed **2024** (fully reproducible).
- Poisson arrivals (~1 admission every 1.5 h) spanning roughly Jan–Jun 2024.
- Baseline median length of stay ≈ **19 h**.

## The injected drift (ground truth)
| field | value |
|---|---|
| perspective | time |
| technique | slow-down, factor **×4** |
| target activity | `Lab Tests` |
| shape | sudden |
| change point | case **1265** of 2500 (2024-03-17) |
| interpretation | laboratory backlog / capacity drop |

The drift rescales **only** the duration of `Lab Tests` for every case after the change point (waiting times
kept constant, arrivals never moved). Its effect:
- `Lab Tests` median duration **2.49 h → 10.34 h** (≈ ×4) after the change point;
- length of stay median **≈ 19 h → ≈ 22 h**;
- directly-follows total variation pre/post = **0.024** in *both* baseline and drifted log → the drift is
  **isolated to the time perspective**, the control-flow is untouched.

Because `Lab Tests` is performed by a **shared/multitasking** unit, no resource reassignment is needed; had
the target been a single-tasking resource, a conflict-resolution policy would apply instead.

## Files
- `hospital_baseline.xes` — the undrifted synthetic log.
- `hospital_drifted.xes` — the same log with the injected drift (same cases/arrivals, only the drift added).
- `ground_truth.json` — machine-readable ground truth (process, resources, attributes, drift, seed).
- `creation.tex` — a paper-ready LaTeX section describing, at a high level, how both logs were created.

## Reproducibility
Everything is deterministic given seed **2024**: the same process, the same random change point, and the same
drift are obtained on every run.
