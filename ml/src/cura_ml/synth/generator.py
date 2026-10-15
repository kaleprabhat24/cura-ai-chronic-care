"""Event-driven synthetic chronic-care cohort simulator.

Each patient is simulated independently with its own random stream (spawned from the config seed),
so results are reproducible and adding patients does not change existing ones.

How outcomes arise (no label is ever written directly):
  * Hidden traits (engagement, access barrier, instability periods, trust) drive booking, attendance,
    cancellations and responses to outreach.
  * A review is "completed" only when a simulated appointment is attended. Follow-up episodes and
    labels are later reconstructed from the ordinary records by `cura_ml.cohort.episodes`.
  * Observable history (e.g. past no-shows) predicts future non-completion only because both
    share the same hidden causes.

Internally all times are integer minutes (or days) since `simulation.start_date` 00:00 UTC.
The final step masks everything that would not yet be recorded at the data cutoff.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from cura_ml.dataset import Records
from cura_ml.synth import catalog as cat

MIN_PER_DAY = 1440
OPEN_MIN = 8 * 60 + 30   # clinic opens 08:30
CLOSE_MIN = 17 * 60      # last slot before 17:00

COLUMNS: dict[str, list[str]] = {
    "patients": [
        "patient_id", "synthetic_mrn", "given_name", "family_name", "birth_date", "sex",
        "deprivation_quintile", "preferred_language", "registered_on", "death_date",
        "death_recorded_at", "transfer_out_date", "transfer_recorded_at", "is_synthetic",
    ],
    "conditions": [
        "condition_id", "patient_id", "code_system", "code", "display", "onset_date",
        "recorded_at", "clinical_status",
    ],
    "appointments": [
        "appointment_id", "patient_id", "appointment_type", "booked_at", "scheduled_for",
        "status", "status_recorded_at", "booking_source",
    ],
    "encounters": [
        "encounter_id", "patient_id", "encounter_type", "start_at", "end_at", "status",
        "appointment_id", "recorded_at",
    ],
    "observations": [
        "observation_id", "patient_id", "encounter_id", "code_system", "code", "display",
        "value_num", "unit", "effective_at", "recorded_at",
    ],
    "medication_statements": [
        "medication_statement_id", "patient_id", "code_system", "code", "display", "drug_class",
        "start_date", "end_date", "recorded_at", "end_recorded_at",
    ],
    "outreach_attempts": [
        "outreach_id", "patient_id", "programme", "purpose", "channel", "attempted_at",
        "outcome", "next_action", "next_action_due", "performed_by", "recorded_at",
    ],
    "simulation_traits": [
        "patient_id", "engagement", "access_barrier", "instability_entry_prob", "trust",
        "unstable_month_fraction", "hba1c_baseline", "sbp_baseline",
    ],
    "simulator_episode_log": [
        "patient_id", "programme", "episode_seq", "index_date", "due_date", "completion_date",
    ],
}
TIMESTAMP_COLS = {
    "patients": ["death_recorded_at", "transfer_recorded_at"],
    "conditions": ["recorded_at"],
    "appointments": ["booked_at", "scheduled_for", "status_recorded_at"],
    "encounters": ["start_at", "end_at", "recorded_at"],
    "observations": ["effective_at", "recorded_at"],
    "medication_statements": ["recorded_at", "end_recorded_at"],
    "outreach_attempts": ["attempted_at", "recorded_at"],
}
DATE_COLS = {
    "patients": ["birth_date", "registered_on", "death_date", "transfer_out_date"],
    "conditions": ["onset_date"],
    "medication_statements": ["start_date", "end_date"],
    "outreach_attempts": ["next_action_due"],
    "simulator_episode_log": ["index_date", "due_date", "completion_date"],
}
HIDDEN = ("simulation_traits", "simulator_episode_log")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class _Booking:
    appt_id: str
    status: str
    resolve_day: int
    status_min: int
    sched_min: int
    dur: int


@dataclass
class _Programme:
    name: str
    interval: int
    encounter_type: str
    incident: bool = False
    onset_day: int = 0
    first_review_min: int = 0
    ep_no: int = 0
    due_day: int = 0
    booking: _Booking | None = None
    log_row: dict | None = None
    first_review_done: bool = False


class PatientSimulator:
    def __init__(self, number: int, rng: np.random.Generator, cfg: dict[str, Any],
                 start_weekday: int, cutoff_day: int) -> None:
        self.num = number
        self.pid = f"pt_{number:06d}"
        self.rng = rng
        self.cfg = cfg
        self.start_weekday = start_weekday
        self.cutoff_day = cutoff_day
        self.rows: dict[str, list[dict]] = defaultdict(list)
        self._ids: dict[str, int] = defaultdict(int)
        self._heap: list[tuple] = []
        self._seq = 0
        self.noshow_days: list[int] = []
        self.active_meds: dict[str, dict] = {}
        self.progs: dict[str, _Programme] = {}

    # ------------------------------------------------------------------ helpers
    def _id(self, prefix: str) -> str:
        self._ids[prefix] += 1
        return f"{prefix}_{self.num:06d}_{self._ids[prefix]:04d}"

    def _workday(self, day: int) -> int:
        wd = (self.start_weekday + day) % 7
        return day + (7 - wd if wd >= 5 else 0)

    def _office_min(self) -> int:
        return int(self.rng.integers(OPEN_MIN, CLOSE_MIN))

    def _slot_min(self) -> int:
        return int(self.rng.integers(OPEN_MIN // 10, CLOSE_MIN // 10)) * 10

    def _unstable(self, day: int) -> bool:
        idx = min(max(day, 0) // 30, len(self.unstable_months) - 1)
        return bool(self.unstable_months[idx])

    def _push(self, day: int, kind: str, prog: str | None = None, ep: int | None = None,
              payload: Any = None) -> None:
        heapq.heappush(self._heap, (day, self._seq, kind, prog, ep, payload))
        self._seq += 1

    # ------------------------------------------------------------------ main
    def run(self) -> dict[str, list[dict]]:
        rng, pop, tr, cl = self.rng, self.cfg["population"], self.cfg["traits"], self.cfg["clinical"]

        self.sex = str(rng.choice(list(pop["sex_probs"]), p=list(pop["sex_probs"].values())))
        self.age0 = float(rng.uniform(pop["age_min"], pop["age_max"]))
        self.birth_day = -int(self.age0 * 365.25)
        self.deprivation = int(rng.integers(1, 6))
        langs = pop["language_probs"]
        self.language = str(rng.choice(list(langs), p=list(langs.values())))
        self.given = str(rng.choice(cat.GIVEN_NAMES[self.sex]))
        self.family = str(rng.choice(cat.FAMILY_NAMES))

        # Hidden traits.
        self.engagement = float(rng.normal())
        self.access = float(rng.normal() + tr["access_deprivation_effect"] * (self.deprivation - 3))
        lo, hi = tr["instability_entry_prob_range"]
        self.instab_entry = float(rng.uniform(lo, hi))
        c = tr["trust_engagement_corr"]
        self.trust = float(c * self.engagement + math.sqrt(1 - c * c) * rng.normal())

        # Registration and exits.
        if rng.random() < pop["registered_before_start_prob"]:
            reg = -int(rng.integers(180, 365 * 20))
        else:
            reg = int(rng.integers(1, pop["late_join_max_days"] + 1))
        self.reg_day = max(reg, self.birth_day + 18 * 365)
        self.active_from = max(0, self.reg_day)
        self._draw_exits()

        # Monthly instability regime (hidden Markov chain).
        n_months = (self.cutoff_day + 120) // 30 + 1
        exit_p = tr["instability_exit_prob"]
        state = rng.random() < self.instab_entry / (self.instab_entry + exit_p)
        months = np.empty(n_months, dtype=bool)
        for m in range(n_months):
            months[m] = state
            state = (rng.random() >= exit_p) if state else (rng.random() < self.instab_entry)
        self.unstable_months = months

        # Clinical baselines (hidden).
        self.hba1c_base = float(rng.normal(*cl["hba1c_baseline"]))
        self.sbp_base = float(rng.normal(*cl["sbp_baseline"]))
        self.weight = float(np.clip(rng.normal(*cl["weight_baseline"]), 45, 180))

        mix = pop["programme_mix"]
        group = str(rng.choice(list(mix), p=list(mix.values())))
        for name in {"t2d_only": ["t2d"], "htn_only": ["htn"], "both": ["t2d", "htn"]}[group]:
            self._setup_programme(name)
        self._schedule_acute()
        self._event_loop()
        self._apply_exits()
        self._emit_patient()
        return self.rows

    def _draw_exits(self) -> None:
        rng, pop = self.rng, self.cfg["population"]
        death = None
        for year_start in range(self.active_from, self.cutoff_day + 1, 365):
            age = self.age0 + year_start / 365.25
            hazard = pop["annual_death_hazard_at_60"] * math.exp(
                pop["death_hazard_age_slope"] * (age - 60))
            if rng.random() < 1 - math.exp(-hazard):
                death = year_start + int(rng.integers(0, 365))
                break
        transfer = self.active_from + int(rng.exponential(365.25 / pop["annual_transfer_out_rate"]))
        death = death if death is not None and death <= self.cutoff_day else None
        transfer = transfer if transfer <= self.cutoff_day else None
        if death is not None and transfer is not None:
            if transfer < death:
                death = None
            else:
                transfer = None
        self.death_day, self.transfer_day = death, transfer
        self.death_rec_min = self.transfer_rec_min = None
        if death is not None:
            self.death_rec_min = (death + int(rng.integers(1, 11))) * MIN_PER_DAY + self._office_min()
        if transfer is not None:
            self.transfer_rec_min = transfer * MIN_PER_DAY + 9 * 60
        self.end_day = min(
            self.cutoff_day,
            death - 1 if death is not None else self.cutoff_day,
            transfer - 1 if transfer is not None else self.cutoff_day,
        )

    def _setup_programme(self, name: str) -> None:
        rng, pop = self.rng, self.cfg["population"]
        pcfg = self.cfg["programmes"][name]
        prog = _Programme(name, pcfg["interval_days"], pcfg["encounter_type"])
        lo_inc, hi_inc = self.active_from + 30, self.cutoff_day - 60
        if rng.random() < pop["incident_condition_prob"] and lo_inc < hi_inc:
            prog.incident = True
            first = self._workday(int(rng.integers(lo_inc, hi_inc)))
            prog.onset_day = first  # diagnosed at this visit
            prog.first_review_min = first * MIN_PER_DAY + self._slot_min()
            if first > self.end_day:
                return  # exited before diagnosis: no condition, no programme
            cond_rec_min = prog.first_review_min + 20
        else:
            hi = max(self.reg_day, 0)
            lo = max(self.birth_day + 30 * 365, hi - 15 * 365)
            if lo >= hi:
                lo = hi - 30
            prog.onset_day = int(rng.integers(lo, hi))
            if self.reg_day <= 0:
                first = int(rng.integers(0, prog.interval))
            else:
                first = self.reg_day + int(rng.integers(14, 61))
            first = self._workday(first)
            prog.first_review_min = first * MIN_PER_DAY + self._slot_min()
            cond_day = min(max(prog.onset_day, self.reg_day) + int(rng.integers(0, 15)), first - 1)
            if cond_day > self.end_day:
                return
            cond_rec_min = cond_day * MIN_PER_DAY + self._office_min()
        c = cat.CONDITIONS[name]
        self.rows["conditions"].append({
            "condition_id": self._id("cd"), "patient_id": self.pid, "code_system": cat.SNOMED,
            "code": c["code"], "display": c["display"], "onset_date": prog.onset_day,
            "recorded_at": cond_rec_min, "clinical_status": "active",
        })
        self.progs[name] = prog
        if first <= self.end_day:
            self._push(first, "first_review", name)

    def _schedule_acute(self) -> None:
        rng, cl = self.rng, self.cfg["clinical"]
        mult = cl["acute_unstable_multiplier"]
        max_rate = cl["acute_visits_per_year"] * mult / 365.25
        t = float(self.active_from)
        while True:
            t += rng.exponential(1.0 / max_rate)
            day = int(t)
            if day > self.end_day:
                break
            if rng.random() < (mult if self._unstable(day) else 1.0) / mult:
                self._push(day, "acute")

    def _event_loop(self) -> None:
        while self._heap:
            day, _, kind, prog, ep, payload = heapq.heappop(self._heap)
            if day > self.end_day:
                break  # heap is day-ordered: everything left is after the end of observation
            if ep is not None and ep != self.progs[prog].ep_no:
                continue  # belongs to an episode that is already closed
            getattr(self, f"_on_{kind}")(day, prog, payload)

    # ------------------------------------------------------------------ care process
    def _start_episode(self, name: str, index_day: int, after_min: int) -> None:
        rng, cp, op = self.rng, self.cfg["care_process"], self.cfg["outreach_policy"]
        prog = self.progs[name]
        prog.ep_no += 1
        prog.due_day = due = index_day + prog.interval
        prog.booking = None
        prog.log_row = {"patient_id": self.pid, "programme": name, "episode_seq": prog.ep_no,
                        "index_date": index_day, "due_date": due, "completion_date": None}
        self.rows["simulator_episode_log"].append(prog.log_row)
        ep = prog.ep_no
        p_prebook = _sigmoid(cp["prebook_base_logit"] + cp["prebook_engagement_weight"] * self.engagement)
        if rng.random() < p_prebook:
            lo, hi = cp["prebook_offset_days"]
            self._create_booking(name, after_min + 5, due + int(rng.integers(lo, hi + 1)),
                                 "practice_prebook")
        else:
            self._push(self._workday(due - cp["invite_lead_days"]), "invite", name, ep)
        self._push(self._workday(due + op["pre_due_reminder_offset_days"]), "reminder", name, ep)
        for k, off in enumerate(op["chase_offsets_days"]):
            self._push(self._workday(due + off), "chase", name, ep, k)
        for j in range(1, cp["max_re_invites"] + 1):
            self._push(self._workday(due + j * cp["re_invite_every_days"]), "letter", name, ep)

    def _p_attend(self, sched_day: int, lead: int) -> float:
        a = self.cfg["attendance"]
        wd = (self.start_weekday + sched_day) % 7
        weekday = {0: a["weekday_effects"].get("mon", 0.0), 4: a["weekday_effects"].get("fri", 0.0)}
        recent = any(sched_day - 180 <= d < sched_day for d in self.noshow_days)
        logit = (
            a["base_logit"]
            + a["engagement_weight"] * self.engagement
            - a["access_weight"] * self.access
            - a["unstable_weight"] * self._unstable(sched_day)
            - a["long_lead_weight"] * (min(max(lead, 14), a["long_lead_cap_days"]) - 14) / 30
            + a["year_drift"] * sched_day / 365.25
            + weekday.get(wd, 0.0)
            - a["recent_noshow_weight"] * recent
            + self.rng.normal(0, a["noise_sd"])
        )
        return _sigmoid(logit)

    def _create_booking(self, name: str, booked_min: int, sched_day: int, source: str) -> None:
        rng, cp = self.rng, self.cfg["care_process"]
        prog = self.progs[name]
        booked_day = booked_min // MIN_PER_DAY
        sched_day = self._workday(max(sched_day, booked_day + 1))
        sched_min = sched_day * MIN_PER_DAY + self._slot_min()
        lead = sched_day - booked_day
        dur = int(rng.integers(15, 31))
        p_pcancel = _sigmoid(cp["patient_cancel_base_logit"]
                             + cp["patient_cancel_unstable_weight"] * self._unstable(sched_day))
        if lead >= 2 and rng.random() < cp["clinic_cancel_prob"]:
            status, res_day = "cancelled_clinic", int(rng.integers(booked_day + 1, sched_day))
            status_min = res_day * MIN_PER_DAY + self._office_min()
        elif lead >= 2 and rng.random() < p_pcancel:
            status = "cancelled_patient"
            res_day = max(booked_day + 1, sched_day - int(rng.integers(1, 8)))
            status_min = res_day * MIN_PER_DAY + self._office_min()
        else:
            status = "attended" if rng.random() < self._p_attend(sched_day, lead) else "no_show"
            res_day, status_min = sched_day, sched_min + dur
        appt_id = self._id("ap")
        self.rows["appointments"].append({
            "appointment_id": appt_id, "patient_id": self.pid,
            "appointment_type": prog.encounter_type, "booked_at": booked_min,
            "scheduled_for": sched_min, "status": status, "status_recorded_at": status_min,
            "booking_source": source,
        })
        prog.booking = _Booking(appt_id, status, res_day, status_min, sched_min, dur)
        self._push(res_day, "resolve", name, prog.ep_no, appt_id)

    def _on_first_review(self, day: int, name: str, _: Any) -> None:
        prog = self.progs[name]
        sched_min, dur = prog.first_review_min, int(self.rng.integers(15, 31))
        booked_day = max(day - int(self.rng.integers(5, 22)), self.active_from)
        booked_min = min(booked_day * MIN_PER_DAY + self._office_min(), sched_min - 60)
        appt_id = self._id("ap")
        self.rows["appointments"].append({
            "appointment_id": appt_id, "patient_id": self.pid,
            "appointment_type": prog.encounter_type, "booked_at": booked_min,
            "scheduled_for": sched_min, "status": "attended", "status_recorded_at": sched_min + dur,
            "booking_source": "practice",
        })
        self._record_review(name, sched_min, dur, appt_id)
        self._start_episode(name, day, sched_min + dur)

    def _on_resolve(self, day: int, name: str, appt_id: str) -> None:
        rng, cp = self.rng, self.cfg["care_process"]
        prog = self.progs[name]
        b = prog.booking
        if b is None or b.appt_id != appt_id:
            return
        prog.booking = None
        rebook_eng = cp["rebook_engagement_weight"] * self.engagement
        if b.status == "attended":
            self._record_review(name, b.sched_min, b.dur, appt_id)
            prog.log_row["completion_date"] = day
            self._start_episode(name, day, b.sched_min + b.dur)
        elif b.status in ("no_show", "cancelled_patient"):
            if b.status == "no_show":
                self.noshow_days.append(day)
                base = cp["rebook_after_noshow_base_logit"]
            else:
                base = cp["rebook_after_cancel_base_logit"]
            if rng.random() < _sigmoid(base + rebook_eng - 0.3 * self.access):
                delay = 1 + int(rng.exponential(cp["rebook_delay_mean_days"]))
                self._push(day + delay, "book", name, prog.ep_no, "patient")
        else:  # cancelled_clinic: the practice rebooks straight away, shortly after the lost slot
            lo, hi = cp["clinic_rebook_lead_days"]
            original_day = b.sched_min // MIN_PER_DAY
            self._create_booking(name, b.status_min + 5, original_day + int(rng.integers(lo, hi + 1)),
                                 "clinic_rebook")

    def _on_invite(self, day: int, name: str, _: Any) -> None:
        rng, cp = self.rng, self.cfg["care_process"]
        prog = self.progs[name]
        if prog.booking is not None:
            return
        p = _sigmoid(cp["invite_response_base_logit"]
                     + cp["invite_response_engagement_weight"] * self.engagement
                     - cp["invite_response_access_weight"] * self.access
                     - cp["invite_response_unstable_weight"] * self._unstable(day))
        if rng.random() < p:
            mean = cp["booking_delay_mean_days"] * math.exp(-0.3 * self.engagement)
            self._push(day + 1 + int(rng.exponential(mean)), "book", name, prog.ep_no, "patient")

    def _on_book(self, day: int, name: str, source: str) -> None:
        if self.progs[name].booking is not None:
            return
        lo, hi = self.cfg["care_process"]["booking_lead_days"]
        self._create_booking(name, day * MIN_PER_DAY + self._office_min(),
                             day + int(self.rng.integers(lo, hi + 1)), source)

    def _on_reminder(self, day: int, name: str, _: Any) -> None:
        op = self.cfg["outreach_policy"]
        if self.progs[name].booking is not None:
            return
        lookback = op["pre_due_reminder_lookback_days"]
        if any(day - lookback <= d < day for d in self.noshow_days) and \
                self.rng.random() < op["pre_due_reminder_prob"]:
            self._outreach(name, day, "pre_due_reminder", "phone")

    def _on_chase(self, day: int, name: str, k: int) -> None:
        if self.progs[name].booking is None and \
                self.rng.random() < self.cfg["outreach_policy"]["chase_probs"][k]:
            self._outreach(name, day, "post_due_chase", "phone")

    def _on_letter(self, day: int, name: str, _: Any) -> None:
        if self.progs[name].booking is None:
            self._outreach(name, day, "recall_letter", "letter")

    def _outreach(self, name: str, day: int, purpose: str, channel: str) -> None:
        rng, op, cp = self.rng, self.cfg["outreach_policy"], self.cfg["care_process"]
        prog = self.progs[name]
        minute = day * MIN_PER_DAY + self._office_min()
        book_after: tuple[int, int, str] | None = None
        if channel == "letter":
            outcome, next_action, next_due = "letter_sent", "await_response", day + 28
            p = _sigmoid(op["letter_response_base_logit"]
                         + op["letter_response_engagement_weight"] * self.engagement)
            if rng.random() < p:
                mean = cp["booking_delay_mean_days"]
                self._push(day + 1 + int(rng.exponential(mean)), "book", name, prog.ep_no, "patient")
        else:
            p_reach = _sigmoid(op["reach_base_logit"] + op["reach_trust_weight"] * self.trust
                               - op["reach_unstable_weight"] * self._unstable(day))
            if rng.random() < p_reach:
                p_book = _sigmoid(op["book_when_reached_base_logit"]
                                  + op["book_when_reached_engagement_weight"] * self.engagement)
                if rng.random() < p_book:
                    outcome, next_action, next_due = "reached_booked", "none", None
                    lo, hi = cp["outreach_booking_lead_days"]
                    book_after = (minute + 10, day + int(rng.integers(lo, hi + 1)), "outreach")
                else:
                    outcome, next_action, next_due = "reached_declined", "retry_later", day + 30
            else:
                outcome = str(rng.choice(["no_answer", "left_message", "wrong_number"],
                                         p=[0.55, 0.35, 0.10]))
                next_action = "update_contact_details" if outcome == "wrong_number" else "retry"
                next_due = day + int(rng.integers(2, 8))
        self.rows["outreach_attempts"].append({
            "outreach_id": self._id("or"), "patient_id": self.pid, "programme": name,
            "purpose": purpose, "channel": channel, "attempted_at": minute, "outcome": outcome,
            "next_action": next_action, "next_action_due": next_due,
            "performed_by": "historical_staff", "recorded_at": minute + int(rng.integers(1, 30)),
        })
        if book_after is not None:
            self._create_booking(name, *book_after)

    # ------------------------------------------------------------------ clinical records
    def _n_active(self, ladder: str) -> int:
        return sum(1 for k in cat.LADDERS[ladder] if k in self.active_meds)

    def _hba1c_true(self, day: int) -> float:
        cl = self.cfg["clinical"]
        trend = cl["hba1c_trend_per_year"] - cl["hba1c_trend_engagement_weight"] * self.engagement
        return (self.hba1c_base + trend * day / 365.25
                + cl["hba1c_drug_effect"] * self._n_active("t2d"))

    def _sbp_true(self, day: int) -> float:
        cl = self.cfg["clinical"]
        trend = cl["sbp_trend_per_year"] - cl["sbp_trend_engagement_weight"] * self.engagement
        return self.sbp_base + trend * day / 365.25 + cl["sbp_drug_effect"] * self._n_active("htn")

    def _obs(self, enc_id: str, key: str, value: float, eff_min: int, rec_min: int) -> None:
        code, display, unit, _ = cat.OBSERVATIONS[key]
        self.rows["observations"].append({
            "observation_id": self._id("ob"), "patient_id": self.pid, "encounter_id": enc_id,
            "code_system": cat.LOINC, "code": code, "display": display, "value_num": value,
            "unit": unit, "effective_at": eff_min, "recorded_at": rec_min,
        })

    def _vital(self, enc_id: str, key: str, value: float, eff_min: int) -> None:
        self._obs(enc_id, key, value, eff_min, eff_min + int(self.rng.integers(1, 20)))

    def _bp(self, enc_id: str, day: int, eff_min: int) -> float:
        rng, cl = self.rng, self.cfg["clinical"]
        sbp = float(np.clip(round(self._sbp_true(day) + rng.normal(0, cl["sbp_measure_sd"])), 85, 225))
        dbp = float(np.clip(round(0.5 * sbp + 12 + rng.normal(0, 6)), 45, 135))
        self._vital(enc_id, "sbp", sbp, eff_min)
        self._vital(enc_id, "dbp", dbp, eff_min)
        return sbp

    def _encounter(self, etype: str, start_min: int, dur: int, appt_id: str | None) -> str:
        enc_id = self._id("en")
        end_min = start_min + dur
        self.rows["encounters"].append({
            "encounter_id": enc_id, "patient_id": self.pid, "encounter_type": etype,
            "start_at": start_min, "end_at": end_min, "status": "finished",
            "appointment_id": appt_id, "recorded_at": end_min + int(self.rng.integers(0, 121)),
        })
        return enc_id

    def _record_review(self, name: str, start_min: int, dur: int, appt_id: str) -> None:
        rng, cl = self.rng, self.cfg["clinical"]
        prog = self.progs[name]
        day = start_min // MIN_PER_DAY
        enc_id = self._encounter(prog.encounter_type, start_min, dur, appt_id)
        self.weight = float(np.clip(self.weight + rng.normal(0, 1.5), 40, 200))
        self._vital(enc_id, "weight", round(self.weight, 1), start_min + 3)
        if name == "t2d":
            hba1c = float(np.clip(self._hba1c_true(day) + rng.normal(0, cl["hba1c_measure_sd"]),
                                  4.5, 15.0))
            lab_rec = (day + int(rng.integers(1, 5))) * MIN_PER_DAY + self._office_min()
            self._obs(enc_id, "hba1c", round(hba1c, 1), start_min + 5, lab_rec)
            if "htn" in self.progs or rng.random() < cl["bp_at_t2d_review_prob"]:
                self._bp(enc_id, day, start_min + 8)
            measured, threshold = hba1c, cl["hba1c_intensify_threshold"]
        else:
            measured, threshold = self._bp(enc_id, day, start_min + 8), cl["sbp_intensify_threshold"]
        self._update_meds(name, day, start_min + dur, measured, threshold)
        prog.first_review_done = True

    def _update_meds(self, name: str, day: int, rec_min: int, measured: float,
                     threshold: float) -> None:
        rng, cl = self.rng, self.cfg["clinical"]
        prog = self.progs[name]
        ladder = cat.LADDERS[name]
        if not prog.first_review_done:
            start = day
            if not prog.incident:
                start = min(prog.onset_day + int(rng.integers(0, 181)), day)
            self._start_med(ladder[0], start, rec_min)
            if name == "t2d" and rng.random() < cl["statin_prob_t2d"]:
                self._start_med("atorvastatin", start, rec_min)
            return
        managed = set(ladder) | ({"atorvastatin"} if name == "t2d" else set())
        for key in sorted(k for k in self.active_meds if k in managed):
            if rng.random() < cl["med_stop_prob"]:
                row = self.active_meds.pop(key)
                row["end_date"], row["end_recorded_at"] = day, rec_min
        if measured > threshold and rng.random() < cl["intensify_prob"]:
            nxt = next((k for k in ladder if k not in self.active_meds), None)
            if nxt is not None:
                self._start_med(nxt, day, rec_min)

    def _start_med(self, key: str, start_day: int, rec_min: int) -> None:
        code, display, drug_class = cat.MEDICATIONS[key]
        row = {
            "medication_statement_id": self._id("ms"), "patient_id": self.pid,
            "code_system": cat.RXNORM, "code": code, "display": display, "drug_class": drug_class,
            "start_date": start_day, "end_date": None, "recorded_at": rec_min,
            "end_recorded_at": None,
        }
        self.rows["medication_statements"].append(row)
        self.active_meds[key] = row

    def _on_acute(self, day: int, _: Any, __: Any) -> None:
        rng, cl = self.rng, self.cfg["clinical"]
        start_min = day * MIN_PER_DAY + self._office_min()
        enc_id = self._encounter("acute", start_min, int(rng.integers(10, 21)), None)
        self._vital(enc_id, "heart_rate", float(np.clip(round(rng.normal(76, 11)), 40, 170)),
                    start_min + 2)
        if rng.random() < cl["acute_bp_prob"]:
            self._bp(enc_id, day, start_min + 4)
        if rng.random() < cl["acute_weight_prob"]:
            self._vital(enc_id, "weight", round(self.weight, 1), start_min + 3)

    # ------------------------------------------------------------------ finalisation
    def _apply_exits(self) -> None:
        """Appointments still pending at death/transfer are cancelled once the exit is recorded,
        or recorded as no-shows if their slot passed before the practice learned of the exit."""
        exit_day = self.death_day if self.death_day is not None else self.transfer_day
        if exit_day is None:
            return
        exit_rec = self.death_rec_min if self.death_day is not None else self.transfer_rec_min
        exit_min = exit_day * MIN_PER_DAY
        for row in self.rows["appointments"]:
            if row["scheduled_for"] >= exit_min and row["status_recorded_at"] >= exit_min:
                if exit_rec < row["scheduled_for"]:
                    row["status"], row["status_recorded_at"] = "cancelled_clinic", exit_rec
                else:
                    row["status"], row["status_recorded_at"] = "no_show", row["scheduled_for"] + 15

    def _emit_patient(self) -> None:
        self.rows["patients"].append({
            "patient_id": self.pid, "synthetic_mrn": f"SYN{self.num:07d}",
            "given_name": self.given, "family_name": self.family, "birth_date": self.birth_day,
            "sex": self.sex, "deprivation_quintile": self.deprivation,
            "preferred_language": self.language, "registered_on": self.reg_day,
            "death_date": self.death_day, "death_recorded_at": self.death_rec_min,
            "transfer_out_date": self.transfer_day, "transfer_recorded_at": self.transfer_rec_min,
            "is_synthetic": True,
        })
        self.rows["simulation_traits"].append({
            "patient_id": self.pid, "engagement": self.engagement, "access_barrier": self.access,
            "instability_entry_prob": self.instab_entry, "trust": self.trust,
            "unstable_month_fraction": float(self.unstable_months.mean()),
            "hba1c_baseline": self.hba1c_base, "sbp_baseline": self.sbp_base,
        })


# ---------------------------------------------------------------------- cohort assembly
def generate(cfg: dict[str, Any]) -> tuple[Records, dict[str, pd.DataFrame]]:
    """Simulate the cohort. Returns (ordinary records as known at the cutoff, hidden tables)."""
    sim = cfg["simulation"]
    start = date.fromisoformat(sim["start_date"])
    cutoff_day = (date.fromisoformat(sim["data_cutoff"]) - start).days
    children = np.random.SeedSequence(cfg["seed"]).spawn(sim["n_patients"])
    rows: dict[str, list[dict]] = defaultdict(list)
    for number, child in enumerate(children, start=1):
        sim_rows = PatientSimulator(number, np.random.default_rng(child), cfg, start.weekday(),
                                    cutoff_day).run()
        for table, table_rows in sim_rows.items():
            rows[table].extend(table_rows)
    frames = {t: pd.DataFrame(rows.get(t, []), columns=cols) for t, cols in COLUMNS.items()}
    _mask_after_cutoff(frames, (cutoff_day + 1) * MIN_PER_DAY - 1)
    for table, df in frames.items():
        frames[table] = _convert_times(table, df, start)
    records = Records(**{t: frames[t] for t in COLUMNS if t not in HIDDEN})
    return records, {t: frames[t] for t in HIDDEN}


def _mask_after_cutoff(f: dict[str, pd.DataFrame], cutoff_min: int) -> None:
    """Make the dataset a snapshot "as known at the data cutoff"."""
    for table in ("conditions", "encounters", "observations", "medication_statements",
                  "outreach_attempts"):
        f[table] = f[table][f[table]["recorded_at"] <= cutoff_min].reset_index(drop=True)
    ap = f["appointments"]
    ap = ap[ap["booked_at"] <= cutoff_min].reset_index(drop=True)
    pending = ap["status_recorded_at"] > cutoff_min
    ap.loc[pending, "status"] = "booked"
    ap.loc[pending, "status_recorded_at"] = ap.loc[pending, "booked_at"]
    f["appointments"] = ap
    ms = f["medication_statements"]
    unknown_end = ms["end_recorded_at"].notna() & (ms["end_recorded_at"] > cutoff_min)
    ms.loc[unknown_end, ["end_date", "end_recorded_at"]] = None
    pt = f["patients"]
    for day_col, rec_col in (("death_date", "death_recorded_at"),
                             ("transfer_out_date", "transfer_recorded_at")):
        unknown = pt[rec_col].notna() & (pt[rec_col] > cutoff_min)
        pt.loc[unknown, [day_col, rec_col]] = None


def _convert_times(table: str, df: pd.DataFrame, start: date) -> pd.DataFrame:
    df = df.copy()
    origin = pd.Timestamp(start, tz="UTC")
    for col in TIMESTAMP_COLS.get(table, []):
        minutes = pd.to_numeric(df[col], errors="coerce")
        df[col] = origin + pd.to_timedelta(minutes, unit="min")
    for col in DATE_COLS.get(table, []):
        days = pd.to_numeric(df[col], errors="coerce")
        values = [None if pd.isna(d) else start + timedelta(days=int(d)) for d in days]
        df[col] = pd.array(values, dtype=pd.ArrowDtype(pa.date32()))
    return df
