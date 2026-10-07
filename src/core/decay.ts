import type { Importance } from "../schema/index.js";

/**
 * Decay constants, per PCM-SPEC.md section 3.1.
 *
 *   T_eff = T_DECAY_DAYS * (1 + SAVINGS_COEFFICIENT * min(n, SAVINGS_CAP))
 *   tau   = T_eff / TAU_DIVISOR
 *   S(t)  = clamp(S_0 * exp(-t / tau))
 *
 * `decayRate` in calculateDecayedStrength is 1/tau at n = 0, i.e.
 * TAU_DIVISOR / T_DECAY_DAYS, which is what makes DEFAULT_DECAY_RATE below
 * equal 1/30 per day. Keeping the parameter as a rate preserves the existing
 * signature while making its value derivable from the spec rather than
 * arbitrary.
 */
export const T_DECAY_DAYS = 90;
export const TAU_DIVISOR = 3;
export const SAVINGS_COEFFICIENT = 0.25;
export const SAVINGS_CAP = 20;

/** 1/30 per day: tau = 30 days at n = 0, so exp(-90/30) = exp(-3) at 90 days. */
export const DEFAULT_DECAY_RATE = TAU_DIVISOR / T_DECAY_DAYS;
export const PINNED_STRENGTH = 1.0;
export const HIGH_INITIAL_STRENGTH = 0.85;
export const DEFAULT_INITIAL_STRENGTH = 0.70;
export const MIN_DECAYED_STRENGTH = 0.01;

const INVARIANT_PATTERNS = [
  /anaphylactic/i,
  /life-threatening\s+allergy/i,
  /epipen/i,
  /severe\s+allergy/i,
  /never\s+disclose/i,
  /critical.*invariant/i,
  /confidential.*invariant/i,
  /strict.*invariant/i,
  /prescription\s+medication.*anxiety/i,
];

export function isInvariantContent(text: string): boolean {
  if (!text) return false;
  return INVARIANT_PATTERNS.some((pat) => pat.test(text));
}

export function getInitialStrength(importance: Importance, text?: string): number {
  if (importance === "pinned" || (text && isInvariantContent(text))) {
    return PINNED_STRENGTH;
  }
  switch (importance) {
    case "high":
      return HIGH_INITIAL_STRENGTH;
    case "default":
    default:
      return DEFAULT_INITIAL_STRENGTH;
  }
}

/**
 * Calculates decayed strength using the Ebbinghaus exponential decay model of
 * PCM-SPEC.md section 3.1:
 *
 *   S(t) = S_0 * exp(-lambda * delta_t / (1 + 0.25 * min(B, 20)))
 *
 * where lambda = TAU_DIVISOR / T_DECAY_DAYS = 1/30 per day, so tau is 30 days
 * unreinforced and 180 days at the B = 20 cap (T_eff = 540, tau = 540/3).
 *
 * Note this savings factor is linear and capped, and is distinct from the
 * logarithmic *retrieval boost* of section 3.2 (delta-S = 0.17 * ln(1 + n)),
 * which replenishes stored strength on access rather than lengthening tau.
 */
export function calculateDecayedStrength(
  initialStrength: number,
  elapsedMs: number,
  boostCount: number = 0,
  importance: Importance = "default",
  decayRate: number = DEFAULT_DECAY_RATE,
  text?: string,
): number {
  if (importance === "pinned" || (text && isInvariantContent(text))) {
    return PINNED_STRENGTH;
  }

  // Garbage in, floor out. Callers can pass NaN here via unparseable date
  // arithmetic, and NaN propagates through every downstream score
  // comparison (NaN < x is always false, so rankers keep the row and sort
  // unpredictably). A memory we cannot date must rank last, never break
  // ordering. Future-dated (negative) elapsed still clamps to zero below.
  if (
    !Number.isFinite(elapsedMs) ||
    !Number.isFinite(boostCount) ||
    !Number.isFinite(initialStrength) ||
    !Number.isFinite(decayRate)
  ) {
    return MIN_DECAYED_STRENGTH;
  }

  const elapsedDays = Math.max(0, elapsedMs / (1000 * 60 * 60 * 24));
  // Savings effect, per spec 3.1: T_eff = T_decay * (1 + 0.25 * min(n, 20)),
  // i.e. linear in retrieval count and capped at 20 retrievals for a sixfold
  // time constant. A logarithmic factor grows far too slowly to offset linear
  // decay -- under 1 + ln(1 + n) no retrieval count keeps a default-importance
  // memory above the stale threshold at one year, which defeats the purpose of
  // modelling reinforcement at all.
  const savingsFactor =
    1.0 + SAVINGS_COEFFICIENT * Math.min(Math.max(0, boostCount), SAVINGS_CAP);
  const effectiveDecay = (decayRate * elapsedDays) / savingsFactor;
  const decayed = initialStrength * Math.exp(-effectiveDecay);

  return Math.max(MIN_DECAYED_STRENGTH, Math.min(1.0, decayed));
}

export function boostStrengthOnAccess(currentStrength: number, importance: Importance): number {
  if (importance === "pinned") {
    return PINNED_STRENGTH;
  }
  const boosted = currentStrength + (1.0 - currentStrength) * 0.35;
  return Math.min(1.0, boosted);
}

export function calculateReRankScore(
  item: {
    memoryId: string;
    similarity: number;
    strength: number;
    createdAt?: Date;
    isProjectMatch?: boolean;
  },
  now: Date = new Date(),
): number {
  let score = item.similarity * 0.70 + item.strength * 0.30;
  if (item.isProjectMatch) {
    score += 0.08;
  }
  return score;
}
