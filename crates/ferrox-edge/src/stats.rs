//! Serving telemetry that is arithmetic, not measurement.
//!
//! Three pieces, and each exists because the obvious version of it lies
//! in a specific way.
//!
//! **[`RequestRing`] carries an all-time cursor.** A bounded ring that
//! renumbers on eviction leaves a poller two bad options: re-read every
//! row each time, or silently skip whatever fell out between polls. Rows
//! here keep a monotonic sequence number for the life of the process, so
//! [`RequestRing::since`] can say what is new *and* how much was missed
//! -- a client that came back after a burst learns it lost rows instead
//! of quietly under-reporting. The `limit` rule matters as much as the
//! cursor: a truncated page reports the cursor of its last returned row
//! plus one, and only a page that returned everything it matched may
//! report the all-time count. Return the all-time count from a truncated
//! page and the next poll skips exactly the rows the limit cut off.
//!
//! **[`percentile`] is nearest-rank.** Interpolated percentiles are fine
//! for continuous data and wrong for a window of twelve requests: they
//! report a p95 latency no request ever had. Nearest-rank always names a
//! real observation.
//!
//! **[`RateWindow`] divides by the observed span, not by the window.**
//! This is the subtle one. A server running at 50 tok/s, polled one
//! second into a five-second window, has produced 50 tokens; dividing by
//! the window reports 10 tok/s and the number is simply wrong. Dividing
//! by `now - oldest_retained_sample` reports 50, and still decays to
//! zero as the samples age out -- which is the property that matters,
//! because a *cumulative* average would have an idle server reporting
//! the rate it managed an hour ago forever.
//!
//! Every clock reading is a parameter. Nothing here reads the wall
//! clock, so every rule is testable without waiting for one.
//!
//! Ported from FreeToken's `server/request_ring.py` and
//! `server/stats.py` (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.
//! One deliberate departure, in both directions from the same rule:
//! where upstream returns `0` for "no request had one", this returns
//! `None`. Zero is a measurement; absence is not, and `cache_report`
//! already establishes the house rule that a column nothing can be said
//! about is dropped rather than zero-filled.

use std::collections::{BTreeSet, VecDeque};

/// Samples retained per rate window.
///
/// Eviction is poll-driven -- only [`RateWindow::tokens_per_second`]
/// trims by age -- so a deployment whose clients never read `/v1/stats`
/// would otherwise grow these without bound. Generous against the
/// window's span at any realistic reply rate.
pub const RATE_SAMPLE_CAPACITY: usize = 4096;

/// A bounded ring of finished requests, addressed by an all-time
/// sequence number.
///
/// The sequence counts every row ever pushed, not every row retained,
/// which is what lets a poller detect its own gap.
#[derive(Debug, Clone)]
pub struct RequestRing<T> {
    rows: VecDeque<(u64, T)>,
    capacity: usize,
    /// The sequence the next pushed row will get, and the count of every
    /// row this process has ever recorded.
    next_seq: u64,
}

/// What one poll of the ring returned, and what it could not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingPage<'a, T> {
    pub rows: Vec<&'a T>,
    /// Pass this back as the next `since`.
    ///
    /// When the page was truncated by `limit`, this is one past the last
    /// row actually returned -- *not* the all-time count, which would
    /// skip everything the limit cut off. Only a page that returned
    /// every row it matched may report the all-time count.
    pub cursor: u64,
    /// Rows that existed and were evicted before this poll could see
    /// them. Non-zero means the caller is polling slower than the server
    /// is finishing requests: a fact worth surfacing, not one worth
    /// hiding by returning fewer rows.
    pub missed: u64,
}

impl<T> RequestRing<T> {
    /// `capacity` is clamped to at least 1: a zero-capacity ring counts
    /// rows it can never show, which is a stats endpoint that reports
    /// nothing while claiming to have seen everything.
    pub fn new(capacity: usize) -> Self {
        RequestRing {
            rows: VecDeque::new(),
            capacity: capacity.max(1),
            next_seq: 0,
        }
    }

    /// Records one row and returns the sequence it was given.
    pub fn push(&mut self, row: T) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        if self.rows.len() == self.capacity {
            self.rows.pop_front();
        }
        self.rows.push_back((seq, row));
        seq
    }

    /// Up to `limit` rows recorded at or after `since`.
    ///
    /// A caller starting fresh passes 0 and gets whatever is retained;
    /// `missed` then says how much history predates the ring, which is a
    /// legitimate answer rather than an error.
    pub fn since(&self, since: u64, limit: usize) -> RingPage<'_, T> {
        let oldest = self
            .rows
            .front()
            .map(|(seq, _)| *seq)
            .unwrap_or(self.next_seq);
        let missed = oldest.saturating_sub(since);
        let matched: Vec<&(u64, T)> = self.rows.iter().filter(|(seq, _)| *seq >= since).collect();
        let returned = &matched[..limit.min(matched.len())];
        let cursor = if returned.len() < matched.len() {
            // Truncated: resume at the row after the last one delivered.
            returned.last().map(|(seq, _)| seq + 1).unwrap_or(since)
        } else {
            self.next_seq
        };
        RingPage {
            rows: returned.iter().map(|(_, row)| row).collect(),
            cursor,
            missed,
        }
    }

    /// Every retained row, oldest first.
    pub fn rows(&self) -> impl Iterator<Item = &T> {
        self.rows.iter().map(|(_, row)| row)
    }

    /// How many rows this process has recorded in total, retained or
    /// not.
    pub fn recorded_total(&self) -> u64 {
        self.next_seq
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// The `p`-th percentile of `values`, nearest-rank.
///
/// `p` is a percentage in `[0, 100]`. The result is always a value that
/// is actually in `values`: over a handful of requests an interpolated
/// percentile reports a latency nothing measured, and a UI showing it
/// beside a request list invites the reader to look for the row it came
/// from.
///
/// `None` for an empty input -- a percentile of nothing is not zero.
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let p = p.clamp(0.0, 100.0);
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[index])
}

/// The mean of the values that exist.
///
/// `None` when none do. The distinction is the whole function: a
/// non-streamed request has no time-to-first-token, and averaging those
/// in as zero drags the mean toward zero in exact proportion to how many
/// clients did not stream -- which reads as the server getting faster.
pub fn mean_of_present<I: IntoIterator<Item = Option<f64>>>(values: I) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.into_iter().flatten() {
        sum += value;
        count += 1;
    }
    (count > 0).then(|| sum / count as f64)
}

/// A sliding-window rate: tokens per second over the recent past.
///
/// Samples older than the window are dropped on every read, so the rate
/// decays to zero by wall clock when nothing is happening. The
/// denominator is the span from the oldest retained sample to `now`, not
/// the window itself -- see the module doc for why the difference is not
/// cosmetic.
#[derive(Debug, Clone)]
pub struct RateWindow {
    window_ms: u64,
    samples: VecDeque<(u64, u64)>,
}

impl RateWindow {
    /// `window_ms` is clamped to at least 1ms so the span floor is never
    /// larger than the window.
    pub fn new(window_ms: u64) -> Self {
        RateWindow {
            window_ms: window_ms.max(1),
            samples: VecDeque::new(),
        }
    }

    /// Records `tokens` produced at `at_ms`.
    ///
    /// Bounded by [`RATE_SAMPLE_CAPACITY`] independently of the window,
    /// because age-based eviction only happens on a read.
    pub fn record(&mut self, at_ms: u64, tokens: u64) {
        if tokens == 0 {
            return;
        }
        if self.samples.len() == RATE_SAMPLE_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back((at_ms, tokens));
    }

    /// Tokens per second over the window ending at `now_ms`.
    ///
    /// `0.0` when nothing is in the window, which is the true throughput
    /// of an idle server.
    pub fn tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.evict_before(now_ms.saturating_sub(self.window_ms));
        let Some((oldest, _)) = self.samples.front().copied() else {
            return 0.0;
        };
        let tokens: u64 = self.samples.iter().map(|(_, t)| *t).sum();
        // A whole-millisecond clock can report a span of zero for
        // samples that really were separated in time; one millisecond is
        // the smallest span it can honestly claim to have measured.
        let span_ms = now_ms.saturating_sub(oldest).max(1);
        tokens as f64 * 1000.0 / span_ms as f64
    }

    fn evict_before(&mut self, cutoff: u64) {
        while self.samples.front().is_some_and(|(at, _)| *at < cutoff) {
            self.samples.pop_front();
        }
    }
}

/// A value that is only reported once something has said what it is.
///
/// Pool occupancy arrives on replies that carry it and is absent from
/// the ones that do not (a prompt-only reply, a non-hybrid model, a
/// model with no window pool). Last-known-value, and `None` until there
/// is one: reporting `0/0` would claim an empty pool was measured, which
/// a client cannot tell apart from "there is no such pool here".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LastKnown<T: Copy> {
    value: Option<T>,
}

impl<T: Copy> LastKnown<T> {
    pub fn get(&self) -> Option<T> {
        self.value
    }

    pub fn set(&mut self, value: T) {
        self.value = Some(value);
    }
}

/// Live serving counters: what is in flight, what completed, and at what
/// rate.
///
/// The `aborting` set is the part worth reading twice. A request that
/// has been sent an abort stays *active* until its terminal reply
/// actually arrives, and then completes the count as an abort rather
/// than as a completion. Dropping it from the in-flight set at dispatch
/// would let a shutdown declare itself drained while a request was still
/// producing tokens, and counting it as completed would make a
/// timed-out shutdown look like a clean one.
#[derive(Debug, Clone)]
pub struct ServingStats {
    inflight: BTreeSet<u64>,
    aborting: BTreeSet<u64>,
    completed: u64,
    aborted: u64,
    prompt_tokens_total: u64,
    completion_tokens_total: u64,
    decode: RateWindow,
    prefill: RateWindow,
}

impl Default for ServingStats {
    fn default() -> Self {
        Self::new(5_000)
    }
}

impl ServingStats {
    pub fn new(window_ms: u64) -> Self {
        ServingStats {
            inflight: BTreeSet::new(),
            aborting: BTreeSet::new(),
            completed: 0,
            aborted: 0,
            prompt_tokens_total: 0,
            completion_tokens_total: 0,
            decode: RateWindow::new(window_ms),
            prefill: RateWindow::new(window_ms),
        }
    }

    /// One request admitted. Re-admitting an id clears any abort left
    /// over from a previous life of that slot.
    pub fn on_admitted(&mut self, uid: u64) {
        self.inflight.insert(uid);
        self.aborting.remove(&uid);
    }

    /// An abort was dispatched. The request stays active until its
    /// terminal reply arrives.
    pub fn on_abort(&mut self, uid: u64) {
        if self.inflight.contains(&uid) {
            self.aborting.insert(uid);
        }
    }

    pub fn on_prefill(&mut self, at_ms: u64, tokens: u64) {
        self.prompt_tokens_total += tokens;
        self.prefill.record(at_ms, tokens);
    }

    pub fn on_decode(&mut self, at_ms: u64, tokens: u64) {
        self.completion_tokens_total += tokens;
        self.decode.record(at_ms, tokens);
    }

    /// A terminal reply arrived for `uid`.
    pub fn on_finished(&mut self, uid: u64) {
        if !self.inflight.remove(&uid) {
            return;
        }
        if self.aborting.remove(&uid) {
            self.aborted += 1;
        } else {
            self.completed += 1;
        }
    }

    pub fn active(&self) -> usize {
        self.inflight.len()
    }

    /// A stable snapshot of everything still admitted, for a shutdown
    /// that has to abort each one by name.
    pub fn inflight_uids(&self) -> Vec<u64> {
        self.inflight.iter().copied().collect()
    }

    pub fn completed(&self) -> u64 {
        self.completed
    }

    pub fn aborted(&self) -> u64 {
        self.aborted
    }

    pub fn prompt_tokens_total(&self) -> u64 {
        self.prompt_tokens_total
    }

    pub fn completion_tokens_total(&self) -> u64 {
        self.completion_tokens_total
    }

    pub fn decode_tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.decode.tokens_per_second(now_ms)
    }

    pub fn prefill_tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.prefill.tokens_per_second(now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_keeps_the_newest_rows_and_numbers_them_for_all_time() {
        let mut ring = RequestRing::new(3);
        for i in 0..5u32 {
            assert_eq!(ring.push(i), i as u64);
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.rows().copied().collect::<Vec<_>>(), vec![2, 3, 4]);
        assert_eq!(ring.recorded_total(), 5);
    }

    /// The reason the cursor is all-time: a poller that keeps up reads
    /// each row exactly once and never re-reads.
    #[test]
    fn a_poller_that_keeps_up_reads_every_row_exactly_once() {
        let mut ring = RequestRing::new(10);
        let mut cursor = 0;
        let mut seen = Vec::new();
        for round in 0..3u32 {
            for i in 0..2 {
                ring.push(round * 2 + i);
            }
            let page = ring.since(cursor, 100);
            assert_eq!(page.missed, 0);
            seen.extend(page.rows.iter().copied().copied());
            cursor = page.cursor;
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    }

    /// The rule that makes `limit` safe. A truncated page that reported
    /// the all-time count would skip exactly the rows the limit cut off,
    /// which is a pagination bug that only shows up under load.
    #[test]
    fn a_truncated_page_resumes_at_the_row_after_the_last_one_delivered() {
        let mut ring = RequestRing::new(10);
        for i in 0..6u32 {
            ring.push(i);
        }
        let page = ring.since(0, 2);
        assert_eq!(page.rows, [&0, &1]);
        assert_eq!(page.cursor, 2, "not 6");

        let page = ring.since(page.cursor, 2);
        assert_eq!(page.rows, [&2, &3]);
        assert_eq!(page.cursor, 4);

        // The last page returns everything it matched, so it may report
        // the all-time count.
        let page = ring.since(page.cursor, 100);
        assert_eq!(page.rows, [&4, &5]);
        assert_eq!(page.cursor, 6);
    }

    /// And a poller that falls behind has to be able to tell, or its own
    /// numbers quietly stop adding up.
    #[test]
    fn a_poller_that_falls_behind_is_told_how_much_it_lost() {
        let mut ring = RequestRing::new(3);
        for i in 0..10u32 {
            ring.push(i);
        }
        let page = ring.since(0, 100);
        assert_eq!(page.rows.len(), 3);
        assert_eq!(page.missed, 7);
        assert_eq!(page.cursor, 10);

        let page = ring.since(page.cursor, 100);
        assert!(page.rows.is_empty());
        assert_eq!(page.missed, 0);
        assert_eq!(page.cursor, 10);
    }

    /// Nearest-rank always names a real observation, which is the whole
    /// reason it is used here over an interpolating definition.
    #[test]
    fn a_percentile_is_always_a_value_that_was_actually_measured() {
        let values = [10.0, 20.0, 30.0, 40.0];
        for p in [0.0, 25.0, 50.0, 75.0, 95.0, 100.0] {
            let got = percentile(&values, p).expect("non-empty");
            assert!(values.contains(&got), "p{p} produced {got}");
        }
        assert_eq!(percentile(&values, 50.0), Some(20.0));
        assert_eq!(percentile(&values, 95.0), Some(40.0));
        assert_eq!(percentile(&values, 0.0), Some(10.0));
    }

    #[test]
    fn a_percentile_of_nothing_is_none_rather_than_zero() {
        assert_eq!(percentile(&[], 95.0), None);
        assert_eq!(percentile(&[f64::NAN], 95.0), None);
    }

    #[test]
    fn a_mean_ignores_absent_values_instead_of_reading_them_as_zero() {
        assert_eq!(
            mean_of_present([Some(100.0), None, Some(300.0), None]),
            Some(200.0)
        );
        assert_eq!(mean_of_present([None, None]), None);
        assert_eq!(mean_of_present(Vec::<Option<f64>>::new()), None);
    }

    /// The rate has to be right *early*, when only part of the window
    /// has elapsed. Dividing by the window instead of the observed span
    /// reports a fifth of the truth one second into a five-second
    /// window, and a UI reads that as a slow model.
    #[test]
    fn the_rate_is_correct_before_the_window_has_filled() {
        let mut window = RateWindow::new(5_000);
        for step in 0..10u64 {
            window.record(step * 100, 5);
        }
        // 50 tokens over the 1000ms actually observed.
        assert_eq!(window.tokens_per_second(1_000), 50.0);
    }

    /// And the reason it is a window at all: an idle server reports 0,
    /// not the rate it managed while it was busy.
    #[test]
    fn an_idle_server_reports_zero_rather_than_its_busiest_minute() {
        let mut window = RateWindow::new(1_000);
        for step in 0..10u64 {
            window.record(step * 100, 10);
        }
        assert!(window.tokens_per_second(900) > 0.0);
        assert_eq!(window.tokens_per_second(3_600_000), 0.0);
    }

    #[test]
    fn a_rate_window_holds_a_bounded_number_of_samples_even_unpolled() {
        let mut window = RateWindow::new(1_000);
        for step in 0..(RATE_SAMPLE_CAPACITY as u64 + 500) {
            window.record(step, 1);
        }
        assert_eq!(window.samples.len(), RATE_SAMPLE_CAPACITY);
    }

    #[test]
    fn a_request_is_counted_once_it_reaches_a_terminal_reply() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(1);
        stats.on_admitted(2);
        assert_eq!(stats.active(), 2);
        assert_eq!(stats.inflight_uids(), vec![1, 2]);
        stats.on_finished(1);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.active(), 1);
    }

    /// The abort rule, which is what a shutdown's correctness rests on:
    /// dispatching an abort does not make a request inactive, and when
    /// it does end it is not a completion.
    #[test]
    fn an_aborted_request_stays_active_until_its_terminal_reply() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(7);
        stats.on_abort(7);
        assert_eq!(stats.active(), 1, "an abort is dispatched, not applied");
        assert_eq!(stats.completed(), 0);

        // Tokens racing the abort still count toward lifetime totals.
        stats.on_decode(10, 3);
        stats.on_finished(7);
        assert_eq!(stats.active(), 0);
        assert_eq!(stats.completed(), 0);
        assert_eq!(stats.aborted(), 1);
        assert_eq!(stats.completion_tokens_total(), 3);
    }

    #[test]
    fn a_terminal_reply_for_an_unknown_request_changes_nothing() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(1);
        stats.on_finished(1);
        stats.on_finished(1);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.active(), 0);
    }
}
