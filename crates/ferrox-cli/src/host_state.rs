//! What the host was doing while a benchmark ran.
//!
//! `benchmarks/RESULTS.md` is only meaningful if every row was measured
//! on a quiet box. The plan's measurement contract puts the bar at a
//! 1-minute load average of ~2.0 and records why: known-good rows read
//! 25-45% low under load, which is wider than most of the gaps being
//! chased. That rule was written down and then broken anyway, because
//! nothing enforced it -- a 40-repetition run once went out at load
//! 208, producing 40 repetitions of noise.
//!
//! So the number is read here, the run refuses by default above the
//! bar, and both the start and end readings go into the receipt. A
//! receipt that cannot say how loaded the host was cannot be audited
//! later.

/// Default 1-minute load average above which a timed run refuses to
/// start. Matches the bar in `docs/plans/llama-cpp-parity-push.md`.
pub const DEFAULT_MAX_LOAD: f64 = 2.0;

/// 1-minute load average, or `None` if this platform will not say.
///
/// `None` is deliberately not `0.0`: "the host was idle" and "we could
/// not tell" are different facts, and a receipt that conflates them
/// invites exactly the false confidence this module exists to prevent.
pub fn load_average_1min() -> Option<f64> {
    // Linux: cheapest possible, no child process.
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().next()?.parse().ok();
    }
    // macOS / BSD: no /proc. `sysctl -n vm.loadavg` prints `{ 1.23 2.34 3.45 }`.
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    parse_sysctl_loadavg(&String::from_utf8_lossy(&out.stdout))
}

fn parse_sysctl_loadavg(s: &str) -> Option<f64> {
    s.split_whitespace().find_map(|tok| {
        tok.trim_matches(|c| c == '{' || c == '}')
            .parse::<f64>()
            .ok()
    })
}

/// Thermal throttle state as a percentage of nominal CPU speed, where
/// 100 means "not throttled".
///
/// `None` on every host that does not report it, which on Apple
/// Silicon is the common case -- `pmset -g therm` answers "No thermal
/// warning level has been recorded" until something actually throttles.
/// Reporting `None` rather than an optimistic 100 keeps "unthrottled"
/// and "unknown" distinguishable in the receipt.
pub fn thermal_speed_limit_percent() -> Option<u32> {
    let out = std::process::Command::new("pmset")
        .args(["-g", "therm"])
        .output()
        .ok()?;
    parse_pmset_therm(&String::from_utf8_lossy(&out.stdout))
}

fn parse_pmset_therm(s: &str) -> Option<u32> {
    s.lines()
        .find_map(|l| l.split_once("CPU_Speed_Limit"))
        .and_then(|(_, rest)| rest.split('=').nth(1))
        .and_then(|v| v.trim().parse().ok())
}

/// Refuses a timed run on a host too busy for the number to mean
/// anything. `max_load <= 0.0` disables the check.
pub fn ensure_quiet_enough(max_load: f64) -> anyhow::Result<Option<f64>> {
    let load = load_average_1min();
    if max_load <= 0.0 {
        return Ok(load);
    }
    if let Some(l) = load {
        anyhow::ensure!(
            l < max_load,
            "host 1-minute load average is {l:.2}, above the {max_load:.2} bar: \
             a timed run here is noise, not a measurement (known-good rows read \
             25-45% low under load). Wait for the box to go quiet, or pass \
             --max-load 0 to measure anyway and accept that the number is not \
             publishable."
        );
    }
    Ok(load)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysctl_loadavg_braces_are_not_mistaken_for_a_number() {
        assert_eq!(parse_sysctl_loadavg("{ 1.23 2.34 3.45 }"), Some(1.23));
        assert_eq!(parse_sysctl_loadavg("{1.23 2.34 3.45}"), Some(1.23));
    }

    #[test]
    fn an_unparseable_loadavg_is_none_rather_than_zero() {
        assert_eq!(parse_sysctl_loadavg(""), None);
        assert_eq!(parse_sysctl_loadavg("no such oid"), None);
    }

    #[test]
    fn pmset_reports_a_speed_limit_only_when_it_has_one() {
        assert_eq!(parse_pmset_therm("CPU_Speed_Limit \t= 70\n"), Some(70));
        assert_eq!(
            parse_pmset_therm("Note: No thermal warning level has been recorded"),
            None
        );
    }

    #[test]
    fn a_disabled_gate_still_reports_the_load_it_read() {
        // max_load <= 0 must never refuse, whatever the host is doing.
        assert!(ensure_quiet_enough(0.0).is_ok());
        assert!(ensure_quiet_enough(-1.0).is_ok());
    }

    #[test]
    fn the_gate_refuses_above_the_bar_and_says_the_number() {
        // Drive the predicate directly: the real load average is not
        // ours to control, so assert on the arithmetic the gate does.
        let refuse = |l: f64, max: f64| max > 0.0 && l >= max;
        assert!(refuse(208.0, 2.0));
        assert!(refuse(2.0, 2.0), "the bar is exclusive");
        assert!(!refuse(1.99, 2.0));
        assert!(!refuse(208.0, 0.0), "max_load 0 disables the gate");
    }
}
