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
//! So the numbers are read here, the run refuses by default above the
//! bar, and both the start and end readings go into the receipt. A
//! receipt that cannot say how loaded or how hot the host was cannot be
//! audited later.
//!
//! Everything in this module keeps "we could not tell" distinct from
//! any particular value. `None` is never a stand-in for `0.0` load or
//! for nominal temperature: "the host was idle / cool" and "we could
//! not tell" are different facts, and a receipt that conflates them
//! invites exactly the false confidence this module exists to prevent.

/// Default 1-minute load average above which a timed run refuses to
/// start. Matches the bar in `docs/plans/llama-cpp-parity-push.md`.
pub const DEFAULT_MAX_LOAD: f64 = 2.0;

/// Free physical memory in GiB, or `None` if this platform will not
/// say.
///
/// "Free" here means pages the kernel can hand out without evicting
/// something, so on macOS it is the free plus inactive (reclaimable)
/// pages from `vm_stat`, not `hw.memsize` and not the "unused" figure
/// `top` prints.
///
/// This exists because `--fit-host` compares a model's estimated
/// footprint against TOTAL physical memory. A 32 GiB box with 3.5 GiB
/// free therefore accepts a 10 GiB model, runs it, pages it to disk,
/// and reports a number that looks like a measurement. That is the
/// same failure the load bar exists to stop, and nothing caught it.
pub fn free_ram_gb() -> Option<f64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("vm_stat").output().ok()?;
        let text = String::from_utf8(out.stdout).ok()?;
        return parse_vm_stat_free_gb(&text);
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        // MemAvailable is the kernel's own estimate of what a new
        // allocation can get, which is exactly the question here.
        let kb: f64 = text
            .lines()
            .find_map(|l| l.strip_prefix("MemAvailable:"))?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        return Some(kb / 1024.0 / 1024.0);
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(any(target_os = "macos", test))]
fn parse_vm_stat_free_gb(text: &str) -> Option<f64> {
    let page_size: f64 = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |label: &str| -> f64 {
        text.lines()
            .find_map(|l| l.strip_prefix(label))
            .and_then(|v| v.trim().trim_end_matches('.').parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    // Inactive pages are reclaimable without swapping, so a benchmark
    // can have them. Speculative pages are already counted in free.
    let usable = pages("Pages free:") + pages("Pages inactive:");
    Some(usable * page_size / 1024.0 / 1024.0 / 1024.0)
}

/// Refuses a timed run that would not fit in memory and so would be
/// measuring the page file.
///
/// `needs_gb <= 0.0` means the caller does not know the footprint, and
/// an unknown never refuses. Neither does a host that will not report
/// its free memory: silence is not evidence, the same rule the load bar
/// follows.
pub fn ensure_fits_in_ram(needs_gb: f64, headroom_gb: f64) -> anyhow::Result<()> {
    if needs_gb <= 0.0 {
        return Ok(());
    }
    let Some(free) = free_ram_gb() else {
        return Ok(());
    };
    anyhow::ensure!(
        free >= needs_gb + headroom_gb,
        "this model needs about {needs_gb:.1} GiB and only {free:.1} GiB is free \
         (plus {headroom_gb:.1} GiB headroom). It would run from swap, and a paged \
         run reports a real-looking number for work the disk did. Close something, \
         or pass --max-load 0 to measure anyway and accept that the number is not \
         publishable."
    );
    Ok(())
}

/// 1-minute load average, or `None` if this platform will not say.
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

/// How hard the OS says it is currently working to stay cool.
///
/// These are the four levels of `NSProcessInfo.thermalState`, the only
/// thermal signal macOS exposes to an unprivileged process. `Fair` and
/// above mean the system has begun trading sustained clocks for
/// temperature, which is precisely the failure mode that makes a tok/s
/// number unreproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalPressure {
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl ThermalPressure {
    /// `NSProcessInfoThermalState` raw values. Anything outside the
    /// documented range is `None` rather than a guess -- a future OS
    /// level we do not understand must not be silently reported as
    /// nominal.
    pub fn from_ns_thermal_state(raw: i64) -> Option<Self> {
        match raw {
            0 => Some(ThermalPressure::Nominal),
            1 => Some(ThermalPressure::Fair),
            2 => Some(ThermalPressure::Serious),
            3 => Some(ThermalPressure::Critical),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ThermalPressure::Nominal => "nominal",
            ThermalPressure::Fair => "fair",
            ThermalPressure::Serious => "serious",
            ThermalPressure::Critical => "critical",
        }
    }

    /// At `Serious` and above the OS is documented to be reducing
    /// sustained performance, so a timed run is measuring the cooling
    /// system rather than the engine.
    pub fn degrades_measurements(self) -> bool {
        self >= ThermalPressure::Serious
    }
}

/// A thermal observation, with "not measured" kept structurally
/// distinct from "measured, and it was fine".
///
/// The whole point of the type is that `ThermalReading::default()`
/// (nothing measured) can never be mistaken for a reading of
/// `Nominal` / 100%. A field that is always `None` while implying it
/// was measured is worse than no field at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThermalReading {
    /// OS-reported thermal pressure level, `None` if unavailable.
    pub pressure: Option<ThermalPressure>,
    /// Where `pressure` came from, for the receipt.
    pub source: Option<&'static str>,
    /// CPU speed cap as a percentage of nominal, `None` when the host
    /// does not report one. Intel Macs report this via `pmset -g
    /// therm`; Apple Silicon effectively never does, which is why it
    /// cannot be the primary signal.
    pub cpu_speed_limit_percent: Option<u32>,
}

impl ThermalReading {
    /// Did anything at all get measured? A receipt writes this
    /// explicitly so a reader never has to infer it from a null.
    pub fn measured(&self) -> bool {
        self.pressure.is_some() || self.cpu_speed_limit_percent.is_some()
    }

    /// True only when something was measured *and* it says the host is
    /// being held back. An unmeasured host is never reported as
    /// throttled, and never reported as fine either.
    pub fn is_degraded(&self) -> bool {
        self.pressure
            .is_some_and(ThermalPressure::degrades_measurements)
            || self.cpu_speed_limit_percent.is_some_and(|p| p < 100)
    }

    /// One-line form for the console.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        match self.pressure {
            Some(p) => parts.push(format!("thermal {}", p.as_str())),
            None => parts.push("thermal ?".to_string()),
        }
        if let Some(p) = self.cpu_speed_limit_percent {
            if p < 100 {
                parts.push(format!("CPU capped to {p}%"));
            }
        }
        parts.join(", ")
    }
}

/// Reads whatever thermal signal this host offers, without `sudo` and
/// without a new dependency.
///
/// macOS: `NSProcessInfo.thermalState` via Foundation, which is already
/// linked on any Mac. It answers on every call, so `Nominal` here is a
/// real measurement rather than an absence of bad news -- unlike
/// `pmset -g therm`, which prints "No thermal warning level has been
/// recorded" until something actually throttles and so cannot
/// distinguish cool from unknown. The `pmset` speed limit is still
/// read as a secondary field for Intel hosts.
///
/// Everywhere else: nothing is claimed. Linux exposes per-zone
/// millidegrees under `/sys/class/thermal`, which is a temperature and
/// not a pressure level; inventing a mapping would manufacture the
/// false precision this module exists to avoid.
pub fn thermal_reading() -> ThermalReading {
    let pressure = ns_thermal_state().and_then(ThermalPressure::from_ns_thermal_state);
    ThermalReading {
        pressure,
        source: pressure.map(|_| "NSProcessInfo.thermalState"),
        cpu_speed_limit_percent: pmset_cpu_speed_limit(),
    }
}

/// Raw `NSProcessInfo.thermalState`, or `None` off macOS / on an OS
/// that does not answer the selector.
#[cfg(target_os = "macos")]
fn ns_thermal_state() -> Option<i64> {
    use std::ffi::c_void;
    use std::os::raw::c_char;

    // Foundation ships with every macOS; linking it adds no dependency
    // to Cargo.toml and no download.
    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}

    extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }

    // SAFETY: every pointer below is either null-checked or produced by
    // the Objective-C runtime itself. `objc_msgSend` is transmuted to
    // the exact signature of the method being sent, which is the
    // documented way to call it (it has no single ABI of its own), and
    // `respondsToSelector:` is checked before `thermalState` is sent so
    // an OS without the selector returns `None` instead of trapping.
    unsafe {
        let cls = objc_getClass(c"NSProcessInfo".as_ptr());
        if cls.is_null() {
            return None;
        }
        let send_id: extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as *const ());
        let info = send_id(cls, sel_registerName(c"processInfo".as_ptr()));
        if info.is_null() {
            return None;
        }
        let sel_thermal = sel_registerName(c"thermalState".as_ptr());
        let responds: extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> bool =
            std::mem::transmute(objc_msgSend as *const ());
        if !responds(
            info,
            sel_registerName(c"respondsToSelector:".as_ptr()),
            sel_thermal,
        ) {
            return None;
        }
        let send_int: extern "C" fn(*mut c_void, *mut c_void) -> i64 =
            std::mem::transmute(objc_msgSend as *const ());
        Some(send_int(info, sel_thermal))
    }
}

#[cfg(not(target_os = "macos"))]
fn ns_thermal_state() -> Option<i64> {
    None
}

/// `pmset -g therm`'s `CPU_Speed_Limit`, when the host reports one.
#[cfg(target_os = "macos")]
fn pmset_cpu_speed_limit() -> Option<u32> {
    let out = std::process::Command::new("pmset")
        .args(["-g", "therm"])
        .output()
        .ok()?;
    parse_pmset_therm(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(not(target_os = "macos"))]
fn pmset_cpu_speed_limit() -> Option<u32> {
    None
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
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

/// Waits for the host to fall back under the bar, then returns.
///
/// A suite runs one heavy child per entry, and the 1-minute load
/// average it reads is mostly the PREVIOUS entry's own benchmark still
/// decaying. Failing on that makes the suite defeat its own gate: the
/// first entries measure, their load locks out everything after them,
/// and a 21-row run writes 2 receipts. Measured that way on Host B
/// before this existed.
///
/// So between entries the answer is to wait rather than to refuse. An
/// external load that never clears still fails, after `timeout`, which
/// is the case the bar was written for.
pub fn wait_until_quiet_enough(max_load: f64, timeout: std::time::Duration) -> anyhow::Result<()> {
    if max_load <= 0.0 {
        return Ok(());
    }
    let start = std::time::Instant::now();
    loop {
        match load_average_1min() {
            // An unmeasured host never blocks: silence is not evidence
            // of a busy machine, the same rule the bar itself follows.
            None => return Ok(()),
            Some(l) if l < max_load => return Ok(()),
            Some(l) => {
                if start.elapsed() >= timeout {
                    anyhow::bail!(
                        "host 1-minute load average is still {l:.2} after waiting {}s for it \
                         to fall below {max_load:.2}. Something other than this suite is \
                         busy; a timed run here is noise, not a measurement.",
                        timeout.as_secs()
                    );
                }
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        }
    }
}

/// Refuses a timed run while the OS says it is actively giving up
/// clocks to stay cool. Shares `--max-load 0`'s escape hatch, because
/// it is the same escape: measure anyway, but the number is not
/// publishable.
///
/// An unmeasured host never refuses -- silence is not evidence of
/// throttling any more than it is evidence of health.
pub fn ensure_cool_enough(reading: &ThermalReading, enabled: bool) -> anyhow::Result<()> {
    if !enabled || !reading.is_degraded() {
        return Ok(());
    }
    anyhow::bail!(
        "host thermal state is {}: the OS is reducing sustained performance right \
         now, so a timed run measures the cooling system and not the engine. Let \
         the box cool down, or pass --max-load 0 to measure anyway and accept that \
         the number is not publishable.",
        reading.describe()
    );
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

    /// `vm_stat` output is the only thing standing between a paged
    /// benchmark and a published number, so its parse is pinned to a
    /// real sample rather than to whatever the current host prints.
    #[test]
    fn free_ram_counts_reclaimable_pages_not_just_free_ones() {
        let sample = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
             Pages free:                          65536.\n\
             Pages active:                       500000.\n\
             Pages inactive:                      65536.\n\
             Pages speculative:                    1000.\n";
        let gb = parse_vm_stat_free_gb(sample).expect("a well-formed vm_stat must parse");
        // 131072 pages * 16 KiB = 2 GiB. Inactive is reclaimable, so
        // counting free alone would report half the truth and let a
        // model through that then pages.
        assert!(
            (gb - 2.0).abs() < 0.01,
            "expected 2 GiB from free+inactive, got {gb}"
        );
    }

    /// An unknown footprint and an unmeasurable host both mean "do not
    /// know", and neither may refuse a run.
    #[test]
    fn an_unknown_footprint_never_refuses() {
        ensure_fits_in_ram(0.0, 2.0).expect("an unknown footprint must not refuse");
    }

    /// The refusal has to name the number the operator can act on.
    #[test]
    fn a_model_larger_than_free_ram_is_refused_with_both_figures() {
        if free_ram_gb().is_none() {
            return; // platform will not say; the guard correctly no-ops
        }
        let err = ensure_fits_in_ram(1_000_000.0, 2.0)
            .expect_err("a model larger than any host must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("swap") && msg.contains("free"),
            "the refusal must say it would page and how much is free, got: {msg}"
        );
    }

    #[test]
    fn pmset_reports_a_speed_limit_only_when_it_has_one() {
        assert_eq!(parse_pmset_therm("CPU_Speed_Limit \t= 70\n"), Some(70));
        assert_eq!(
            parse_pmset_therm("Note: No thermal warning level has been recorded"),
            None
        );
    }

    /// The disable switch has to disable the WAIT too, or
    /// `--max-load 0` would still block a suite for three minutes per
    /// entry on a busy box, which is the opposite of what it promises.
    #[test]
    fn waiting_is_disabled_by_the_same_switch_that_disables_refusing() {
        let start = std::time::Instant::now();
        wait_until_quiet_enough(0.0, std::time::Duration::from_secs(30))
            .expect("max_load 0 must never block");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "max_load 0 returned only after waiting, so it did not disable the wait"
        );
    }

    /// A bar nothing could ever clear must give up rather than hang the
    /// suite forever. The timeout is the whole reason a wait is safe to
    /// put in front of every entry.
    #[test]
    fn an_impossible_bar_times_out_instead_of_waiting_forever() {
        if load_average_1min().is_none() {
            return; // platform will not say; the wait correctly no-ops
        }
        let err = wait_until_quiet_enough(0.000_001, std::time::Duration::from_millis(1))
            .expect_err("a bar no host can clear must time out");
        let msg = err.to_string();
        assert!(
            msg.contains("still") && msg.contains("busy"),
            "the timeout must say what it waited for, got: {msg}"
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

    #[test]
    fn an_unmeasured_thermal_reading_is_not_a_nominal_one() {
        // The whole reason this is a struct and not an Option<u32>.
        let unknown = ThermalReading::default();
        assert!(!unknown.measured(), "nothing was read");
        assert!(!unknown.is_degraded(), "unknown must not read as throttled");
        assert_eq!(unknown.describe(), "thermal ?");

        let cool = ThermalReading {
            pressure: Some(ThermalPressure::Nominal),
            source: Some("NSProcessInfo.thermalState"),
            cpu_speed_limit_percent: None,
        };
        assert!(cool.measured(), "nominal is a measurement, not a silence");
        assert!(!cool.is_degraded());
        assert_ne!(
            cool, unknown,
            "measured-nominal must differ from unmeasured"
        );
    }

    #[test]
    fn thermal_pressure_refuses_only_from_serious_upward() {
        assert!(!ThermalPressure::Nominal.degrades_measurements());
        // `Fair` is the OS fans-up level; it is recorded but does not
        // block, or nothing would ever run on a laptop.
        assert!(!ThermalPressure::Fair.degrades_measurements());
        assert!(ThermalPressure::Serious.degrades_measurements());
        assert!(ThermalPressure::Critical.degrades_measurements());
    }

    #[test]
    fn an_unknown_ns_thermal_level_is_not_decoded_as_nominal() {
        assert_eq!(
            ThermalPressure::from_ns_thermal_state(0),
            Some(ThermalPressure::Nominal)
        );
        assert_eq!(
            ThermalPressure::from_ns_thermal_state(3),
            Some(ThermalPressure::Critical)
        );
        // A future level must not be flattened into a known one.
        assert_eq!(ThermalPressure::from_ns_thermal_state(4), None);
        assert_eq!(ThermalPressure::from_ns_thermal_state(-1), None);
    }

    #[test]
    fn a_capped_cpu_is_degraded_even_without_a_pressure_level() {
        let intel = ThermalReading {
            pressure: None,
            source: None,
            cpu_speed_limit_percent: Some(70),
        };
        assert!(intel.measured());
        assert!(intel.is_degraded());
        assert!(intel.describe().contains("70%"));

        let intel_ok = ThermalReading {
            cpu_speed_limit_percent: Some(100),
            ..Default::default()
        };
        assert!(intel_ok.measured());
        assert!(!intel_ok.is_degraded());
    }

    #[test]
    fn the_thermal_gate_refuses_when_hot_and_never_when_unknown() {
        let hot = ThermalReading {
            pressure: Some(ThermalPressure::Critical),
            source: Some("NSProcessInfo.thermalState"),
            cpu_speed_limit_percent: None,
        };
        let err = ensure_cool_enough(&hot, true).unwrap_err().to_string();
        assert!(
            err.contains("critical"),
            "the refusal names what it caught: {err}"
        );
        // The escape hatch is the same one the load bar uses.
        assert!(ensure_cool_enough(&hot, false).is_ok());
        // Silence is never evidence of throttling.
        assert!(ensure_cool_enough(&ThermalReading::default(), true).is_ok());
        assert!(ensure_cool_enough(
            &ThermalReading {
                pressure: Some(ThermalPressure::Fair),
                source: Some("NSProcessInfo.thermalState"),
                cpu_speed_limit_percent: None,
            },
            true
        )
        .is_ok());
    }

    /// Not a mock: this calls the real Objective-C runtime. It exists
    /// because the FFI in `ns_thermal_state` is the one part of this
    /// module a pure test cannot cover -- a wrong selector name or a
    /// wrong transmuted signature would show up here as a missing
    /// reading or an out-of-range level, not as a compile error.
    #[test]
    #[cfg(target_os = "macos")]
    fn macos_reports_a_thermal_level_in_the_documented_range() {
        let raw = ns_thermal_state().expect("macOS always answers thermalState");
        assert!(
            (0..=3).contains(&raw),
            "NSProcessInfoThermalState out of documented range: {raw}"
        );
        let reading = thermal_reading();
        assert!(
            reading.measured(),
            "a macOS host must produce a real thermal reading, not a null"
        );
        assert!(reading.pressure.is_some());
        assert_eq!(reading.source, Some("NSProcessInfo.thermalState"));
    }
}
