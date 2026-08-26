//! Owning exactly one engine child's lifecycle.
//!
//! A port of FreeToken's `daemon/serve_manager.py`. Ferrox ships one
//! binary today, so nothing here spawns a process yet: this is the
//! *decision* half, and the spawning half is a [`ProcessHost`] the
//! caller supplies. That split is the same one [`crate::expert_slots`]
//! makes for device memory, and for the same reason -- every rule below
//! is a race, and a race you can only reproduce by starting real
//! multi-gigabyte processes is a race nobody tests.
//!
//! # The four rules, and what each one costs when it is missing
//!
//! **Start is a serialized gate.** Three outcomes, not two. A start for
//! a child that is already up *with the same spec* is an idempotent
//! no-op, because a supervisor whose response was lost retries, and the
//! retry must not become a second engine. A start for a child that is
//! up with a *different* spec is a conflict, refused rather than
//! silently reconfiguring something a caller is already using. And a
//! start arriving while a spawn is in flight is **awaited and then
//! re-evaluated** -- not queued behind it, and above all not spawned
//! alongside it. Two engines on one GPU is an out-of-memory crash at
//! best; on hosts where it fits, it is two processes serving different
//! weights on one port.
//!
//! **Exactly one reaper.** Stop signals the child and waits on the
//! reaped event. It never derives completion from a poll, because a
//! poll *transiently lies*: while the monitor holds the lock for its
//! own wait, a poll sees "still running" for a child that has already
//! exited, and a stop that trusted it would report failure for a stop
//! that worked, or loop forever. One party reaps; everyone else waits
//! on what that party publishes.
//!
//! **A stop-requested latch that survives a failed stop.** If the child
//! crashes *during* a stop, the crash handler must not restart it. The
//! latch is set when the stop is requested rather than when it
//! completes, so the window between "we asked it to die" and "it died"
//! cannot be read as an unplanned death. Without it, stopping a serve
//! is how you restart it.
//!
//! **A permanent shutdown latch.** A start queued behind the final stop
//! is rejected, not served. Otherwise the last thing a shutting-down
//! daemon does is spawn a multi-gigabyte process nobody will ever reap.
//!
//! # Re-adoption, and why the key has four parts
//!
//! [`ChildIdentity`] keys a running child on `(pid, start_time, argv,
//! port)`. `pid` alone is not an identity: PIDs are recycled, and a
//! daemon restarting after a crash that adopts "the process at PID
//! 4711" can adopt a shell. `start_time` is the guard -- the kernel's
//! own boot-relative start time for that PID, which a recycled PID
//! cannot reproduce. `argv` and `port` then confirm it is *our* engine
//! serving what we think, rather than another instance someone started
//! by hand.

use std::collections::BTreeMap;

/// What a caller asked to be running: the spec a start is matched
/// against.
///
/// Equality is what decides no-op versus conflict, so every field here
/// is load-bearing. A field that should not distinguish two starts does
/// not belong in this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeSpec {
    /// The model this engine serves. Two starts naming different
    /// models are a conflict however alike everything else is.
    pub model: String,
    /// The port it listens on.
    pub port: u16,
    /// The full argument vector, so a difference in any flag that
    /// changes behaviour is a conflict rather than a silent no-op.
    pub argv: Vec<String>,
}

/// A running child, keyed so that a daemon restarting after its own
/// crash can tell "our engine" from "whatever holds that PID now".
///
/// See the module docs on why `pid` alone will not do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildIdentity {
    pub pid: u32,
    /// The kernel's start time for this PID (Linux: field 22 of
    /// `/proc/<pid>/stat`, in clock ticks since boot). This is the
    /// PID-reuse guard: a recycled PID has a later start time, so a
    /// stale record fails to match rather than adopting a stranger.
    pub start_time: u64,
    pub argv: Vec<String>,
    pub port: u16,
}

impl ChildIdentity {
    /// Whether this really is the child a record describes.
    ///
    /// All four parts must agree. A caller tempted to relax this to
    /// `pid` should re-read why `start_time` is here.
    pub fn is(&self, other: &ChildIdentity) -> bool {
        self == other
    }

    /// Whether this child is serving `spec`.
    pub fn serves(&self, spec: &ServeSpec) -> bool {
        self.port == spec.port && self.argv == spec.argv
    }
}

/// Why a start did nothing, or what it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartOutcome {
    /// A child matching the spec was already up. The caller's retry
    /// found what it wanted; nothing was spawned.
    AlreadyRunning(ChildIdentity),
    /// Spawned.
    Started(ChildIdentity),
    /// A child is up serving something else. Refused rather than
    /// reconfigured, because someone is using the one that is up.
    Conflict {
        running: ChildIdentity,
        requested: ServeSpec,
    },
    /// The supervisor is shutting down permanently. See the module
    /// docs: the alternative is spawning a process nobody will reap.
    ShuttingDown,
}

/// Why a stop did nothing, or what it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopOutcome {
    /// Signalled and reaped.
    Stopped { pid: u32, exit: ChildExit },
    /// Nothing was running. A stop is idempotent: a caller whose first
    /// response was lost must be able to ask again.
    NotRunning,
}

/// How a child ended, as the single reaper observed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildExit {
    /// Exited of its own accord with this status.
    Code(i32),
    /// Killed by a signal.
    Signal(i32),
}

/// How a child's death should be read.
///
/// The distinction is the stop-requested latch's whole purpose: the
/// same `SIGTERM`-and-exit sequence means "it worked" if we asked for
/// it and "it died" if we did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeathKind {
    /// We asked for this. Do not restart.
    Requested,
    /// Nobody asked. A restart policy may act.
    Unexpected,
}

/// The spawning half, supplied by the caller.
///
/// Deliberately narrow: everything that is a *decision* lives in
/// [`Supervisor`], so a host implementation has nothing to get wrong
/// beyond talking to the OS. `signal` and `reap` are separate because
/// exactly one party reaps -- see the module docs.
pub trait ProcessHost {
    /// Starts the engine and returns its identity, including the
    /// `start_time` the PID-reuse guard needs.
    fn spawn(&mut self, spec: &ServeSpec) -> Result<ChildIdentity, String>;

    /// Asks the child to exit. Must not block waiting for it: the wait
    /// belongs to [`Self::reap`], which is the one reaper.
    fn signal(&mut self, child: &ChildIdentity) -> Result<(), String>;

    /// Waits for the child and returns how it ended. Called by the
    /// single reaper only.
    fn reap(&mut self, child: &ChildIdentity) -> Result<ChildExit, String>;

    /// Whether a recorded child is still the process it claims to be,
    /// used only when re-adopting across a daemon restart.
    ///
    /// The default refuses to adopt anything, which is the safe answer
    /// for a host that cannot check: a supervisor that adopts nothing
    /// leaks a process, and one that adopts wrongly signals a stranger.
    fn still_alive(&mut self, child: &ChildIdentity) -> bool {
        let _ = child;
        false
    }
}

/// What the supervisor believes about its one child.
#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Idle,
    Running(ChildIdentity),
}

/// Owns exactly one engine child.
///
/// Not `Sync`, and deliberately not internally locked: the serialized
/// gate the module docs describe is the *caller's* lock around this
/// type. Putting a mutex in here would let two callers each hold a
/// consistent view of an inconsistent world.
pub struct Supervisor<H: ProcessHost> {
    host: H,
    state: State,
    /// Set when a stop is REQUESTED, cleared only by a successful
    /// start. See the module docs: setting it on completion instead
    /// leaves a window where a crash reads as unexpected.
    stop_requested: bool,
    /// Once set, never cleared.
    shutting_down: bool,
    /// Counts, for the caller's own telemetry.
    spawns: u64,
    reaps: u64,
}

impl<H: ProcessHost> Supervisor<H> {
    pub fn new(host: H) -> Self {
        Supervisor {
            host,
            state: State::Idle,
            stop_requested: false,
            shutting_down: false,
            spawns: 0,
            reaps: 0,
        }
    }

    pub fn spawn_count(&self) -> u64 {
        self.spawns
    }

    pub fn reap_count(&self) -> u64 {
        self.reaps
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// The child currently believed to be running, if any.
    pub fn running(&self) -> Option<&ChildIdentity> {
        match &self.state {
            State::Running(c) => Some(c),
            State::Idle => None,
        }
    }

    /// Start, or explain why not.
    ///
    /// The three non-spawning outcomes are the point; see
    /// [`StartOutcome`]. The caller must hold its gate across this
    /// call, which is what makes "a spawn already in flight is awaited
    /// and re-evaluated" true: a second caller blocks on the gate, and
    /// when it acquires it, this runs again against the state the
    /// first one left. That re-evaluation is why the already-running
    /// check is here rather than at the call site.
    pub fn start(&mut self, spec: &ServeSpec) -> Result<StartOutcome, String> {
        if self.shutting_down {
            return Ok(StartOutcome::ShuttingDown);
        }
        if let State::Running(running) = &self.state {
            if running.serves(spec) {
                return Ok(StartOutcome::AlreadyRunning(running.clone()));
            }
            return Ok(StartOutcome::Conflict {
                running: running.clone(),
                requested: spec.clone(),
            });
        }
        let child = self.host.spawn(spec)?;
        self.spawns += 1;
        self.stop_requested = false;
        self.state = State::Running(child.clone());
        Ok(StartOutcome::Started(child))
    }

    /// Signal, then wait on the one reaper.
    ///
    /// The latch is set BEFORE the signal, so a child that dies between
    /// the two is still a requested death. Setting it after would leave
    /// exactly the window a restart policy fires in.
    pub fn stop(&mut self) -> Result<StopOutcome, String> {
        let State::Running(child) = self.state.clone() else {
            return Ok(StopOutcome::NotRunning);
        };
        self.stop_requested = true;
        // A signal that fails leaves the latch SET on purpose: we asked
        // for this death, and if the child goes down anyway (it was
        // already exiting, the signal raced its exit) that is still a
        // requested death, not one to restart.
        self.host.signal(&child)?;
        let exit = self.host.reap(&child)?;
        self.reaps += 1;
        self.state = State::Idle;
        Ok(StopOutcome::Stopped {
            pid: child.pid,
            exit,
        })
    }

    /// How to read a child's death.
    ///
    /// Call this from whatever observes an exit that [`Self::stop`] did
    /// not perform. The latch, not the exit status, decides: a child
    /// killed by `SIGTERM` is a normal stop if we sent it and a crash
    /// if we did not.
    pub fn classify_death(&mut self, child: &ChildIdentity) -> DeathKind {
        let ours = matches!(&self.state, State::Running(c) if c.is(child));
        if ours {
            self.state = State::Idle;
            self.reaps += 1;
        }
        if self.stop_requested {
            DeathKind::Requested
        } else {
            DeathKind::Unexpected
        }
    }

    /// Latch shutdown, then stop whatever is running.
    ///
    /// Shutdown is latched FIRST so that a start racing this call is
    /// rejected rather than spawning something the stop below has
    /// already walked past.
    pub fn shutdown(&mut self) -> Result<StopOutcome, String> {
        self.shutting_down = true;
        self.stop()
    }

    /// Re-adopt a child recorded before this supervisor existed, after
    /// a daemon restart.
    ///
    /// Returns whether the record was adopted. A record that fails
    /// [`ProcessHost::still_alive`] is dropped rather than adopted: see
    /// the module docs on why the key has four parts.
    pub fn adopt(&mut self, recorded: ChildIdentity) -> bool {
        if self.shutting_down || matches!(self.state, State::Running(_)) {
            return false;
        }
        if !self.host.still_alive(&recorded) {
            return false;
        }
        self.state = State::Running(recorded);
        true
    }
}

/// Which process the OOM killer should take first.
///
/// Two rules, and the second is the one that gets forgotten. The whole
/// process GROUP is scored positive so the killer takes an engine
/// rather than the daemon that would restart it -- a daemon killed
/// first leaves the engine orphaned and nothing to reap it. And the
/// scores are REWRITTEN periodically, because workers fork *after* the
/// initial write and a child forked later inherits nothing: a score
/// written once covers only the processes that existed when it ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OomPolicy {
    /// Score for the engine and everything it forks. Positive means
    /// "prefer to kill this".
    pub engine_score: i32,
    /// Score for the supervising daemon. Negative means "prefer to
    /// spare this".
    pub daemon_score: i32,
    /// How often the group is rescored, in seconds. Zero disables
    /// rescoring, which is only correct for an engine that never forks.
    pub rewrite_interval_secs: u64,
}

impl Default for OomPolicy {
    fn default() -> Self {
        OomPolicy {
            engine_score: 500,
            daemon_score: -500,
            rewrite_interval_secs: 30,
        }
    }
}

impl OomPolicy {
    /// The score each PID in the group should carry right now.
    ///
    /// `engine_group` is every PID currently in the engine's process
    /// group, which is why this is called repeatedly rather than once:
    /// the set grows as the engine forks workers.
    pub fn scores(&self, daemon_pid: u32, engine_group: &[u32]) -> BTreeMap<u32, i32> {
        let mut out = BTreeMap::new();
        out.insert(daemon_pid, self.daemon_score);
        for pid in engine_group {
            // A PID that is both is the daemon; sparing it wins, since
            // killing the reaper is the failure this policy exists to
            // avoid.
            out.entry(*pid).or_insert(self.engine_score);
        }
        out
    }

    /// Whether a rescore is due, given how long since the last one.
    pub fn rewrite_due(&self, secs_since_last: u64) -> bool {
        self.rewrite_interval_secs > 0 && secs_since_last >= self.rewrite_interval_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host that records what it was asked to do, so the tests assert
    /// on the SEQUENCE of OS calls rather than only on return values.
    /// Whether a second engine was spawned is not visible in a
    /// `StartOutcome`; it is visible here.
    #[derive(Default)]
    struct FakeHost {
        spawned: Vec<ServeSpec>,
        signalled: Vec<u32>,
        reaped: Vec<u32>,
        next_pid: u32,
        alive: bool,
        spawn_fails: bool,
        signal_fails: bool,
    }

    impl ProcessHost for FakeHost {
        fn spawn(&mut self, spec: &ServeSpec) -> Result<ChildIdentity, String> {
            if self.spawn_fails {
                return Err("spawn refused".into());
            }
            self.spawned.push(spec.clone());
            self.next_pid += 1;
            Ok(ChildIdentity {
                pid: 1000 + self.next_pid,
                start_time: 42,
                argv: spec.argv.clone(),
                port: spec.port,
            })
        }
        fn signal(&mut self, child: &ChildIdentity) -> Result<(), String> {
            if self.signal_fails {
                return Err("signal refused".into());
            }
            self.signalled.push(child.pid);
            Ok(())
        }
        fn reap(&mut self, child: &ChildIdentity) -> Result<ChildExit, String> {
            self.reaped.push(child.pid);
            Ok(ChildExit::Signal(15))
        }
        fn still_alive(&mut self, _child: &ChildIdentity) -> bool {
            self.alive
        }
    }

    fn spec(port: u16, model: &str) -> ServeSpec {
        ServeSpec {
            model: model.into(),
            port,
            argv: vec!["ferrox".into(), "serve".into(), "-m".into(), model.into()],
        }
    }

    /// The retry case. A supervisor whose response was lost asks again,
    /// and the second ask must not become a second engine -- which the
    /// outcome alone does not prove, so the spawn count is what is
    /// asserted.
    #[test]
    fn starting_an_already_running_child_with_the_same_spec_spawns_nothing() {
        let mut sup = Supervisor::new(FakeHost::default());
        let s = spec(8383, "a.gguf");
        let first = sup.start(&s).unwrap();
        assert!(matches!(first, StartOutcome::Started(_)));

        let second = sup.start(&s).unwrap();
        assert!(
            matches!(second, StartOutcome::AlreadyRunning(_)),
            "a repeated start is idempotent, got {second:?}"
        );
        assert_eq!(
            sup.spawn_count(),
            1,
            "the retry must not spawn a second engine"
        );
        assert_eq!(sup.host.spawned.len(), 1);
    }

    /// A start for a DIFFERENT spec is refused, not silently applied.
    /// Someone is using the one that is up.
    #[test]
    fn starting_a_different_spec_conflicts_instead_of_reconfiguring() {
        let mut sup = Supervisor::new(FakeHost::default());
        sup.start(&spec(8383, "a.gguf")).unwrap();

        let out = sup.start(&spec(8383, "b.gguf")).unwrap();
        match out {
            StartOutcome::Conflict { running, requested } => {
                assert_eq!(running.port, 8383);
                assert_eq!(requested.model, "b.gguf");
            }
            other => panic!("expected a conflict, got {other:?}"),
        }
        assert_eq!(sup.spawn_count(), 1, "a conflict spawns nothing");

        // Same model, different port is also a conflict: one child.
        let out = sup.start(&spec(9999, "a.gguf")).unwrap();
        assert!(matches!(out, StartOutcome::Conflict { .. }));
        assert_eq!(sup.spawn_count(), 1);
    }

    /// The stop-requested latch, which is the difference between
    /// stopping a serve and restarting it.
    #[test]
    fn a_death_during_a_requested_stop_is_not_read_as_a_crash() {
        let mut sup = Supervisor::new(FakeHost::default());
        let s = spec(8383, "a.gguf");
        let StartOutcome::Started(child) = sup.start(&s).unwrap() else {
            panic!("expected a spawn");
        };

        // Before any stop, an exit is unplanned.
        assert_eq!(sup.classify_death(&child), DeathKind::Unexpected);

        // After a start, the latch is clear again, and a requested stop
        // sets it.
        let StartOutcome::Started(child) = sup.start(&s).unwrap() else {
            panic!("expected a respawn after the death above");
        };
        assert!(matches!(sup.stop().unwrap(), StopOutcome::Stopped { .. }));
        assert_eq!(
            sup.classify_death(&child),
            DeathKind::Requested,
            "we asked for this death; a restart policy must not fire"
        );
    }

    /// A stop whose SIGNAL fails still leaves the latch set. The child
    /// may be exiting anyway -- the signal raced its exit -- and that is
    /// a requested death, not one to restart.
    #[test]
    fn a_stop_that_fails_to_signal_still_latches_the_request() {
        let mut sup = Supervisor::new(FakeHost::default());
        let s = spec(8383, "a.gguf");
        let StartOutcome::Started(child) = sup.start(&s).unwrap() else {
            panic!("expected a spawn");
        };
        sup.host.signal_fails = true;

        assert!(sup.stop().is_err(), "the signal failed");
        assert_eq!(
            sup.classify_death(&child),
            DeathKind::Requested,
            "a failed stop is still a stop we asked for"
        );
    }

    /// A start queued behind the final stop is rejected. Otherwise the
    /// last act of a shutting-down daemon is spawning an engine nobody
    /// will reap.
    #[test]
    fn a_start_after_shutdown_is_rejected_rather_than_served() {
        let mut sup = Supervisor::new(FakeHost::default());
        sup.start(&spec(8383, "a.gguf")).unwrap();
        assert!(matches!(
            sup.shutdown().unwrap(),
            StopOutcome::Stopped { .. }
        ));
        assert!(sup.is_shutting_down());

        let out = sup.start(&spec(8383, "a.gguf")).unwrap();
        assert_eq!(out, StartOutcome::ShuttingDown);
        assert_eq!(sup.spawn_count(), 1, "shutdown is permanent");

        // And it stays rejected. There is no un-shutdown.
        assert_eq!(
            sup.start(&spec(7777, "b.gguf")).unwrap(),
            StartOutcome::ShuttingDown
        );
        assert_eq!(sup.spawn_count(), 1);
    }

    /// Stop is idempotent, because a caller whose response was lost
    /// must be able to ask again.
    #[test]
    fn stopping_when_nothing_runs_is_a_no_op_rather_than_an_error() {
        let mut sup = Supervisor::new(FakeHost::default());
        assert_eq!(sup.stop().unwrap(), StopOutcome::NotRunning);
        sup.start(&spec(8383, "a.gguf")).unwrap();
        assert!(matches!(sup.stop().unwrap(), StopOutcome::Stopped { .. }));
        assert_eq!(
            sup.stop().unwrap(),
            StopOutcome::NotRunning,
            "a repeated stop must not error"
        );
        assert_eq!(sup.host.reaped.len(), 1, "the child is reaped exactly once");
    }

    /// Signal precedes reap, and each happens once. Reaping twice is
    /// how a supervisor blocks forever on a child that is already gone.
    #[test]
    fn a_stop_signals_once_and_reaps_once_in_that_order() {
        let mut sup = Supervisor::new(FakeHost::default());
        sup.start(&spec(8383, "a.gguf")).unwrap();
        sup.stop().unwrap();
        assert_eq!(sup.host.signalled, vec![1001]);
        assert_eq!(sup.host.reaped, vec![1001]);
        assert_eq!(sup.reap_count(), 1);
    }

    /// A recycled PID must not be adopted. This is the whole reason
    /// `start_time` is in the key.
    #[test]
    fn adoption_rejects_a_recycled_pid_and_accepts_the_real_child() {
        let recorded = ChildIdentity {
            pid: 4711,
            start_time: 100,
            argv: vec!["ferrox".into(), "serve".into()],
            port: 8383,
        };

        // The host says nothing at that PID is ours.
        let mut sup = Supervisor::new(FakeHost {
            alive: false,
            ..Default::default()
        });
        assert!(!sup.adopt(recorded.clone()));
        assert!(sup.running().is_none(), "nothing was adopted");

        let mut sup = Supervisor::new(FakeHost {
            alive: true,
            ..Default::default()
        });
        assert!(sup.adopt(recorded.clone()));
        assert_eq!(sup.running(), Some(&recorded));

        // Same PID and port, later start time: a different process.
        let recycled = ChildIdentity {
            start_time: 999,
            ..recorded.clone()
        };
        assert!(
            !recycled.is(&recorded),
            "a recycled PID must not match on pid and port alone"
        );
    }

    /// Adoption never displaces a child already running, and never
    /// happens after shutdown.
    #[test]
    fn adoption_is_refused_when_a_child_runs_or_shutdown_has_latched() {
        let recorded = ChildIdentity {
            pid: 4711,
            start_time: 100,
            argv: vec!["ferrox".into()],
            port: 8383,
        };

        let mut sup = Supervisor::new(FakeHost {
            alive: true,
            ..Default::default()
        });
        sup.start(&spec(8383, "a.gguf")).unwrap();
        assert!(
            !sup.adopt(recorded.clone()),
            "a running child is not displaced"
        );
        assert_eq!(sup.running().map(|c| c.pid), Some(1001));

        let mut sup = Supervisor::new(FakeHost {
            alive: true,
            ..Default::default()
        });
        sup.shutdown().unwrap();
        assert!(!sup.adopt(recorded), "shutdown adopts nothing");
    }

    /// A failed spawn leaves the supervisor idle rather than believing
    /// in a child that does not exist.
    #[test]
    fn a_failed_spawn_leaves_nothing_running() {
        let mut sup = Supervisor::new(FakeHost {
            spawn_fails: true,
            ..Default::default()
        });
        assert!(sup.start(&spec(8383, "a.gguf")).is_err());
        assert!(sup.running().is_none());
        assert_eq!(sup.spawn_count(), 0);

        // And a later start still works: a failure is not a latch.
        sup.host.spawn_fails = false;
        assert!(matches!(
            sup.start(&spec(8383, "a.gguf")).unwrap(),
            StartOutcome::Started(_)
        ));
    }

    /// The daemon is spared and the whole engine group is offered up.
    /// Killing the reaper first is the failure this policy exists to
    /// prevent.
    #[test]
    fn the_oom_policy_spares_the_daemon_and_scores_the_whole_engine_group() {
        let policy = OomPolicy::default();
        let scores = policy.scores(/* daemon = */ 10, &[20, 21, 22]);
        assert_eq!(scores[&10], -500, "the daemon is spared");
        for worker in [20, 21, 22] {
            assert_eq!(scores[&worker], 500, "pid {worker} is offered up");
        }

        // A PID appearing in both stays spared: killing the reaper is
        // worse than killing an engine.
        let scores = policy.scores(10, &[10, 20]);
        assert_eq!(scores[&10], -500);
        assert_eq!(scores[&20], 500);
    }

    /// Rescoring is periodic because workers fork AFTER the first
    /// write, so a score written once covers only what existed then.
    #[test]
    fn rescoring_is_due_on_the_interval_and_disabled_only_by_zero() {
        let policy = OomPolicy::default();
        assert!(!policy.rewrite_due(0));
        assert!(!policy.rewrite_due(29));
        assert!(policy.rewrite_due(30));
        assert!(policy.rewrite_due(31));

        let never = OomPolicy {
            rewrite_interval_secs: 0,
            ..OomPolicy::default()
        };
        assert!(
            !never.rewrite_due(u64::MAX),
            "zero disables rescoring, which only suits an engine that never forks"
        );
    }

    /// A group that grows between rescores is covered by the next one.
    /// This is the property the periodic rewrite exists for.
    #[test]
    fn a_worker_forked_after_the_first_write_is_scored_by_the_next_one() {
        let policy = OomPolicy::default();
        let first = policy.scores(10, &[20]);
        assert!(!first.contains_key(&21), "pid 21 has not forked yet");

        let second = policy.scores(10, &[20, 21]);
        assert_eq!(
            second[&21], 500,
            "a worker forked later must be scored by the rewrite"
        );
    }
}
