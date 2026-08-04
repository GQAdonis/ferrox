//! Real (not guessed) Metal device detection, mirroring
//! `ferrox_cuda::capability::HardwareProfile`'s shape: a plain struct
//! that's always constructible, with GPU fields honestly zeroed/false
//! unless built with `--features metal` on macOS and a real device is
//! present.

/// Snapshot of Metal GPU availability. Always constructible; `available`
/// is only ever `true` when built with `--features metal` on macOS and
/// `MTLCreateSystemDefaultDevice()` actually returns a device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalProfile {
    pub available: bool,
    pub device_name: Option<String>,
}

impl MetalProfile {
    /// Probe the machine. Cheap: on macOS this just asks the system for
    /// the default Metal device's name, no buffers or kernels involved.
    pub fn detect() -> Self {
        #[cfg(feature = "metal")]
        {
            match crate::gpu::probe() {
                Some(name) => MetalProfile {
                    available: true,
                    device_name: Some(name),
                },
                None => MetalProfile::default(),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            MetalProfile::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics() {
        let _ = MetalProfile::detect();
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn without_metal_feature_profile_always_reports_unavailable() {
        let profile = MetalProfile::detect();
        assert!(!profile.available);
        assert_eq!(profile.device_name, None);
    }
}
