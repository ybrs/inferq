//! Linux host and process telemetry used by repeatable benchmark artifacts.

use std::{collections::BTreeSet, fs, path::Path, process::Command};

use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct HostInfo {
    pub cpu_model: Option<String>,
    pub logical_cpus: usize,
    pub physical_cores: Option<usize>,
    pub memory_total_bytes: Option<u64>,
    pub kernel: Option<String>,
    pub rustc: Option<String>,
    pub filesystem: Option<String>,
    pub storage_source: Option<String>,
    pub storage_rotational: Option<bool>,
}

impl HostInfo {
    pub fn detect(model_path: &Path) -> Self {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok();
        let cpu_model = cpuinfo
            .as_deref()
            .and_then(|text| value_after_colon(text, "model name").map(ToOwned::to_owned));
        let physical_cores = cpuinfo.as_deref().and_then(physical_core_count);
        let memory_total_bytes = fs::read_to_string("/proc/meminfo")
            .ok()
            .as_deref()
            .and_then(|text| kib_value(text, "MemTotal:"));
        let mount = command_output(
            "findmnt",
            &[
                "-n",
                "-o",
                "SOURCE,FSTYPE",
                "-T",
                &model_path.display().to_string(),
            ],
        );
        let (storage_source, filesystem) = mount
            .as_deref()
            .and_then(|line| line.split_once(' '))
            .map(|(source, fs)| (Some(source.to_owned()), Some(fs.trim().to_owned())))
            .unwrap_or((None, None));
        let storage_rotational = storage_source.as_deref().and_then(|source| {
            command_output("lsblk", &["-dn", "-o", "ROTA", source]).and_then(|value| {
                match value.trim() {
                    "0" => Some(false),
                    "1" => Some(true),
                    _ => None,
                }
            })
        });
        Self {
            cpu_model,
            logical_cpus: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            physical_cores,
            memory_total_bytes,
            kernel: command_output("uname", &["-srmo"]),
            rustc: command_output("rustc", &["--version"]),
            filesystem,
            storage_source,
            storage_rotational,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ProcessSnapshot {
    pub resident_bytes: Option<u64>,
    pub peak_resident_bytes: Option<u64>,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
    pub read_bytes: Option<u64>,
    pub write_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessDelta {
    pub resident_bytes_before: Option<u64>,
    pub resident_bytes_after: Option<u64>,
    pub peak_resident_bytes: Option<u64>,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
    pub read_bytes: Option<u64>,
    pub write_bytes: Option<u64>,
}

impl ProcessSnapshot {
    pub fn capture() -> Result<Self> {
        let status =
            fs::read_to_string("/proc/self/status").context("failed to read /proc/self/status")?;
        let stat =
            fs::read_to_string("/proc/self/stat").context("failed to read /proc/self/stat")?;
        let io = fs::read_to_string("/proc/self/io").context("failed to read /proc/self/io")?;
        let (minor_faults, major_faults) = parse_faults(&stat);
        Ok(Self {
            resident_bytes: kib_value(&status, "VmRSS:"),
            peak_resident_bytes: kib_value(&status, "VmHWM:"),
            minor_faults,
            major_faults,
            read_bytes: integer_value(&io, "read_bytes:"),
            write_bytes: integer_value(&io, "write_bytes:"),
        })
    }

    pub fn delta(&self, after: &Self) -> ProcessDelta {
        ProcessDelta {
            resident_bytes_before: self.resident_bytes,
            resident_bytes_after: after.resident_bytes,
            peak_resident_bytes: after.peak_resident_bytes,
            minor_faults: difference(self.minor_faults, after.minor_faults),
            major_faults: difference(self.major_faults, after.major_faults),
            read_bytes: difference(self.read_bytes, after.read_bytes),
            write_bytes: difference(self.write_bytes, after.write_bytes),
        }
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn value_after_colon<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name.trim() == key).then(|| value.trim())
    })
}

fn integer_value(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        line.strip_prefix(key)
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
    })
}

fn kib_value(text: &str, key: &str) -> Option<u64> {
    integer_value(text, key).map(|value| value * 1024)
}

fn difference(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    Some(after?.saturating_sub(before?))
}

fn parse_faults(stat: &str) -> (Option<u64>, Option<u64>) {
    let Some((_, fields)) = stat.rsplit_once(") ") else {
        return (None, None);
    };
    // `fields` begins at proc(5) field 3 (state). minflt and majflt are
    // fields 10 and 12, so their zero-based indexes here are 7 and 9.
    let fields: Vec<_> = fields.split_whitespace().collect();
    (
        fields.get(7).and_then(|value| value.parse().ok()),
        fields.get(9).and_then(|value| value.parse().ok()),
    )
}

fn physical_core_count(cpuinfo: &str) -> Option<usize> {
    let mut pairs = BTreeSet::new();
    for processor in cpuinfo.split("\n\n") {
        if let (Some(physical), Some(core)) = (
            value_after_colon(processor, "physical id"),
            value_after_colon(processor, "core id"),
        ) {
            pairs.insert((physical.to_owned(), core.to_owned()));
        }
    }
    (!pairs.is_empty()).then_some(pairs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_process_fault_fields() {
        let stat = "42 (bench worker) R 1 2 3 4 5 6 101 8 202 9";
        assert_eq!(parse_faults(stat), (Some(101), Some(202)));
    }

    #[test]
    fn process_delta_saturates_counters() {
        let before = ProcessSnapshot {
            minor_faults: Some(10),
            read_bytes: Some(100),
            ..Default::default()
        };
        let after = ProcessSnapshot {
            minor_faults: Some(15),
            read_bytes: Some(90),
            ..Default::default()
        };
        let delta = before.delta(&after);
        assert_eq!(delta.minor_faults, Some(5));
        assert_eq!(delta.read_bytes, Some(0));
    }

    #[test]
    fn parses_status_values_with_kib_suffix() {
        assert_eq!(kib_value("VmRSS:\t  123 kB\n", "VmRSS:"), Some(125_952));
    }
}
