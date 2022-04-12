//! helper functions for the logging backend
use crate::util::find_project_root;
use flexi_logger::{self, writers::FileLogWriter, Duplicate, LogTarget, Logger};
use log::Level::Warn;
use std::fs;

/// Creates a logging backend
/// By default all logs with Info or higher are written to a logfile in folder logs.
/// All logs with level at least Info are also written to stdout.
/// Logs with level at least Error are also written to stderr.
///
/// logs can be written via log::{error!, warn!, info!, debug!, trace!}
pub fn init_logging() {
    let mut output_dir = find_project_root().unwrap();
    output_dir.push("logs");
    fs::create_dir(&output_dir).unwrap_or_else(|_| {});
    Logger::with_env_or_str("info")
        .format(flexi_logger::colored_opt_format)
        .log_target(LogTarget::Writer(Box::new(
            FileLogWriter::builder()
                .directory(output_dir)
                .format(flexi_logger::colored_opt_format)
                .try_build()
                .expect("Directory 'logs' does not exist"),
        )))
        .duplicate_to_stdout(Duplicate::Info)
        .duplicate_to_stderr(Duplicate::Error)
        .start()
        .unwrap_or_else(|error| panic!("Logging initialization failed: {}", error));
    log_panics::init();
}

/// Creates a logging backend for use in testing
/// By default all logs with Warn or higher are printed to stdout.
pub fn init_test_logging() {
    if !log::log_enabled!(Warn) {
        Logger::with_env_or_str("warn")
            .format(flexi_logger::colored_opt_format)
            .start()
            .unwrap_or_else(|error| panic!("Logging initialization failed: {}", error));
    }
}
