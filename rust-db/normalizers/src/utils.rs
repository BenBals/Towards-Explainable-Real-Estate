use chrono::{Datelike, NaiveDate};

type Year = u16;
type QuarterNum = u8;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Quarter {
    year: Year,
    quarter: QuarterNum,
}

impl From<NaiveDate> for Quarter {
    fn from(date: NaiveDate) -> Self {
        Self {
            year: date.year() as Year,
            quarter: ((date.month() + 2) / 3) as QuarterNum,
        }
    }
}

/// This is a reasonable default for normalizers to normalize values to.
/// It's always *after* any date that passes [is_reasonable_date],
/// in this way the output of [days_until_reference_date] will all be positive.
pub fn reference_date() -> NaiveDate {
    NaiveDate::from_ymd(2021, 1, 1)
}

/// How many days from a given date until the reference date will be reached?
/// This function counts the `date`, but not the reference date.
/// # Panics
/// If you pass a date that is after the [reference_date].
pub fn days_until_reference_date(date: NaiveDate) -> f64 {
    assert!(date <= reference_date());
    reference_date().signed_duration_since(date).num_days() as f64
}

/// Check if a date is likely to be an outlier.
/// # Returns
/// - `true` if the date is after a set start date and [reference_date]
/// - `false` else
pub fn is_reasonable_date(date: NaiveDate) -> bool {
    date >= NaiveDate::from_ymd(2007, 1, 1) && date <= reference_date()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::NaiveDate;

    #[test]
    fn days_until_reference_date_examples() {
        assert_approx_eq!(
            days_until_reference_date(NaiveDate::from_ymd(2020, 12, 15)),
            17.0
        );
        assert_approx_eq!(
            days_until_reference_date(NaiveDate::from_ymd(2020, 1, 1)),
            // note that 2020 was a leap year
            366.0
        );
    }
}
