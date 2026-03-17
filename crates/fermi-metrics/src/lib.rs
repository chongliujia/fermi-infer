use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Default)]
pub struct Metrics {
    request_count: AtomicU64,
    request_error_count: AtomicU64,
    active_requests: AtomicU64,
    queue_wait_ms_total: AtomicU64,
    queue_wait_count: AtomicU64,
    ttft_ms_total: AtomicU64,
    ttft_count: AtomicU64,
    decode_tokens_total: AtomicU64,
    generation_ms_total: AtomicU64,
    generation_count: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_request(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.request_error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_queue_wait(&self, duration: Duration) {
        self.queue_wait_ms_total
            .fetch_add(duration_to_ms(duration), Ordering::Relaxed);
        self.queue_wait_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_ttft(&self, duration: Duration) {
        self.ttft_ms_total
            .fetch_add(duration_to_ms(duration), Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_generation(&self, decode_tokens: usize, duration: Duration) {
        self.decode_tokens_total
            .fetch_add(decode_tokens as u64, Ordering::Relaxed);
        self.generation_ms_total
            .fetch_add(duration_to_ms(duration), Ordering::Relaxed);
        self.generation_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn track_active_request(self: &Arc<Self>) -> ActiveRequestGuard {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        ActiveRequestGuard {
            metrics: Arc::clone(self),
        }
    }

    pub fn render_prometheus(&self) -> String {
        let request_count = self.request_count.load(Ordering::Relaxed);
        let request_error_count = self.request_error_count.load(Ordering::Relaxed);
        let active_requests = self.active_requests.load(Ordering::Relaxed);
        let queue_wait_ms_total = self.queue_wait_ms_total.load(Ordering::Relaxed);
        let queue_wait_count = self.queue_wait_count.load(Ordering::Relaxed);
        let ttft_ms_total = self.ttft_ms_total.load(Ordering::Relaxed);
        let ttft_count = self.ttft_count.load(Ordering::Relaxed);
        let decode_tokens_total = self.decode_tokens_total.load(Ordering::Relaxed);
        let generation_ms_total = self.generation_ms_total.load(Ordering::Relaxed);
        let generation_count = self.generation_count.load(Ordering::Relaxed);

        let avg_queue_wait_ms = average_ms(queue_wait_ms_total, queue_wait_count);
        let avg_ttft_ms = average_ms(ttft_ms_total, ttft_count);
        let tokens_per_second = if generation_ms_total == 0 {
            0.0
        } else {
            (decode_tokens_total as f64 * 1000.0) / generation_ms_total as f64
        };

        let mut out = String::new();
        push_counter(
            &mut out,
            "fermi_request_count",
            "Total number of handled inference requests.",
            request_count,
        );
        push_counter(
            &mut out,
            "fermi_request_error_count",
            "Total number of inference requests that ended in error.",
            request_error_count,
        );
        push_gauge(
            &mut out,
            "fermi_active_requests",
            "Number of in-flight generation tasks.",
            active_requests as f64,
        );
        push_counter(
            &mut out,
            "fermi_queue_wait_ms_total",
            "Total queue wait time in milliseconds before acquiring an engine slot.",
            queue_wait_ms_total,
        );
        push_counter(
            &mut out,
            "fermi_queue_wait_count",
            "Number of queue wait observations.",
            queue_wait_count,
        );
        push_gauge(
            &mut out,
            "fermi_queue_wait_ms_avg",
            "Average queue wait time in milliseconds.",
            avg_queue_wait_ms,
        );
        push_counter(
            &mut out,
            "fermi_ttft_ms_total",
            "Total time-to-first-token in milliseconds.",
            ttft_ms_total,
        );
        push_counter(
            &mut out,
            "fermi_ttft_count",
            "Number of TTFT observations.",
            ttft_count,
        );
        push_gauge(
            &mut out,
            "fermi_ttft_ms_avg",
            "Average time-to-first-token in milliseconds.",
            avg_ttft_ms,
        );
        push_counter(
            &mut out,
            "fermi_decode_tokens_total",
            "Total number of generated completion tokens.",
            decode_tokens_total,
        );
        push_counter(
            &mut out,
            "fermi_generation_ms_total",
            "Total generation time in milliseconds.",
            generation_ms_total,
        );
        push_counter(
            &mut out,
            "fermi_generation_count",
            "Number of completed generation runs.",
            generation_count,
        );
        push_gauge(
            &mut out,
            "fermi_tokens_per_second_avg",
            "Average generated tokens per second.",
            tokens_per_second,
        );
        out
    }
}

pub struct ActiveRequestGuard {
    metrics: Arc<Metrics>,
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.metrics.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

fn duration_to_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn average_ms(total_ms: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total_ms as f64 / count as f64
    }
}

fn push_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push_str(" counter\n");
    out.push_str(name);
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

fn push_gauge(out: &mut String, name: &str, help: &str, value: f64) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push_str(" gauge\n");
    out.push_str(name);
    out.push(' ');
    out.push_str(&format!("{value:.3}"));
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_expected_metric_names() {
        let metrics = Arc::new(Metrics::new());
        metrics.record_request();
        metrics.record_error();
        metrics.observe_queue_wait(Duration::from_millis(5));
        metrics.observe_ttft(Duration::from_millis(8));
        metrics.observe_generation(12, Duration::from_millis(24));
        let body = metrics.render_prometheus();

        assert!(body.contains("fermi_request_count"));
        assert!(body.contains("fermi_request_error_count"));
        assert!(body.contains("fermi_active_requests"));
        assert!(body.contains("fermi_ttft_ms_avg"));
        assert!(body.contains("fermi_tokens_per_second_avg"));
    }

    #[test]
    fn active_request_guard_updates_gauge() {
        let metrics = Arc::new(Metrics::new());
        {
            let _guard = metrics.track_active_request();
            let body = metrics.render_prometheus();
            assert!(body.contains("fermi_active_requests 1.000"));
        }
        let body = metrics.render_prometheus();
        assert!(body.contains("fermi_active_requests 0.000"));
    }
}
