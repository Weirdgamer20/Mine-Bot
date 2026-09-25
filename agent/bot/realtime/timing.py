import time
import os

class MonotonicDeadlineScheduler:
    """
    Precision monotonic deadline scheduler for 100 Hz real-time loops.
    Uses time.perf_counter_ns() to avoid clock drift and accumulative sleep errors.
    """

    def __init__(self, period_ns: int = 10_000_000):  # Default 10ms = 100 Hz
        self.period_ns = period_ns
        self.next_deadline_ns = time.perf_counter_ns() + self.period_ns
        self.missed_deadlines = 0

    def reset(self):
        self.next_deadline_ns = time.perf_counter_ns() + self.period_ns
        self.missed_deadlines = 0

    def sleep_until_next_deadline(self) -> float:
        """
        Sleeps until next deadline.
        Returns:
            elapsed_compute_ms: Duration spent computing before sleeping (ms).
        """
        now_ns = time.perf_counter_ns()
        compute_ns = now_ns - (self.next_deadline_ns - self.period_ns)
        compute_ms = compute_ns / 1_000_000.0

        remaining_ns = self.next_deadline_ns - now_ns

        if remaining_ns > 1_000_000:  # > 1ms: sleep coarsely, then busy-wait for sub-millisecond precision
            time.sleep((remaining_ns - 800_000) / 1_000_000_000.0)

        # Spin-lock / busy-wait the remaining fraction for microsecond precision
        while time.perf_counter_ns() < self.next_deadline_ns:
            pass

        # Check deadline adherence
        after_ns = time.perf_counter_ns()
        if after_ns > self.next_deadline_ns + 1_000_000:  # >1ms overshoot
            self.missed_deadlines += 1
            # Advance deadline from current time to prevent burst catch-up
            self.next_deadline_ns = after_ns + self.period_ns
        else:
            self.next_deadline_ns += self.period_ns

        return compute_ms
