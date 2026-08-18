//! Stop-sequence matching over streamed text.
//!
//! OpenAI's `stop` is a list of strings, and a stop string can straddle token
//! boundaries. Emitting each token's text as it arrives would leak the first
//! half of a stop string to the client, so text whose tail could still grow
//! into one is held back until it either completes the match or cannot.

/// What one decoded chunk produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopStep {
    /// Text that is now known not to be part of a stop sequence.
    pub emit: String,
    /// A stop sequence completed; generation should end here and the sequence
    /// itself must not be emitted.
    pub matched: bool,
}

#[derive(Debug, Default)]
pub struct StopBuffer {
    stops: Vec<String>,
    /// Emitted text's tail, retained because it is a prefix of some stop
    /// sequence. Always shorter than the longest stop sequence.
    held: String,
}

impl StopBuffer {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops
                .iter()
                .filter(|stop| !stop.is_empty())
                .cloned()
                .collect(),
            held: String::new(),
        }
    }

    /// Add a decoded chunk and return the text that may be released.
    pub fn push(&mut self, chunk: &str) -> StopStep {
        if self.stops.is_empty() {
            return StopStep {
                emit: chunk.to_owned(),
                matched: false,
            };
        }
        self.held.push_str(chunk);
        // The earliest match wins, so a short stop sequence inside a longer
        // one still truncates at the position the client asked for.
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min()
        {
            let emit = self.held[..at].to_owned();
            self.held.clear();
            return StopStep {
                emit,
                matched: true,
            };
        }
        let hold = self.longest_partial_suffix();
        let split = self.held.len() - hold;
        let emit = self.held[..split].to_owned();
        self.held.drain(..split);
        StopStep {
            emit,
            matched: false,
        }
    }

    /// Release whatever is still held. Called once generation ends without a
    /// match, since held text is then just output.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Length in bytes of the longest suffix of the buffer that is a proper
    /// prefix of some stop sequence.
    fn longest_partial_suffix(&self) -> usize {
        let mut hold = 0;
        for stop in &self.stops {
            // A full match was already ruled out, so only shorter prefixes can
            // apply; the longest candidate is bounded by the buffer.
            let longest = (stop.len() - 1).min(self.held.len());
            for len in (hold + 1..=longest).rev() {
                if stop.is_char_boundary(len) && self.held.ends_with(&stop[..len]) {
                    hold = len;
                    break;
                }
            }
        }
        hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn passes_text_through_when_no_stops_are_configured() {
        let mut buffer = StopBuffer::new(&[]);
        let step = buffer.push("anything at all");
        assert_eq!(step.emit, "anything at all");
        assert!(!step.matched);
        assert!(buffer.flush().is_empty());
    }

    #[test]
    fn truncates_at_a_stop_inside_one_chunk() {
        let mut buffer = StopBuffer::new(&stops(&["END"]));
        let step = buffer.push("hello END world");
        assert_eq!(step.emit, "hello ");
        assert!(step.matched);
    }

    #[test]
    fn holds_a_partial_stop_across_chunks() {
        let mut buffer = StopBuffer::new(&stops(&["<|stop|>"]));
        let first = buffer.push("done <|st");
        assert_eq!(first.emit, "done ");
        assert!(!first.matched);
        let second = buffer.push("op|> more");
        assert_eq!(second.emit, "");
        assert!(second.matched);
    }

    #[test]
    fn releases_held_text_when_the_match_fails() {
        let mut buffer = StopBuffer::new(&stops(&["END"]));
        assert_eq!(buffer.push("go EN").emit, "go ");
        let step = buffer.push("D?");
        // "END" completed here, so only what precedes it is released.
        assert_eq!(step.emit, "");
        assert!(step.matched);

        let mut buffer = StopBuffer::new(&stops(&["END"]));
        assert_eq!(buffer.push("go EN").emit, "go ");
        let step = buffer.push("ough");
        assert_eq!(step.emit, "ENough");
        assert!(!step.matched);
        assert!(buffer.flush().is_empty());
    }

    #[test]
    fn flushes_a_trailing_partial_match() {
        let mut buffer = StopBuffer::new(&stops(&["END"]));
        assert_eq!(buffer.push("all done E").emit, "all done ");
        assert_eq!(buffer.flush(), "E");
        assert!(buffer.flush().is_empty());
    }

    #[test]
    fn earliest_match_wins_across_stops() {
        let mut buffer = StopBuffer::new(&stops(&["world", "lo w"]));
        let step = buffer.push("hello world");
        assert_eq!(step.emit, "hel");
        assert!(step.matched);
    }

    #[test]
    fn never_splits_a_multi_byte_character() {
        let mut buffer = StopBuffer::new(&stops(&["日本語"]));
        let first = buffer.push("読む日本");
        assert_eq!(first.emit, "読む");
        assert!(!first.matched);
        let second = buffer.push("語です");
        assert_eq!(second.emit, "");
        assert!(second.matched);
    }

    #[test]
    fn holds_the_longest_partial_across_stops() {
        let mut buffer = StopBuffer::new(&stops(&["ab", "xyz"]));
        let step = buffer.push("wxy");
        assert_eq!(step.emit, "w");
        assert!(!step.matched);
        let step = buffer.push("z");
        assert_eq!(step.emit, "");
        assert!(step.matched);
    }

    #[test]
    fn holds_only_what_a_stop_could_still_consume() {
        let mut buffer = StopBuffer::new(&stops(&["abc"]));
        // "ab" can grow into the stop, "xab" cannot as a whole.
        let step = buffer.push("xab");
        assert_eq!(step.emit, "x");
        assert!(!step.matched);
        let step = buffer.push("z");
        assert_eq!(step.emit, "abz");
        assert!(!step.matched);
    }
}
