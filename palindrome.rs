/// Checks if a string is a palindrome.
///
/// Ignores case, spaces, and non-alphanumeric characters.
/// An empty string or a string with no alphanumeric characters
/// is considered a palindrome.
pub fn is_palindrome(s: &str) -> bool {
    let cleaned: Vec<char> = s
        .chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();

    let len = cleaned.len();
    for i in 0..len / 2 {
        if cleaned[i] != cleaned[len - 1 - i] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_palindromes() {
        assert!(is_palindrome("racecar"));
        assert!(is_palindrome("level"));
        assert!(is_palindrome("madam"));
    }

    #[test]
    fn test_non_palindromes() {
        assert!(!is_palindrome("hello"));
        assert!(!is_palindrome("rust"));
        assert!(!is_palindrome("openai"));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(is_palindrome("RaceCar"));
        assert!(is_palindrome("LeVeL"));
        assert!(is_palindrome("Madam"));
    }

    #[test]
    fn test_with_spaces_and_punctuation() {
        assert!(is_palindrome("A man a plan a canal Panama"));
        assert!(is_palindrome("Was it a car or a cat I saw"));
        assert!(is_palindrome("No 'x' in Nixon"));
        assert!(!is_palindrome("Hello, world!"));
    }

    #[test]
    fn test_empty_and_single_char() {
        assert!(is_palindrome(""));
        assert!(is_palindrome("a"));
        assert!(is_palindrome("A"));
    }

    #[test]
    fn test_non_alphanumeric_only() {
        assert!(is_palindrome("!!!"));
        assert!(is_palindrome("   "));
        assert!(is_palindrome("..."));
    }

    #[test]
    fn test_long_palindrome() {
        let s = "abccba";
        assert!(is_palindrome(s));

        let s = "abcddcba";
        assert!(is_palindrome(s));

        let s = "12345678987654321";
        assert!(is_palindrome(s));
    }
}
