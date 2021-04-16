import unittest
import processing


class MyTestCase(unittest.TestCase):
    def test_concatenate_text_as_array(self):
        text = ["aa.", "bb bb.", "ccc ccc ccc.", "2"]
        actual = processing.concatenate_text_as_array(text)
        self.assertTrue("aa. bb bb. ccc ccc ccc. 2", actual)

    def test_parse_text_to_sentences(self):
        text = "Dani has apples. Maria has peaches. Dave! Come here, Dr. John. Sami?"
        self.assertEqual(5, len(processing.parse_text_to_sentences(text)))
        self.assertEqual("Come here, Dr. John.", processing.parse_text_to_sentences(text)[3])
        self.assertEqual(3, len(processing.parse_text_to_sentences(text, sentences_in_batch=2)))
        self.assertEqual("Sami?", processing.parse_text_to_sentences(text, sentences_in_batch=2)[2])

    def test_remove_punctuation(self):
        text = "Dani. are, mere? Pere! She's just beautiful— who?"
        self.assertEqual("Dani are mere Pere Shes just beautiful who", processing.remove_punctuation(text))

    def test_spit_in_tokens(self):
        text = "Dani are, multe. mere? pere!"
        self.assertEqual(["Dani", "are,", "multe.", "mere?", "pere!"], processing.split_in_tokens(text))

    def test_remove_stop_words(self):
        tokens = ["Alex", "goes", "to", "the", "park",
                  "Here", "he", "sees", "a", "monkey",
                  "It", "is", "big",
                  "Its", "tail", "is", "just", "pretty", "small",
                  "It's", "small"]
        expected = ["Alex", "goes", "park", "Here", "sees", "monkey", "It",
                    "big", "Its", "tail", "pretty", "small", "It's", "small"]
        self.assertEqual(expected, processing.remove_stop_words(tokens))

    def test_tokens_to_lower_case(self):
        tokens = ["Dani", "ana", "ANCU", "ABC!", "123"]
        self.assertEqual(["dani", "ana", "ancu", "abc!", "123"], processing.tokens_to_lower_case(tokens))

    def test_is_english(self):
        self.assertTrue(processing.is_english("dani"))
        self.assertTrue(processing.is_english("dani!@34423"))
        self.assertFalse(processing.is_english("αλλήλων"))
        self.assertFalse(processing.is_english("dani αλλήλων"))

    def test_transliterate_non_english_words(self):
        self.assertEqual(["dani"], processing.transliterate_non_english_words(["dani"]))
        self.assertEqual(["allelon"], processing.transliterate_non_english_words(["αλλήλων"]))
        self.assertEqual(["dani allelon"], processing.transliterate_non_english_words(["dani αλλήλων"]))

if __name__ == '__main__':
    unittest.main()
