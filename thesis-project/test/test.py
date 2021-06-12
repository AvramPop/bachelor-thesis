import unittest
import processing.processing_utils as processing
import evo.evo_utils as evo
import graphs.graph_utils as graphs


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

    def test_remove_footnotes(self):
        text = "Dani said something.1 He2 is a cool guy. We3 would rather love him!4 \"Or go skying\"5 (Or not)6"
        self.assertEqual("Dani said something. He is a cool guy. We would rather love him! \"Or go skying\" (Or not)",
                         processing.remove_footnotes(text))

    def test_rouge_score(self):
        self.assertEqual(1, processing.rouge_score("a", "a")["ROUGE-1-F"])
        self.assertEqual(0, processing.rouge_score("a", "b")["ROUGE-1-F"])

    def test_number_of_sentences_in_text(self):
        text = "Dani has apples. Maria has peaches. Dave! Come here, Dr. John. Sami?"
        self.assertEqual(5, processing.number_of_sentences_in_text(text))

    def test_final_results(self):
        score1 = {"ROUGE-1-L": 0.25, "ROUGE-1-F": 0.5}
        score2 = {"ROUGE-1-L": 0.65, "ROUGE-1-F": 0.1}
        self.assertEqual({"ROUGE-1-L": 0.45, "ROUGE-1-F": 0.3}, processing.final_results([score1, score2]))

    def test_get_average_clustering_coefficient(self):
        coefficients_of_clusters = {1: 5, 2: 7}
        self.assertEqual(6.0, graphs.get_average_clustering_coefficient(coefficients_of_clusters))

    def test_best_from_cluster(self):
        coefficients_of_nodes = {1: 5, 2: 7}
        clusters = {0: [1, 2]}
        self.assertEqual(2, graphs.best_from_cluster(coefficients_of_nodes, clusters, 0))

    def test_remove_best(self):
        coefficients_of_nodes = {1: 5, 2: 7}
        clusters = {0: [1, 2]}
        graphs.remove_best(2, 0, clusters, coefficients_of_nodes)
        self.assertEqual(1, len(clusters[0]))

    def test_no_available_nodes(self):
        clusters = {0: [1, 2]}
        self.assertFalse(graphs.no_available_nodes(clusters))
        clusters = {}
        self.assertTrue(graphs.no_available_nodes(clusters))

    def test_bits_in_individual(self):
        individual = [0, 1, 0, 1]
        self.assertEqual(2, evo.bits_in_individual(individual))

    def test_max_weight_dag(self):
        graph = [[0, 5, 3, 0, 0, 0], [0, 0, 2, 6, 0, 0], [0, 0, 0, 7, 4, 2], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        self.assertEqual(15, evo.max_weight_dag(graph))

    def test_sentence_position(self):
        individual = [1, 1, 1, 1]
        self.assertEqual(1, evo.sentence_position(individual))

    def test_number_of_sentences_in_directory(self):
        self.assertEqual(2, processing.get_number_of_texts_in_folder('/home/dani/Desktop/licenta/bachelor-thesis/thesis-project/test/resources'))


if __name__ == '__main__':
    unittest.main()
