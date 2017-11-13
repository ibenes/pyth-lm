import unittest
import vocab

class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.example_line = "hello world of vocabularies !"

    def test_getitem(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(vocabulary['<unk>'], 0)

    def test_getitem_for_unknown(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(vocabulary['a'], 0)

    def test_addition_un_unks(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(vocabulary['world'], 0)

        vocabulary.add_from_text(self.example_line)
        self.assertNotEqual(vocabulary['world'], 0)

    def test_addition_is_unique(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_from_text(self.example_line)
        self.assertNotEqual(vocabulary['world'], vocabulary['hello'])
        self.assertNotEqual(vocabulary['of'], vocabulary['hello'])
        self.assertNotEqual(vocabulary['of'], vocabulary['!'])
        self.assertNotEqual(vocabulary['world'], vocabulary['!'])

    def test_lenght_of_empty(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        self.assertEqual(len(vocabulary), 1)

    def test_lenght_after_addition(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_from_text(self.example_line)
        self.assertEqual(len(vocabulary), 6)

    def test_add_word(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_word('hi')
        self.assertEqual(len(vocabulary), 2)

    def test_add_already_known_word(self):
        vocabulary = vocab.Vocabulary('<unk>', 0) 
        vocabulary.add_word('<unk>')
        self.assertEqual(len(vocabulary), 1)
